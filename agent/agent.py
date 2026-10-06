"""Мой агент (Q-карта) и посмотреть на него без симулятора. Нужны только python, torch, numpy, matplotlib.

    python agent/agent.py                              # несколько готовых сцен
    python agent/agent.py can 0.1 0.5                  # поставить банку в (0.1, 0.5) метра
    python agent/agent.py tetrapak-lying -0.1 0.45 30  # лежачий тетрапак, длинная сторона под 30 градусов
    python agent/agent.py can 0.1 0.5 + tetrapak 0 0.6 # несколько сцен через +

Объекты: tetrapak (стоит), tetrapak-lying, can, chips. Координаты - в метрах по столу, база робота в нуле,
x вбок (от -0.3 до 0.3), y вперёд от робота (от 0.35 до 0.75 нормально). Угол - куда смотрит длинная сторона, градусы.
Карту высот рисую сам по размерам объекта (как её видела бы камера сверху) и отдаю сети. Физики тут нет -
возьмёт ли рука на самом деле, проверяется в sim.py.

Сеть смотрит на карту высот 128x128 над рабочей зоной стола 70x70 см и для каждой клетки и каждого из 8 углов
говорит, какой шанс, что схват сверху туда поднимет объект. Хватаю туда, где шанс больше всего, но только по клеткам
на объекте. Этот же файл использует sim.py в симуляторе.
"""

import math
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WS_X, WS_Y = (-0.35, 0.35), (0.15, 0.85)  # рабочая зона в осях стола, метры (база робота в нуле)
IMG, N_YAW = 128, 8
GRIP_DEPTH, MIN_Z = 0.03, 0.015  # хватаю на 3 см ниже верха, но не ниже 1.5 см над столом
ARM_Y = 0.29  # ближе к роботу на снимке висит кисть - туда не хватаю
RU = {"tetrapak": "тетрапак", "can": "банка", "chips": "чипсы"}


# ---------------- сеть


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class QMap(nn.Module):
    """маленький U-Net. Входов 4 (так училась первая попытка с RGB-D), но RGB тут всегда нули - смотрит только высоты"""

    def __init__(self):
        super().__init__()
        self.d1, self.d2, self.d3, self.mid = block(4, 32), block(32, 64), block(64, 128), block(128, 128)
        self.u3, self.u2, self.u1 = block(256, 64), block(128, 32), block(64, 32)
        self.out = nn.Conv2d(32, N_YAW, 1)

    def forward(self, img):
        x = img.float() / 255.0
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return self.out(u)  # [B,8,128,128] логиты


def load(path=os.path.join(HERE, "qmap_depth.pt"), dev="cpu"):
    net = QMap()
    net.load_state_dict(torch.load(path, map_location="cpu")["net"])
    return net.to(dev).eval()


def pix_to_xy(i, j):
    return WS_X[0] + (j + 0.5) / IMG * (WS_X[1] - WS_X[0]), WS_Y[1] - (i + 0.5) / IMG * (WS_Y[1] - WS_Y[0])


def xy_to_pix(x, y):
    j = min(max(int((x - WS_X[0]) / (WS_X[1] - WS_X[0]) * IMG), 0), IMG - 1)
    i = min(max(int((WS_Y[1] - y) / (WS_Y[1] - WS_Y[0]) * IMG), 0), IMG - 1)
    return i, j


@torch.no_grad()
def choose(net, heights):
    """heights [B,128,128] метры над столом -> на каждый стол (x, y, z, yaw, шанс по сети, карта шансов) или None"""
    heights = heights.to(next(net.parameters()).device)
    B = heights.shape[0]
    img = torch.zeros(B, 4, IMG, IMG, dtype=torch.uint8, device=heights.device)
    img[:, 3] = (heights / 0.25 * 255).clamp(0, 255).to(torch.uint8)
    on_obj = img[:, 3] > int(0.01 / 0.25 * 255)
    rows = torch.arange(IMG, device=heights.device).float()
    on_obj[:, (WS_Y[1] - (rows + 0.5) / IMG * (WS_Y[1] - WS_Y[0])) < ARM_Y, :] = False
    logits = net(img)
    prob = torch.sigmoid(logits)
    out = []
    for b in range(B):
        if on_obj[b].sum() == 0:
            out.append(None)
            continue
        flat = int(logits[b].masked_fill(~on_obj[b].unsqueeze(0), float("-inf")).flatten().argmax())
        k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
        i, j = rest // IMG, rest % IMG
        x, y = pix_to_xy(i, j)
        z = max(float(heights[b, i, j]) - GRIP_DEPTH, MIN_Z)
        out.append((x, y, z, k * math.pi / N_YAW, float(prob[b, k, i, j]), prob[b].max(0).values.cpu()))
    return out


# ---------------- картинки и цифры


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def scenes_png(heights, picks, titles, path, true_xy=None, cols=4):
    """на каждую сцену два ряда: карта высот с крестиком (куда хватает, палка - вдоль чего сойдутся пальцы,
    зелёная точка - настоящий центр объекта) и тепловая карта сети поверх высот"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(heights)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(2 * rows, cols, figsize=(3 * cols, 6 * rows), squeeze=False)
    for a in axs.flat:
        a.axis("off")
    for k in range(n):
        r, c = 2 * (k // cols), k % cols
        h = heights[k].cpu().numpy() * 100
        axs[r, c].imshow(h, cmap="gray", vmin=0, vmax=20)
        axs[r + 1, c].imshow(h, cmap="gray", vmin=0, vmax=20)
        g = picks[k]
        if g is not None:
            i, j = xy_to_pix(g[0], g[1])
            axs[r, c].plot(j, i, "+", color="red", ms=12, mew=2)
            dj, di = 12 * math.cos(g[3]), -12 * math.sin(g[3])
            axs[r, c].plot([j - dj, j + dj], [i - di, i + di], color="red", lw=2)
            axs[r + 1, c].imshow(g[5].numpy(), cmap="jet", alpha=0.55, vmin=0, vmax=1)
        if true_xy is not None:
            ti, tj = xy_to_pix(*true_xy[k])
            axs[r, c].plot(tj, ti, "o", color="lime", ms=4)
        axs[r, c].set_title(titles[k], fontsize=8)
    fig.suptitle("сверху: карта высот (белое - высокое) и куда хватает агент; снизу: где он думает, что возьмёт (красное - уверен)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------- без симулятора: ставишь объект сам

# размеры, как в сцене: (форма, длинная сторона, короткая сторона, высота), метры
SHAPES = {
    "tetrapak": ("box", 0.06, 0.06, 0.19),
    "tetrapak-lying": ("box", 0.19, 0.06, 0.06),
    "can": ("circle", 0.08, 0.08, 0.038),
    "chips": ("box", 0.22, 0.15, 0.05),
}
PRESETS = [("can", 0.0, 0.5, 0), ("tetrapak", -0.12, 0.45, 20), ("tetrapak-lying", 0.1, 0.55, 30),
           ("tetrapak-lying", -0.05, 0.6, 100), ("can", 0.2, 0.4, 0), ("chips", 0.0, 0.5, 15)]


def fake_heights(obj, x, y, yaw_deg):
    """карта высот [128,128]: объект на пустом столе + кисть робота внизу кадра, как на настоящих снимках"""
    shape, a, b, h = SHAPES[obj]
    ys = torch.linspace(WS_Y[1], WS_Y[0], IMG)
    xs = torch.linspace(WS_X[0], WS_X[1], IMG)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    yaw = math.radians(yaw_deg)
    u = (gx - x) * math.cos(yaw) + (gy - y) * math.sin(yaw)  # вдоль длинной стороны
    v = -(gx - x) * math.sin(yaw) + (gy - y) * math.cos(yaw)  # поперёк
    inside = (u.abs() <= a / 2) & (v.abs() <= b / 2) if shape == "box" else (u**2 + v**2) <= (a / 2) ** 2
    hm = torch.where(inside, torch.tensor(h), torch.tensor(0.0))
    arm = (gx.abs() < 0.08) & (gy < 0.27)  # кисть в домашней позе висит над ближним краем стола
    hm = torch.where(arm, torch.tensor(0.25), hm)
    return F.max_pool2d(hm[None, None], 5, stride=1, padding=2)[0, 0]  # как в среде


def put(scenes):
    net = load()
    heights = torch.stack([fake_heights(*sc) for sc in scenes])
    picks = choose(net, heights)
    titles = []
    for sc, g in zip(scenes, picks):
        obj, x, y, yaw = sc
        if g is None:
            print(f"{obj} в ({x}, {y}): объект не виден агенту (слишком близко к роботу или вне стола)")
            titles.append(f"{obj}: не виден")
            continue
        dist = math.hypot(g[0] - x, g[1] - y) * 100
        # угол пальцев относительно длинной стороны (0 и 180 одно и то же). Сжимать надо поперёк, т.е. ~90
        rel = (math.degrees(g[3]) - yaw) % 180
        rel = min(rel, 180 - rel)
        print(f"{obj:15s} в ({x:+.2f}, {y:.2f}), угол {yaw:4.0f}:  хватаю ({g[0]:+.3f}, {g[1]:.3f}) на высоте {g[2] * 100:.1f} см, "
              f"{dist:.1f} см от центра, пальцы под {rel:.0f} град к длинной стороне, шанс по сети {g[4]:.2f}")
        titles.append(f"{obj}: шанс {g[4]:.2f}, {dist:.1f} см от центра")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    path = os.path.join(HERE, "out", "put.png")
    scenes_png(heights, picks, titles, path, true_xy=[(sc[1], sc[2]) for sc in scenes], cols=min(len(scenes), 3))
    print(f"\nкартинка: {path}")


def parse(argv):
    if not argv:
        return PRESETS
    scenes = []
    for chunk in " ".join(argv).split("+"):
        p = chunk.split()
        if p[0] not in SHAPES or len(p) < 3:
            raise SystemExit(f"не понял '{chunk.strip()}'. Надо: <{'|'.join(SHAPES)}> x y [угол]")
        scenes.append((p[0], float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 else 0.0))
    return scenes


if __name__ == "__main__":
    import sys

    put(parse(sys.argv[1:]))
