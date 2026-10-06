"""Проверка агента на настоящей руке. Учился он в упрощённой среде (рука телепортом над точкой), а тут всё как
у baseline 11/11: одна рука, едет из домашней позы моим планировщиком (лесенка с проверкой столкновений, top_plan.py)
на 10 см над точкой, потом спуск, сжатие, подъём, держим 1 с. Отличие от baseline одно: куда и под каким углом
хватать, решает Q-карта (веса из конца попытки 2), а не правило по глубине.

    bash server/run.sh --timeout 100m tmp_rl3/eval_arm.py <gpu> [--n 20] [--obj tetrapak can] [--seed 300]

Карту высот для сети строю из кадра глубины этой сцены (камера 640x480) по той же сетке 128x128 над столом,
что в fly_env.py. RGB сеть не видит (нули), как на обучении. Сид другой, чем на обучении - раскладки новые.
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20, help="сколько эпизодов на объект")
parser.add_argument("--obj", nargs="*", default=["tetrapak", "can"])
parser.add_argument("--seed", type=int, default=300)
parser.add_argument("--video", type=int, default=2, help="для скольких первых эпизодов на объект писать gif сбоку")
parser.add_argument("--weights", default=os.path.expanduser("~/grasp_task/ckpt/qmap_d/last.pt"))
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUT, save_gif
from top_plan import PlanEnv

# --- то же, что в fly_env.py: рабочая зона, сетка 128x128, 8 углов
WS_X, WS_Y, IMG, N_YAW = (-0.35, 0.35), (0.15, 0.85), 128, 8
CAM_Z, CAM_Y = 1.2, 0.5
GRIP_DEPTH, MIN_Z = 0.03, 0.015
ARM_Y = 0.29  # полосу у кисти робота выкидываю, как в train_common.object_pixels


# --- сеть: копия QMap из попытки 2 (train_qmap.py), импортить тот файл нельзя - он при импорте сам стартует обучение
def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class QMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1, self.d2, self.d3, self.mid = block(4, 32), block(32, 64), block(64, 128), block(128, 128)
        self.u3, self.u2, self.u1 = block(256, 64), block(128, 32), block(64, 32)
        self.out = nn.Conv2d(32, N_YAW, 1)

    def forward(self, img):
        x = img.float() / 255.0
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)  # RGB - нули, как в попытке 2
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return self.out(u)


def height_map(depth, dev):
    """кадр глубины [H,W] -> высоты над столом [128,128] по сетке fly_env (пересчёт под разрешение этой камеры)"""
    H, W = depth.shape
    fx = W * 18.0 / 20.955
    xs = torch.linspace(WS_X[0], WS_X[1], IMG, device=dev)
    ys = torch.linspace(WS_Y[1], WS_Y[0], IMG, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u = W / 2 + gx * fx / CAM_Z
    v = H / 2 - (gy - CAM_Y) * fx / CAM_Z
    grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], dim=-1).unsqueeze(0)
    d = torch.as_tensor(depth, device=dev, dtype=torch.float32)
    h = torch.where(d > 0, CAM_Z - d, torch.zeros_like(d))[None, None]
    h = F.max_pool2d(F.grid_sample(h, grid, align_corners=False), 5, stride=1, padding=2)
    return h[0, 0].clamp(0, 0.25)


def pix_to_xy(i, j):
    x = WS_X[0] + (j + 0.5) / IMG * (WS_X[1] - WS_X[0])
    y = WS_Y[1] - (i + 0.5) / IMG * (WS_Y[1] - WS_Y[0])
    return x, y


def choose(net, height):
    """жадный выбор Q-карты только среди клеток на объекте. Вернёт x, y, z, yaw, и шанс по сети"""
    img = torch.zeros(1, 4, IMG, IMG, dtype=torch.uint8, device=height.device)
    img[0, 3] = (height / 0.25 * 255).clamp(0, 255).to(torch.uint8)
    on_obj = img[0, 3] > int(0.01 / 0.25 * 255)
    rows = torch.arange(IMG, device=height.device).float()
    y_row = WS_Y[1] - (rows + 0.5) / IMG * (WS_Y[1] - WS_Y[0])
    on_obj[y_row < ARM_Y, :] = False
    if on_obj.sum() == 0:
        return None
    with torch.no_grad():
        logits = net(img)[0]
    logits = logits.masked_fill(~on_obj.unsqueeze(0), float("-inf"))
    flat = int(logits.flatten().argmax())
    k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
    i, j = rest // IMG, rest % IMG
    x, y = pix_to_xy(i, j)
    z = max(float(height[i, j]) - GRIP_DEPTH, MIN_Z)
    return x, y, z, k * math.pi / N_YAW, float(torch.sigmoid(logits[k, i, j]))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def main():
    t = time.time()
    env = PlanEnv(seed=args.seed, video=args.video > 0)
    dev = env.dev
    net = QMap().to(dev)
    st = torch.load(args.weights, map_location="cpu")
    net.load_state_dict(st["net"])
    net.eval()
    print(f"env ready in {time.time() - t:.1f}s, веса {args.weights} (попытка {st['attempt']}, {st.get('hours', 0):.1f} ч)", flush=True)

    stats, log = {}, []
    t_all = time.time()
    for name in args.obj:
        ok = 0
        for k in range(args.n):
            t = time.time()
            obs = env.reset(obj=name)
            tr = obs["true"]
            true_x, true_y = env.oracle()[:2]  # только для лога
            g = choose(net, height_map(obs["depth"], dev))
            env.set_obstacle(obs["depth"])
            if g is None:
                res = {"success": False, "rise": 0.0, "knocked": False, "fail": "не видно объекта"}
                err = float("nan")
            else:
                env.rec = [] if k < args.video else None
                res = env.step_plan(*g[:4])
                if env.rec:
                    save_gif(env.rec, f"{name}_{k:02d}_{tr['pose']}_{'ok' if res['success'] else 'fail'}.gif", ms=50)
                env.rec = None
                err = math.hypot(g[0] - true_x, g[1] - true_y)
            ok += res["success"]
            print(
                f"{name:8s} #{k:02d} {tr['pose']:9s} "
                + ("НЕ ВИДНО" if g is None else f"grasp ({g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}) yaw {math.degrees(g[3]):+4.0f} P {g[4]:.2f}")
                + f"  до центра {err * 1000:5.1f} мм  rise {res['rise'] * 100:5.1f} см  {res.get('fail', '')}"
                + f"{'OK' if res['success'] else 'fail'}  план {res.get('plan_s', 0):.0f} с  {time.time() - t:.0f} с",
                flush=True,
            )
            log.append({"obj": name, **tr, "grasp": None if g is None else list(g), "dist_to_center": err,
                        **{a: b for a, b in res.items() if a not in ("contact_before_close", "hits")},
                        "hits": res.get("hits"), "contact_before_close": list(res.get("contact_before_close", ("", 0.0)))})
        stats[name] = ok
        lo, hi = wilson(ok, args.n)
        print(f"--- {name}: {ok}/{args.n} = {ok / args.n:.0%}  95% CI [{lo:.0%}, {hi:.0%}]", flush=True)

    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    print(f"SR на руке: {stats}; {(time.time() - t_all) / (args.n * len(stats)):.0f} с на эпизод", flush=True)


if __name__ == "__main__":
    main()
    app.close()
