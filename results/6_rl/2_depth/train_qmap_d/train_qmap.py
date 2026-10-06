"""Q-карта, попытка 2: то же, что ночью 05.10 (1_rgbd/night_qmap), но
- RGB на вход сети больше не идёт: на его место всегда нули, сеть смотрит только на карту высот.
  Ночью цвет был от прошлой сцены и почти белый, сеть целилась по призракам (см. 2_depth/README.md)
- стартую не с нуля, а с ночного бэкапа qmap (веса + буфер): где объект, сеть по глубине уже выучила
- eps снова 0.5 и опускаю его гораздо медленнее: до 0.1 за первые 8 часов (по часам, не по попыткам), всего учу 12 ч.
  Ночью eps дошёл до 0.1 за 3000 попыток, и сеть почти не перебирала точки и углы на объекте
- жадный выбор только среди клеток на объекте (выше стола и не у кисти робота) - ночью тыкала в угол стола и кисть
- 8 столов вместо 4, камера 320x240, фазы попытки короче (это в fly_env.py) - чтоб попыток в час было больше
- проверки раз в час по 32 попытки (было раз в 30 мин по 8), видео на каждой второй
- в train.csv пишу ещё настоящий центр и yaw объекта, чтоб потом разобрать, куда промахивается

Остальное как было. Сеть смотрит на стол 128x128 и для каждой клетки и каждого из 8 yaw говорит, какой шанс,
что схват сверху туда поднимет объект. Хватаю туда, где шанс больше всего. Учу BCE в одной клетке после попытки.
По сути Q-learning с эпизодом из одного шага (гамма = 0). Идея из Zeng et al. 2018 (Visual Pushing-Grasping).

    bash server/run.sh --timeout 14h tmp_rl2/train_qmap.py <gpu> --name qmap_d

Падать можно: следующий запуск с тем же --name продолжит со своего бэкапа (~/grasp_task/ckpt/<name>).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="qmap_d")
parser.add_argument("--init-from", default="qmap", help="если своего бэкапа нет - взять веса и буфер из этого (ночной qmap)")
parser.add_argument("--eps-start", type=float, default=0.5)
parser.add_argument("--eps-end", type=float, default=0.1)
parser.add_argument("--eps-hours", type=float, default=8.0, help="за сколько часов обучения eps доходит до eps-end, дальше держу")
parser.add_argument("--hours", type=float, default=12.0, help="сколько всего учиться, с учётом прошлых запусков")
parser.add_argument("--stop-after-min", type=float, default=0, help="для проверок: выйти через столько минут этого запуска")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-min", type=float, default=60, help="как часто проверка жадной политикой (4 раунда = 32 попытки на 8 столах) + картинка карт")
parser.add_argument("--video-every", type=int, default=2, help="видео на каждой N-й проверке (видео дорогое)")
parser.add_argument("--ckpt-min", type=float, default=10)
parser.add_argument("--warmup", type=int, default=10, help="сколько рендеров перед снимком. rgb_check 05.10: 2 - кадр от прошлой сцены, 4 - каша, 8+ - норм")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

app = make_app()

import torch
import torch.nn as nn
import torch.nn.functional as F

import viz
from fly_env import IMG, N_YAW, FlyEnv
from train_common import Buffer, Run, object_pixels, random_on_object, to_kij

# --- сеть: маленький U-Net. Вход 4 канала (RGB + высота), выход 8 каналов (по одному на yaw) того же размера


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class QMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = block(4, 32)  # 128
        self.d2 = block(32, 64)  # 64
        self.d3 = block(64, 128)  # 32
        self.mid = block(128, 128)  # 16
        self.u3 = block(128 + 128, 64)
        self.u2 = block(64 + 64, 32)
        self.u1 = block(32 + 32, 32)
        self.out = nn.Conv2d(32, N_YAW, 1)

    def forward(self, img):
        x = img.float() / 255.0
        # попытка 2: RGB выкидываю, на его место нули. Каналов оставил 4, чтоб подошли ночные веса
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return self.out(u)  # [B,8,128,128] логиты, sigmoid - вероятность успеха


def greedy(net, img):
    """лучшая клетка и yaw по сети. Попытка 2: только среди клеток на объекте (выше стола на 1+ см и не у кисти
    робота, та же маска, что для случайных тычков). Ночью сеть тратила жадные попытки на угол стола и кисть"""
    with torch.no_grad():
        logits = net(img)
    on_obj = object_pixels(img)  # [N,128,128]
    on_obj[on_obj.flatten(1).sum(1) == 0] = True  # объекта не видно - тогда куда угодно
    masked = logits.masked_fill(~on_obj.unsqueeze(1), float("-inf"))
    flat = masked.flatten(1).argmax(dim=1)
    k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
    i, j = rest // IMG, rest % IMG
    x, y = FlyEnv.pix_to_xy(i, j)
    return x, y, k.float() * torch.pi / N_YAW, torch.sigmoid(logits)


def eps_at(hours):
    # сколько случайных попыток: сначала eps-start, за eps-hours часов обучения ровно до eps-end, дальше так и держу.
    # По часам, а не по попыткам, чтоб не зависело от скорости
    a = min(1.0, hours / args.eps_hours)
    return args.eps_start + (args.eps_end - args.eps_start) * a


def main():
    torch.manual_seed(args.seed)
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=True, warmup=args.warmup)
    dev = env.dev
    net = QMap().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    buf = Buffer()
    run = Run(args.name)
    attempt = 0
    st = run.load(buf)
    if st is not None:
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        attempt = st["attempt"]
    elif args.init_from:
        # своего бэкапа нет - стартую с ночного: веса и весь опыт. Счётчик попыток и eps начинаю заново
        src = os.path.join(os.path.dirname(run.dir), args.init_from)
        old = torch.load(os.path.join(src, "last.pt"), map_location="cpu")
        net.load_state_dict(old["net"])
        opt.load_state_dict(old["opt"])
        buf.load(torch.load(os.path.join(src, "buffer.pt"), map_location="cpu"))
        print(f"[init] взял веса и буфер из {src}: там было {old['attempt']} попыток, в буфере {buf.n}", flush=True)
    t_this = time.time()
    first_eval = True

    n_eval = [0]

    def evaluate():
        """4 раунда жадно, без случайности (на 8 столах это 32 попытки - ночью было 8, интервалы выходили огромные).
        В первом раунде картинка карт, видео первого стола - только на каждой video-every-й проверке, оно дорогое"""
        stats = {}
        video = n_eval[0] % args.video_every == 0
        n_eval[0] += 1
        for rnd in range(4):
            obs = env.reset()
            x, y, yaw, prob = greedy(net, obs["img"].to(dev))
            if rnd == 0:
                tag = f"{attempt:06d}"
                titles = [f"{obs['obj'][k]}" for k in range(4)]
                viz.qmap_png(obs["img"][:4], prob[:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(run.viz, f"qmap_{tag}.png"), titles)
                env.rec = [] if video else None
            r, _ = env.grasp(x, y, yaw)
            if env.rec:
                viz.save_gif(env.rec, os.path.join(run.viz, f"video_{tag}_{obs['obj'][0]}_{'ok' if r[0] > 0 else 'fail'}.gif"))
            env.rec = None
            for k in range(env.N):
                s = stats.setdefault(obs["obj"][k], [0, 0])
                s[0] += int(r[k].item())
                s[1] += 1
        run.log_eval(attempt, stats)
        print(f"[eval] попытка {attempt}: " + ", ".join(f"{o} {a}/{b}" for o, (a, b) in stats.items()), flush=True)
        if os.path.exists(run.train_csv):
            viz.curves_png(run.train_csv, run.eval_csv, os.path.join(run.viz, "curves.png"), "Q-карта")

    while run.hours() < args.hours:
        if args.stop_after_min and time.time() - t_this > args.stop_after_min * 60:
            print("stop-after-min: выхожу (проверка resume)", flush=True)
            break
        # проверка жадной политикой: в самом начале (до обучения) и дальше раз в eval-min
        if first_eval and st is None or time.time() - run.last_eval > args.eval_min * 60:
            evaluate()
            run.last_eval = time.time()
        first_eval = False

        # --- раунд попыток: на каждом столе своя
        obs = env.reset()
        img = obs["img"].to(dev)
        x, y, yaw, _ = greedy(net, img)
        eps = eps_at(run.hours())
        rnd = torch.rand(env.N, device=dev) < eps
        rx, ry, ryaw = random_on_object(img)
        x, y, yaw = torch.where(rnd, rx, x), torch.where(rnd, ry, y), torch.where(rnd, ryaw, yaw)
        true = torch.stack(env.oracle(), dim=1)  # только в лог: где объект на самом деле, сеть это не видит
        r, info = env.grasp(x, y, yaw)
        act = torch.stack([x, y, yaw], dim=1)
        buf.add(img, act, to_kij(x, y, yaw), r)
        run.log_train(attempt, obs["obj"], act.cpu(), r.cpu(), eps, true.cpu())
        attempt += env.N

        # --- учим: на каждую новую попытку 4 батча по 8 (было 1 по 32 - U-Net на 128x128 съел +2 ГБ, run.sh убил за 6 ГБ)
        losses = []
        for _ in range(4 * env.N):
            b_img, _, b_kij, b_r = buf.sample(8, dev)
            logits = net(b_img)
            pred = logits[torch.arange(len(b_r), device=dev), b_kij[:, 0], b_kij[:, 1], b_kij[:, 2]]
            loss = F.binary_cross_entropy_with_logits(pred, b_r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(
            f"попытка {attempt:6d}  {run.hours():.2f} ч  eps {eps:.2f}  взял {int(r.sum())}/{env.N}  "
            f"недостижимо {int((~info['reach']).sum())}  loss {sum(losses) / len(losses):.3f}",
            flush=True,
        )

        if time.time() - run.last_ckpt > args.ckpt_min * 60:
            run.save({"net": net.state_dict(), "opt": opt.state_dict(), "attempt": attempt}, buf)
            run.last_ckpt = time.time()

    run.save({"net": net.state_dict(), "opt": opt.state_dict(), "attempt": attempt}, buf)
    if run.hours() >= args.hours:
        evaluate()
        open(os.path.join(run.dir, "DONE"), "w").close()
        print("DONE: время вышло, обучение закончено", flush=True)


if __name__ == "__main__":
    main()
    app.close()
