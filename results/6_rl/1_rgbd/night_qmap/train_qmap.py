"""Q-карта: сеть смотрит на стол (RGB + высоты 128x128) и для каждой клетки и каждого из 8 yaw говорит,
какой шанс, что схват сверху туда поднимет объект. Хватаю туда, где шанс больше всего.
Учу по-простому: после попытки знаю, взял или нет, и подтягиваю выход сети в этой одной клетке к 0 или 1 (BCE).
По сути это Q-learning, где эпизод из одного шага (гамма = 0), так что никаких целевых сетей не надо.
Идею взял из статьи Zeng et al. 2018 (Visual Pushing-Grasping), сеть сильно проще.

    bash server/run.sh --timeout 10h tmp_rl/train_qmap.py <gpu> --name qmap --hours 8 [--num-envs 8]

Падать можно: следующий запуск с тем же --name продолжит с бэкапа (~/grasp_task/ckpt/<name>).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="qmap")
parser.add_argument("--hours", type=float, default=8.0, help="сколько всего учиться, с учётом прошлых запусков")
parser.add_argument("--stop-after-min", type=float, default=0, help="для проверок: выйти через столько минут этого запуска")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-min", type=float, default=30, help="как часто проверка жадной политикой + видео + картинки")
parser.add_argument("--ckpt-min", type=float, default=10)
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
from train_common import Buffer, Run, random_on_object, to_kij

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
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return self.out(u)  # [B,8,128,128] логиты, sigmoid - вероятность успеха


def greedy(net, img):
    """лучшая клетка и yaw по сети"""
    with torch.no_grad():
        logits = net(img)
    flat = logits.flatten(1).argmax(dim=1)
    k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
    i, j = rest // IMG, rest % IMG
    x, y = FlyEnv.pix_to_xy(i, j)
    return x, y, k.float() * torch.pi / N_YAW, torch.sigmoid(logits)


def eps_at(attempt):
    # сколько случайных попыток: сначала половина, к 3000-й попытке 10%, дальше так и держу
    return max(0.1, 0.5 - 0.4 * attempt / 3000)


def main():
    torch.manual_seed(args.seed)
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=True)
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
    t_this = time.time()
    first_eval = True

    def evaluate():
        """2 раунда жадно, без случайности. Первый раунд с видео первого стола и картинкой карт"""
        stats = {}
        for rnd in range(2):
            obs = env.reset()
            x, y, yaw, prob = greedy(net, obs["img"].to(dev))
            if rnd == 0:
                tag = f"{attempt:06d}"
                titles = [f"{obs['obj'][k]}" for k in range(4)]
                viz.qmap_png(obs["img"][:4], prob[:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(run.viz, f"qmap_{tag}.png"), titles)
                env.rec = []
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
        eps = eps_at(attempt)
        rnd = torch.rand(env.N, device=dev) < eps
        rx, ry, ryaw = random_on_object(img)
        x, y, yaw = torch.where(rnd, rx, x), torch.where(rnd, ry, y), torch.where(rnd, ryaw, yaw)
        r, info = env.grasp(x, y, yaw)
        act = torch.stack([x, y, yaw], dim=1)
        buf.add(img, act, to_kij(x, y, yaw), r)
        run.log_train(attempt, obs["obj"], act.cpu(), r.cpu(), eps)
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
