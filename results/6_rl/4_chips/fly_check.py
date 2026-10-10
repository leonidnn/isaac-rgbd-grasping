"""Проверка быстрой среды: хватает оракул (знает настоящий центр и yaw), N столов разом.
Если оракул тут берёт тетрапак и банку - среда годится для обучения. Заодно меряю, сколько секунд на попытку.

    bash server/run.sh results/6_rl/4_chips/fly_check.py <gpu> [--num-envs 8] [--rounds 3] [--obj chips] [--video]

Чипсы: копия из 2_depth, научрук попросил ужать пакет (узкая сторона 6 см, assets/usd/chips_small). Смотрю,
берёт ли их оракул вообще. --obj - какие объекты класть, по столам по кругу (по умолч. все три, как раньше).

Кладёт в папку запуска: what_agent_sees.png (что видит агент: RGB и высоты по первым столам), gif первого стола.
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# корень репо ищу вверх по папкам, чтоб не копировать в tmp_rl* как раньше
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "grasp")):
    ROOT = os.path.dirname(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--obj", nargs="*", default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import numpy as np
import torch

from common import OUT
from fly_env import FlyEnv
from viz import save_gif, sees_png


def main():
    t = time.time()
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=args.video)
    print(f"env ready in {time.time() - t:.1f}s, столов {args.num_envs}", flush=True)
    stats = {}
    t_att = []
    for rnd in range(args.rounds):
        t = time.time()
        obs = env.reset([args.obj[k % len(args.obj)] for k in range(env.N)] if args.obj else None)
        x, y, yaw = env.oracle()
        if rnd == 0:
            sees_png(obs["img"][:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(OUT, "what_agent_sees.png"))
        env.rec = [] if (args.video and rnd == 0) else None
        r, info = env.grasp(x, y, yaw)
        if env.rec:
            save_gif(env.rec, os.path.join(OUT, f"oracle_env0_{obs['obj'][0]}.gif"))
        env.rec = None
        dt = time.time() - t
        t_att.append(dt / env.N)
        for k in range(env.N):
            s = stats.setdefault(obs["obj"][k], [0, 0])
            s[0] += int(r[k].item())
            s[1] += 1
        print(
            f"round {rnd}: {int(r.sum())}/{env.N} взял, недостижимо {int((~info['reach']).sum())}, снёс {int(info['knocked'].sum())}, "
            f"{dt:.1f} с на раунд = {dt / env.N:.2f} с на попытку",
            flush=True,
        )
        for k in range(env.N):
            print(f"    {k:2d} {obs['obj'][k]:8s} rise {info['rise'][k] * 100:5.1f} cm  z {info['z'][k]:.3f}  reach {bool(info['reach'][k])}", flush=True)
    print("SR оракула: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in stats.items()), flush=True)
    # первый раунд с видео дольше, его не считаю, если есть другие
    tt = t_att[1:] if len(t_att) > 1 else t_att
    print(f"в среднем {np.mean(tt):.2f} с на попытку при {env.N} столах = {3600 / np.mean(tt):.0f} попыток в час", flush=True)
    print(f"память torch: пик {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB (без Isaac, весь процесс смотри в run.sh)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
