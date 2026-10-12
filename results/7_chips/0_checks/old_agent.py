"""Старый агент (agent/qmap_depth.pt, учился на тетрапаке и банке) как есть на маленьких чипсах, в быстрой среде.
Ничего не учу, просто смотрю, сколько он берёт сейчас - это точка отсчёта перед дообучением.
Копия fly_check.py, только вместо оракула точку и yaw выбирает сеть, жадно, как в проверках при обучении
(greedy из 6_rl/2_depth/train_qmap.py, сеть из agent/agent.py).

    bash server/run.sh results/7_chips/0_checks/old_agent.py <gpu> [--num-envs 8] [--rounds 12] [--obj chips tetrapak can] [--video]

Кладёт в папку запуска: qmap_round0.png (куда целится, тепловые карты первых столов), gif первого стола.
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
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--obj", nargs="*", default=["chips", "tetrapak", "can"])
parser.add_argument("--video", action="store_true")
parser.add_argument("--seed", type=int, default=1000)
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import torch

from common import OUT
from fly_env import IMG, N_YAW, FlyEnv
from viz import qmap_png, save_gif

sys.path.insert(0, os.path.join(ROOT, "agent"))
from agent import load, wilson  # сеть и интервал Уилсона - из пакета агента, там без Isaac

ARM_Y = 0.29  # как в agent.py: ближе к роботу висит кисть, туда не хватаю


def object_pixels(img):
    # копия из 6_rl/2_depth/train_common.py: клетки выше стола на 1+ см и не у кисти
    on = img[:, 3] > int(0.01 / 0.25 * 255)
    rows = torch.arange(IMG, device=img.device)
    _, y = FlyEnv.pix_to_xy(rows, torch.zeros_like(rows))
    on[:, y < ARM_Y, :] = False
    return on


def greedy(net, img):
    # копия из 6_rl/2_depth/train_qmap.py
    with torch.no_grad():
        logits = net(img)
    on_obj = object_pixels(img)
    on_obj[on_obj.flatten(1).sum(1) == 0] = True
    masked = logits.masked_fill(~on_obj.unsqueeze(1), float("-inf"))
    flat = masked.flatten(1).argmax(dim=1)
    k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
    i, j = rest // IMG, rest % IMG
    x, y = FlyEnv.pix_to_xy(i, j)
    return x, y, k.float() * math.pi / N_YAW, torch.sigmoid(logits)


def main():
    t = time.time()
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=args.video)
    net = load(os.path.join(ROOT, "agent", "qmap_depth.pt"), env.dev)
    print(f"env ready in {time.time() - t:.1f}s, столов {args.num_envs}", flush=True)
    stats = {}
    for rnd in range(args.rounds):
        t = time.time()
        obs = env.reset([args.obj[k % len(args.obj)] for k in range(env.N)])
        img = obs["img"].to(env.dev)
        x, y, yaw, prob = greedy(net, img)
        ox, oy, oyaw = env.oracle()  # только в лог: где объект на самом деле
        if rnd == 0:
            qmap_png(obs["img"][:4], prob[:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(OUT, "qmap_round0.png"), [obs["obj"][k] for k in range(4)])
        env.rec = [] if (args.video and rnd == 0) else None
        r, info = env.grasp(x, y, yaw)
        if env.rec:
            save_gif(env.rec, os.path.join(OUT, f"agent_env0_{obs['obj'][0]}_{'ok' if r[0] > 0 else 'fail'}.gif"))
        env.rec = None
        for k in range(env.N):
            s = stats.setdefault(obs["obj"][k], [0, 0])
            s[0] += int(r[k].item())
            s[1] += 1
        print(f"round {rnd}: {int(r.sum())}/{env.N} взял, {time.time() - t:.1f} с", flush=True)
        for k in range(env.N):
            d = math.hypot(float(x[k] - ox[k]), float(y[k] - oy[k]))
            print(
                f"    {k:2d} {obs['obj'][k]:8s} {'OK  ' if r[k] > 0 else 'fail'} от центра {d * 100:4.1f} cm  "
                f"yaw сеть {math.degrees(float(yaw[k])):4.0f} объект {math.degrees(float(oyaw[k])):4.0f}  "
                f"шанс по сети {float(prob[k].max()):.2f}  rise {info['rise'][k] * 100:5.1f} cm",
                flush=True,
            )
    print("SR старого агента:", flush=True)
    for o, (a, b) in stats.items():
        lo, hi = wilson(a, b)
        print(f"    {o:8s} {a}/{b} = {a / b:.0%}  95% CI [{lo:.0%}, {hi:.0%}]", flush=True)


if __name__ == "__main__":
    main()
    app.close()
