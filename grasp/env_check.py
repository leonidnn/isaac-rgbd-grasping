"""Проверяем саму среду: хватаем оракулом, который подглядывает, где объект на самом деле.
Если даже так не берёт - значит косяк в самом схвате, а не в агенте.

    bash server/run.sh grasp/env_check.py <gpu> [--n 10] [--obj tetrapak] [--yaw-offset 0] [--side] [--seed 0]

Печатает каждый эпизод, сколько взяли по каждому объекту и сколько секунд ушло на эпизод.
Пару кадров сверху на объект кидает в frames/, чтоб глазами глянуть.
"""

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="сколько эпизодов на объект")
parser.add_argument("--obj", nargs="*", default=["tetrapak", "can", "chips"])
parser.add_argument("--yaw-offset", type=float, default=0.0, help="на сколько градусов докрутить кисть к yaw объекта")
parser.add_argument("--side", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--frames", type=int, default=3, help="сколько кадров сохранить на объект")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

from PIL import Image

from common import OUT
from env import GraspEnv


def main():
    t = time.time()
    env = GraspEnv(seed=args.seed)
    print(f"env ready in {time.time() - t:.1f}s", flush=True)
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

    stats, log = {}, []
    for name in args.obj:
        ok = 0
        for k in range(args.n):
            t = time.time()
            obs = env.reset(obj=name)
            grasp = env.oracle(math.radians(args.yaw_offset), args.side)
            res = env.step(*grasp)
            dt = time.time() - t
            ok += res["success"]
            tr = obs["true"]
            print(
                f"{name:8s} #{k:02d} {tr['pose']:9s} yaw {math.degrees(tr['yaw']):+6.0f}  "
                f"grasp ({grasp[0]:+.3f} {grasp[1]:+.3f} {grasp[2]:+.3f}) yaw {math.degrees(grasp[3]):+6.0f}  "
                f"rise {res['rise'] * 100:5.1f} cm  {'KNOCKED ' if res['knocked'] else ''}"
                f"{'OK' if res['success'] else 'fail'}  {dt:.1f}s",
                flush=True,
            )
            if k < args.frames:
                Image.fromarray(obs["rgb"]).save(os.path.join(OUT, "frames", f"{name}_{k:02d}_{tr['pose']}.png"))
            log.append({"obj": name, **tr, "grasp": list(grasp[:4]), **res})
        stats[name] = ok
        print(f"--- {name}: {ok}/{args.n}", flush=True)

    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("SR: " + ", ".join(f"{k} {v}/{args.n}" for k, v in stats.items()), flush=True)


if __name__ == "__main__":
    main()
    app.close()
