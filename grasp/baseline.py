"""Baseline без RL: смотрим на глубину, vision.pose() даёт позу, хватаем сверху. Никакого подглядывания в симулятор.
Это копия env_check.py, только вместо env.oracle() стоит vision.pose(). Оракула зову чисто чтобы в лог записать,
насколько vision промахнулся мимо настоящего центра, на схват он не влияет.

    bash server/run.sh grasp/baseline.py <gpu> [--n 30] [--obj tetrapak can chips] [--seed 100]

В конце SR по каждому объекту с 95% интервалом Уилсона.
"""

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=30, help="сколько эпизодов на объект")
parser.add_argument("--obj", nargs="*", default=["tetrapak", "can", "chips"])
parser.add_argument("--seed", type=int, default=100, help="не 0, чтоб сцены не совпали с env_check")
parser.add_argument("--frames", type=int, default=3, help="сколько кадров с крестиком сохранить на объект")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import numpy as np
from PIL import Image, ImageDraw

import vision
from common import OUT
from env import GraspEnv


def wilson(k, n, z=1.96):
    # интервал Уилсона для доли, формулу взял с википедии (Binomial proportion confidence interval)
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def check_camera(env):
    # vision.py знает камеру по захардкоженным числам, тут сверяю с тем, что реально в сцене. Не сходится - всё мимо
    K = env.cam.data.intrinsic_matrices[0].cpu().numpy()
    pos = env.cam.data.pos_w[0].cpu().numpy()
    print(f"camera K fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, pos {pos.round(3).tolist()}", flush=True)
    print(f"vision  K fx {vision.FX:.1f} fy {vision.FY:.1f} cx {vision.CX:.1f} cy {vision.CY:.1f}, pos {list(vision.CAM)}", flush=True)
    bad = abs(K[0, 0] - vision.FX) > 1 or abs(K[1, 1] - vision.FY) > 1 or np.abs(pos - np.array(vision.CAM)).max() > 0.005
    if bad:
        print("ВНИМАНИЕ: камера в vision.py не совпадает с настоящей, цифрам не верить", flush=True)


def draw(obs, g, path):
    img = Image.fromarray(obs["rgb"])
    if g is not None:
        d = ImageDraw.Draw(img)
        u, v = vision.pixel(g[0], g[1], vision.CAM[2] - g[2])
        # крестик - куда целимся, палка - вдоль чего сожмутся пальцы
        d.line([u - 6, v, u + 6, v], fill=(255, 0, 0), width=2)
        d.line([u, v - 6, u, v + 6], fill=(255, 0, 0), width=2)
        du, dv = 40 * math.cos(g[3]), -40 * math.sin(g[3])
        d.line([u - du, v - dv, u + du, v + dv], fill=(0, 0, 255), width=2)
    img.save(path)


def main():
    t = time.time()
    env = GraspEnv(seed=args.seed)
    print(f"env ready in {time.time() - t:.1f}s", flush=True)
    check_camera(env)
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

    stats, log = {}, []
    t_all = time.time()
    for name in args.obj:
        ok = 0
        for k in range(args.n):
            t = time.time()
            obs = env.reset(obj=name)
            g = vision.pose(obs["depth"])
            tr = obs["true"]
            true_xy = env.oracle()[:2]  # только для лога
            if g is None:
                # не нашли объект на картинке - эпизод в минус, хватать нечем
                res = {"success": False, "rise": 0.0, "knocked": False, "ee_err": float("nan")}
                err = float("nan")
            else:
                res = env.step(*g, side=False)
                err = math.hypot(g[0] - true_xy[0], g[1] - true_xy[1])
            dt = time.time() - t
            ok += res["success"]
            gs = "NOT FOUND" if g is None else f"grasp ({g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}) yaw {math.degrees(g[3]):+5.0f}"
            print(
                f"{name:8s} #{k:02d} {tr['pose']:9s} yaw {math.degrees(tr['yaw']):+5.0f}  {gs}  "
                f"xy err {err * 1000:5.1f} mm  rise {res['rise'] * 100:5.1f} cm  {'KNOCKED ' if res['knocked'] else ''}"
                f"{'OK' if res['success'] else 'fail'}  {dt:.1f}s",
                flush=True,
            )
            if k < args.frames:
                draw(obs, g, os.path.join(OUT, "frames", f"{name}_{k:02d}_{tr['pose']}.png"))
            log.append({"obj": name, **tr, "grasp": None if g is None else list(g), "xy_err": err, **res})
        stats[name] = ok
        lo, hi = wilson(ok, args.n)
        print(f"--- {name}: {ok}/{args.n} = {ok / args.n:.0%}  95% CI [{lo:.0%}, {hi:.0%}]", flush=True)

    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump(log, f, indent=1)
    total = sum(stats.values())
    n_all = args.n * len(stats)
    lo, hi = wilson(total, n_all)
    print(
        "SR: " + ", ".join(f"{k} {v}/{args.n}" for k, v in stats.items())
        + f"; всего {total}/{n_all} = {total / n_all:.0%} [{lo:.0%}, {hi:.0%}]; "
        f"{(time.time() - t_all) / n_all:.1f} с на эпизод",
        flush=True,
    )


if __name__ == "__main__":
    main()
    app.close()
