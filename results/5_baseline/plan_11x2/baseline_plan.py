"""Baseline с планировщиком: vision.pose() даёт позу по глубине, а едет рука через top_plan (лесенка + проверка
столкновений по 50 точкам, потом спуск по прямой). Копия baseline.py, только env другой.

    bash server/run.sh tmp_top/baseline_plan.py <gpu> [--n 1] [--obj tetrapak] [--seed 100]
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
parser.add_argument("--n", type=int, default=1, help="сколько эпизодов на объект")
parser.add_argument("--obj", nargs="*", default=["tetrapak"])
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--video", type=int, default=0, help="для скольких первых эпизодов на объект писать gif сбоку")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import numpy as np
from PIL import Image, ImageDraw

import vision
from common import OUT, save_gif
from top_plan import PlanEnv


# wilson, check_camera, draw - копия из baseline.py (импортить его нельзя, он при импорте сам запускает кит)
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def check_camera(env):
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
        d.line([u - 6, v, u + 6, v], fill=(255, 0, 0), width=2)
        d.line([u, v - 6, u, v + 6], fill=(255, 0, 0), width=2)
        du, dv = 40 * math.cos(g[3]), -40 * math.sin(g[3])
        d.line([u - du, v - dv, u + du, v + dv], fill=(0, 0, 255), width=2)
    img.save(path)


def main():
    t = time.time()
    env = PlanEnv(seed=args.seed, video=args.video > 0)
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
            box = env.set_obstacle(obs["depth"])
            tr = obs["true"]
            true_xy = env.oracle()[:2]  # только для лога
            if g is None:
                res = {"success": False, "rise": 0.0, "knocked": False, "fail": "не нашёл объект"}
                err = float("nan")
            else:
                env.rec = [] if k < args.video else None
                res = env.step_plan(*g)
                if env.rec:
                    save_gif(env.rec, f"{name}_{k:02d}_{tr['pose']}.gif", ms=50)
                env.rec = None
                err = math.hypot(g[0] - true_xy[0], g[1] - true_xy[1])
            dt = time.time() - t
            ok += res["success"]
            print(
                f"{name:8s} #{k:02d} {tr['pose']:9s} yaw {math.degrees(tr['yaw']):+5.0f}  "
                + ("NOT FOUND" if g is None else f"grasp ({g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}) yaw {math.degrees(g[3]):+5.0f}")
                + f"  xy err {err * 1000:5.1f} mm  box {None if box is None else [round(v, 3) for v in box[1]]}",
                flush=True,
            )
            print(
                f"    IK решений {res.get('ik_ok')}, проверено лесенок {res.get('tried')}, отбраковано {res.get('hits')}, "
                f"взял №{res.get('sol')} порядок {res.get('order')}, план {res.get('plan_s', 0):.1f} с; "
                f"промах над объектом {res.get('pre_err', float('nan')) * 1000:.0f} мм, в схвате {res.get('grasp_err', float('nan')) * 1000:.0f} мм; "
                f"самый сильный контакт руки до сжатия {res.get('contact_before_close')}",
                flush=True,
            )
            print(
                f"    rise {res['rise'] * 100:5.1f} cm  {'KNOCKED ' if res['knocked'] else ''}{res.get('fail', '')} "
                f"{'OK' if res['success'] else 'fail'}  {dt:.1f}s",
                flush=True,
            )
            if k < args.frames:
                draw(obs, g, os.path.join(OUT, "frames", f"{name}_{k:02d}_{tr['pose']}.png"))
            log.append({"obj": name, **tr, "grasp": None if g is None else list(g), "xy_err": err,
                        **{a: b for a, b in res.items() if a != "contact_before_close"},
                        "contact_before_close": list(res.get("contact_before_close", ("", 0.0)))})
        stats[name] = ok
        lo, hi = wilson(ok, args.n)
        print(f"--- {name}: {ok}/{args.n} = {ok / args.n:.0%}  95% CI [{lo:.0%}, {hi:.0%}]", flush=True)

    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_all = args.n * len(stats)
    print(f"SR: {stats}; {(time.time() - t_all) / n_all:.1f} с на эпизод", flush=True)


if __name__ == "__main__":
    main()
    app.close()
