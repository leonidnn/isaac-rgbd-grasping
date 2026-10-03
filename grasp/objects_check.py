"""Кидаем все три объекта на стол (стоя и лёжа), ждём 2 с и смотрим: не провалились, не улетели,
улеглись. В конце снимаем RGB-D сверху.

    bash server/run.sh grasp/objects_check.py <gpu> [--video]

--video: плюс гифка objects.gif, кадр раз в 2 шага, крутится в 2 раза медленнее, чтоб успеть разглядеть
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude

from common import CAMERA_TOP, GROUND, LIGHT, SIDE, TABLE, grab, obj_cfg, save_frame, save_gif

STEPS = 200  # dt 0.01 -> 2 с
CHECK_FROM = 100  # к этому шагу всё уже должно было улечься
VIDEO_EVERY = 2
VIDEO_SIZE = (480, 360)


def obj(name, x, y, lying):
    # стоя ноль меша на дне, а лёжа кладём боком и роняем чуть сверху, чтоб не застрял в столе
    prim = f"{name}_{'side' if lying else 'up'}"
    return obj_cfg(name, (x, y, 0.06 if lying else 0.005), SIDE if lying else (1.0, 0.0, 0.0, 0.0), prim)


@configclass
class ObjectsSceneCfg(InteractiveSceneCfg):
    ground = GROUND
    light = LIGHT
    table = TABLE

    # сверху ряд стоя, снизу лёжа
    tetrapak_up = obj("tetrapak", -0.3, 0.65, False)
    can_up = obj("can", 0.0, 0.65, False)
    chips_up = obj("chips", 0.3, 0.65, False)
    tetrapak_side = obj("tetrapak", -0.3, 0.35, True)
    can_side = obj("can", 0.0, 0.35, True)
    chips_side = obj("chips", 0.3, 0.35, True)

    camera = CAMERA_TOP


NAMES = ["tetrapak_up", "can_up", "chips_up", "tetrapak_side", "can_side", "chips_side"]


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
    scene = InteractiveScene(ObjectsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("setup done", flush=True)

    objs = {n: scene[n] for n in NAMES}
    quat0 = {n: o.data.root_quat_w.clone() for n, o in objs.items()}
    pos_mid = {}
    frames = []

    for count in range(1, STEPS + 1):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if args.video and count % VIDEO_EVERY == 0:
            frames.append(grab(scene["camera"], VIDEO_SIZE))
        if count == CHECK_FROM:
            pos_mid = {n: o.data.root_pos_w[0].clone() for n, o in objs.items()}
        if count % 50 == 0:
            print(f"step {count}/{STEPS}", flush=True)

    ok = True
    for n, o in objs.items():
        pos = o.data.root_pos_w[0]
        vel = o.data.root_lin_vel_w[0].norm().item()
        drift = (pos - pos_mid[n]).norm().item() * 1000
        rot = np.degrees(quat_error_magnitude(o.data.root_quat_w, quat0[n]).item())
        # провалился сквозь стол, уехал или до сих пор шевелится
        bad = pos[2].item() < -0.01 or drift > 5.0 or vel > 0.01
        ok &= not bad
        print(
            f"{n:14s} pos {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}  drift {drift:5.1f} mm  "
            f"vel {vel:.4f} m/s  rot {rot:5.1f} deg  {'BAD' if bad else 'ok'}",
            flush=True,
        )

    save_frame(scene["camera"])
    if frames:
        # кадр снят каждые 0.02 с, а показываем по 0.04, вот и замедление в 2 раза
        save_gif(frames, "objects.gif", ms=40)
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
    app.close()
