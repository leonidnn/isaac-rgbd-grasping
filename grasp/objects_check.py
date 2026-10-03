"""Ставим все три объекта на стол (стоя и лёжа), ждём 2 с физики и смотрим, что они не провалились,
не улетели и успокоились. В конце кадр RGB-D сверху.

    bash server/run.sh grasp/objects_check.py <gpu> [--video]

--video: ещё и objects.gif, кадр каждые 2 шага, в 2 раза медленнее реального времени
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(os.environ.get("GT_RUN_DIR", "."))
USD_DIR = os.path.join(ROOT, "assets", "usd")

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude

STEPS = 200  # dt 0.01 -> 2 с
CHECK_FROM = 100  # с этого шага объект уже должен лежать спокойно

# лёжа = повернули на 90 градусов вокруг X
UP = (1.0, 0.0, 0.0, 0.0)
SIDE = (0.70710678, 0.70710678, 0.0, 0.0)


def obj(name, x, y, lying):
    # стоя начало координат меша на дне, лёжа объект ложится боком, роняем с небольшой высоты
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{name}_{'side' if lying else 'up'}",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD_DIR, name, f"{name}.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.06 if lying else 0.005), rot=SIDE if lying else UP),
    )


@configclass
class ObjectsSceneCfg(InteractiveSceneCfg):
    # пол и стол как в scene_check: плита и коробка, верх стола на z=0
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.4, 1.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.4, -0.525)),
    )

    # верхний ряд стоя, нижний лёжа
    tetrapak_up = obj("tetrapak", -0.3, 0.65, False)
    can_up = obj("can", 0.0, 0.65, False)
    chips_up = obj("chips", 0.3, 0.65, False)
    tetrapak_side = obj("tetrapak", -0.3, 0.35, True)
    can_side = obj("can", 0.0, 0.35, True)
    chips_side = obj("chips", 0.3, 0.35, True)

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.5, 1.2), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
    )


NAMES = ["tetrapak_up", "can_up", "chips_up", "tetrapak_side", "can_side", "chips_side"]


def save_frame(camera):
    from PIL import Image

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    depth = np.nan_to_num(depth, posinf=0.0)

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "depth.npy"), depth)
    Image.fromarray(rgb).save(os.path.join(OUT, "rgb.png"))
    # растягиваем только диапазон стола и объектов, иначе всё одного цвета
    lo, hi = depth[depth > 0].min(), 1.2
    depth_img = (255 * np.clip((hi - depth) / (hi - lo), 0, 1)).astype(np.uint8)
    Image.fromarray(depth_img).save(os.path.join(OUT, "depth.png"))
    print(f"frame saved to {OUT}, depth {lo:.3f}..{depth.max():.3f} m", flush=True)


VIDEO_EVERY = 2
VIDEO_SIZE = (480, 360)


def grab(camera):
    from PIL import Image

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    return Image.fromarray(rgb).resize(VIDEO_SIZE)


def save_video(frames):
    # реальный шаг кадра 0.02 с, показываем по 0.04 -> замедление в 2 раза
    path = os.path.join(OUT, "objects.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"video saved to {path}, {len(frames)} frames", flush=True)


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
            frames.append(grab(scene["camera"]))
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
        # провалился ниже стола / укатился / ещё движется
        bad = pos[2].item() < -0.01 or drift > 5.0 or vel > 0.01
        ok &= not bad
        print(
            f"{n:14s} pos {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}  drift {drift:5.1f} mm  "
            f"vel {vel:.4f} m/s  rot {rot:5.1f} deg  {'BAD' if bad else 'ok'}",
            flush=True,
        )

    save_frame(scene["camera"])
    if frames:
        save_video(frames)
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
    app.close()
