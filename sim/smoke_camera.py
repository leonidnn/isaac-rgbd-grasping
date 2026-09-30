"""Smoke test: headless Isaac Sim, one camera -> RGB + depth to files.

Run only through server/run.sh:
    bash server/run.sh sim/smoke_camera.py <gpu>
"""

import argparse
import json
import os
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--out", default=os.environ.get("GT_RUN_DIR", "."))
args = parser.parse_args()

t_start = time.time()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_app import make_app

app = make_app(args.width, args.height)
print(f"[smoke] app started in {time.time() - t_start:.1f}s", flush=True)

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from pxr import UsdGeom, UsdLux, UsdPhysics

CAM_HEIGHT = 1.2
CUBE_SIZE = 0.1


def build_scene(stage):
    UsdGeom.Xform.Define(stage, "/World")

    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    ground.CreateDisplayColorAttr([(0.6, 0.6, 0.6)])
    xf = UsdGeom.XformCommonAPI(ground)
    xf.SetTranslate((0.0, 0.0, -0.05))
    xf.SetScale((4.0, 4.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    cube.CreateSizeAttr(CUBE_SIZE)
    cube.CreateDisplayColorAttr([(0.8, 0.1, 0.1)])
    xf = UsdGeom.XformCommonAPI(cube)
    xf.SetTranslate((0.1, 0.0, 0.3))
    xf.SetRotate((0.0, 0.0, 30.0))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.2)

    UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(500.0)

    # USD camera looks along its -Z, so identity rotation = straight down
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr((0.01, 10.0))
    UsdGeom.XformCommonAPI(cam).SetTranslate((0.0, 0.0, CAM_HEIGHT))
    return "/World/cam"


def save_images(out, rgb, depth):
    np.save(os.path.join(out, "depth.npy"), depth)
    np.save(os.path.join(out, "rgb.npy"), rgb)
    try:
        from PIL import Image
    except ImportError:
        print("[smoke] PIL not found, saved only .npy", flush=True)
        return
    Image.fromarray(rgb[..., :3]).save(os.path.join(out, "rgb.png"))
    finite = np.isfinite(depth)
    d = np.zeros(depth.shape, dtype=np.float32)
    if finite.any():
        lo, hi = depth[finite].min(), depth[finite].max()
        d[finite] = (depth[finite] - lo) / max(hi - lo, 1e-6)
    Image.fromarray((d * 255).astype(np.uint8)).save(os.path.join(out, "depth.png"))


def main():
    os.makedirs(args.out, exist_ok=True)
    carb.settings.get_settings().set_int("/persistent/physics/numThreads", 2)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 60,
        rendering_dt=1 / 60,
        backend="torch",
        device="cuda",
    )
    cam_path = build_scene(omni.usd.get_context().get_stage())

    rp = rep.create.render_product(cam_path, (args.width, args.height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    world.reset()
    t0 = time.time()
    for _ in range(args.steps):
        world.step(render=True)
    print(f"[smoke] {args.steps} steps in {time.time() - t0:.1f}s", flush=True)

    rgb = depth = None
    for _ in range(30):
        rgb = np.asarray(rgb_annot.get_data())
        depth = np.asarray(depth_annot.get_data())
        if rgb.size and depth.size:
            break
        world.step(render=True)
    if rgb is None or not rgb.size or not depth.size:
        print("[smoke] FAIL: annotators returned no data", flush=True)
        return False

    save_images(args.out, rgb, depth)

    finite = np.isfinite(depth)
    fd = depth[finite]
    ground = CAM_HEIGHT
    cube_top = CAM_HEIGHT - CUBE_SIZE
    stats = {
        "rgb_shape": list(rgb.shape),
        "rgb_mean": float(rgb[..., :3].mean()),
        "rgb_std": float(rgb[..., :3].std()),
        "depth_shape": list(depth.shape),
        "depth_finite_frac": float(finite.mean()),
        "depth_min": float(fd.min()) if fd.size else None,
        "depth_median": float(np.median(fd)) if fd.size else None,
        "total_time_s": round(time.time() - t_start, 1),
    }
    checks = {
        "rgb_not_black": stats["rgb_mean"] > 10 and stats["rgb_std"] > 2,
        "depth_valid": stats["depth_finite_frac"] > 0.9,
        "ground_depth": fd.size > 0 and abs(stats["depth_median"] - ground) < 0.05,
        "cube_visible": fd.size > 0 and stats["depth_median"] - stats["depth_min"] > 0.05,
    }
    # cube fell from 0.3 m onto the ground -> its top is at 0.1 m
    physics_ok = fd.size > 0 and abs(stats["depth_min"] - cube_top) < 0.03
    stats["checks"] = checks
    stats["physics_cube_landed"] = bool(physics_ok)
    with open(os.path.join(args.out, "smoke.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2), flush=True)
    if not physics_ok:
        print("[smoke] WARN: cube is not on the ground, check GPU physics", flush=True)
    ok = all(checks.values())
    print(f"[smoke] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


ok = False
try:
    ok = main()
except Exception:
    traceback.print_exc()
finally:
    app.close()
sys.exit(0 if ok else 1)
