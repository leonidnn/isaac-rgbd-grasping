"""OBJ -> USD for the three task objects (Isaac Lab MeshConverter, same as scripts/tools/convert_mesh.py).

Input:  assets/prepared/<name>/*.obj  (textures fixed and downscaled)
Output: assets/usd/<name>/<name>.usd  + printed bounding boxes

Run only through server/run.sh:
    bash server/run.sh infra/convert_objects.py <gpu>
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# meshes are Y-up, Isaac is Z-up: +90 deg around X, quaternion (w, x, y, z)
Y_UP_TO_Z_UP = (0.70710678, 0.70710678, 0.0, 0.0)

# name: (obj file, scale, mass kg)
OBJECTS = {
    "tetrapak": ("tetra-pak-carton.obj", 1.0, 0.25),
    "can": ("tin-can.obj", 1.0, 0.2),
    "chips": ("chips-bag.obj", 1.8, 0.1),
}

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=os.path.join(REPO, "assets", "prepared"))
parser.add_argument("--dst", default=os.path.join(REPO, "assets", "usd"))
parser.add_argument("--only", nargs="*", default=list(OBJECTS))
parser.add_argument("--collision", default="convexHull")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"[convert] app started in {time.time() - t0:.1f}s", flush=True)

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from pxr import Usd, UsdGeom


def bbox(usd_path):
    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


ok = True
for name in args.only:
    obj, scale, mass = OBJECTS[name]
    src = os.path.join(args.src, name, obj)
    if not os.path.isfile(src):
        print(f"[convert] {name}: no file {src}", flush=True)
        ok = False
        continue
    cfg = MeshConverterCfg(
        asset_path=src,
        usd_dir=os.path.join(args.dst, name),
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        collision_approximation=args.collision,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        rotation=Y_UP_TO_Z_UP,
        scale=(scale, scale, scale),
    )
    try:
        out = MeshConverter(cfg).usd_path
        lo, hi = bbox(out)
        size = hi - lo
        print(f"[convert] {name}: {out}", flush=True)
        print(f"[convert] {name}: size xyz {size[0]:.3f} {size[1]:.3f} {size[2]:.3f} m, "
              f"z {lo[2]:.3f}..{hi[2]:.3f}, mass {mass} kg", flush=True)
    except Exception as e:
        print(f"[convert] {name}: FAILED {e!r}", flush=True)
        ok = False

print(f"[convert] {'PASS' if ok else 'FAIL'} in {time.time() - t0:.1f}s", flush=True)
app.close()
sys.exit(0 if ok else 1)
