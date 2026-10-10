"""Всё общее для скриптов в grasp, чтоб не копипастить по файлам.
Импортить только после make_app(), тут внутри уже isaaclab, без запущенного кита упадёт.
"""

import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(os.environ.get("GT_RUN_DIR", "."))
USD_DIR = os.path.join(ROOT, "assets", "usd")
ROBOT_USD = os.path.join(ROOT, "MetaIsaacGrasp", "models", "ur10e_with_hand_e_and_camera_mount.usd")

# --- сцена, взял из scene_check. Пол и стол просто коробки, потому что нормальные с Nucleus с сервера не качаются

GROUND = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.CuboidCfg(
        size=(6.0, 6.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
)
LIGHT = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
# верх стола ровно на z=0, так удобнее считать
TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(1.4, 1.4, 1.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.4, -0.525)),
)

# --- объекты, вытащил из objects_check

UP = (1.0, 0.0, 0.0, 0.0)
SIDE = (0.70710678, 0.70710678, 0.0, 0.0)  # лёжа - это просто повернули на 90 вокруг X


# чипсы ужал по просьбе научрука (узкая сторона 6 см вместо 15), лежат в отдельной папке.
# старые assets/usd/chips не трогаю - на них учился агент из agent/
USD_NAME = {"chips": "chips_small"}


def obj_cfg(name, pos, rot=UP, prim="obj"):
    usd = USD_NAME.get(name, name)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim,
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD_DIR, usd, f"{usd}.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


# --- робот, тоже из scene_check (а туда он попал из test_ur10cfg.py MetaIsaacGrasp)

DOWN_RATION, BOW_ANGLE, WRIST_LIFT = 0.75, 0.1, 0.5
ARM_JOINT = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -np.pi * DOWN_RATION + BOW_ANGLE,
    "elbow_joint": np.pi * DOWN_RATION,
    "wrist_1_joint": -np.pi / 2 - BOW_ANGLE - WRIST_LIFT,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": np.pi,
}
FINGERS = ["hande_left_finger_joint", "hande_right_finger_joint"]
# так в оригинале: 0.0425 открыто, 0 закрыто. Не факт что правда, проверяем в grasp_check
OPEN, CLOSED = 0.0425, 0.0

ROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=50.0,
            linear_damping=2,
            angular_damping=2,
            max_contact_impulse=float("inf"),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={**ARM_JOINT, **{j: OPEN for j in FINGERS}}),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT),
            velocity_limit_sim=50.0,
            effort_limit_sim=1e4,
            stiffness=5e3,
            damping=400.0,
        ),
        # это уже моё: жмёт с силой stiffness * ошибка, но не сильнее effort_limit (у живого Hand-E 130 Н)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGERS,
            velocity_limit_sim=0.15,
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
)

# --- камеры

_PINHOLE = sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0))

# висит над столом и смотрит вниз, это то, что потом будет видеть агент
CAMERA_TOP = CameraCfg(
    prim_path="{ENV_REGEX_NS}/CameraTop",
    update_period=0.0,
    width=640,
    height=480,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=_PINHOLE,
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.5, 1.2), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
)
# сбоку, чисто для видео, чтоб было видно, поднялось или нет
CAMERA_SIDE = CameraCfg(
    prim_path="{ENV_REGEX_NS}/CameraSide",
    update_period=0.0,
    width=480,
    height=360,
    data_types=["rgb"],
    spawn=_PINHOLE,
    offset=CameraCfg.OffsetCfg(pos=(0.9, 0.5, 0.3), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
)


def save_frame(camera, out=OUT):
    from PIL import Image

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    depth = np.nan_to_num(depth, posinf=0.0)

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "depth.npy"), depth)
    Image.fromarray(rgb).save(os.path.join(out, "rgb.png"))
    # растягиваем только стол и объекты, а то на картинке всё одним серым
    lo, hi = depth[depth > 0].min(), 1.2
    depth_img = (255 * np.clip((hi - depth) / (hi - lo), 0, 1)).astype(np.uint8)
    Image.fromarray(depth_img).save(os.path.join(out, "depth.png"))
    print(f"frame saved to {out}, depth {lo:.3f}..{depth.max():.3f} m", flush=True)


def grab(camera, size=None):
    from PIL import Image

    img = Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
    return img.resize(size) if size else img


def save_gif(frames, name, ms=40, out=OUT):
    path = os.path.join(out, name)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    print(f"video saved to {path}, {len(frames)} frames", flush=True)
