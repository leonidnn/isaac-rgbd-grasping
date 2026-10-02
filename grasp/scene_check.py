"""This script demonstrates how to use the differential inverse kinematics CONTROLLER with the simulator.

The differential IK CONTROLLER can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.
Скопировано из MetaIsaacGrasp/test_ur10cfg.py. Run: gt run grasp/scene_check.py <gpu>
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from isaac_app import make_app

app = make_app()

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms

OUT = os.environ.get("GT_RUN_DIR", ".")
USD_PATH = ROOT / "MetaIsaacGrasp" / "models"

DOWN_RATION = 0.75
BOW_ANGLE = 0.1
WRIST_LIFT = 0.5
ARM_JOINT = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -np.pi * DOWN_RATION + BOW_ANGLE,
    "elbow_joint": np.pi * DOWN_RATION,
    "wrist_1_joint": -np.pi / 2 - BOW_ANGLE - WRIST_LIFT,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": np.pi,
}
GRIPPER_OPEN = 0.0425
GRIPPER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]

# x, y, z, qw, qx, qy, qz in robot base frame
EE_GOALS = [
    [0.0, 0.51, 0.2, 0.707, -0.707, 0.0, 0.0],
    [0.0, 0.51, 0.05, 0.707, -0.707, 0.0, 0.0],
    [0.2, 0.51, 0.3, 0.707, -0.707, 0.0, 0.0],
    [-0.2, 0.7, 0.15, 0.707, -0.707, 0.0, 0.0],
]
STEPS_PER_GOAL = 300
GRIPPER_STEPS = 100
POS_TOL = 0.01
CAM_POS = (0.0, 0.6, 1.2)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    )
    # table top at z=0, robot base stands on it
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.4, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.4, -0.4)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH / "ur10e_with_hand_e_and_camera_mount.usd"),
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
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**ARM_JOINT, **{j: GRIPPER_OPEN for j in GRIPPER_JOINTS}}
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT),
                velocity_limit_sim=50.0,
                effort_limit_sim=1e4,
                stiffness=5e3,
                damping=400.0,
            ),
            "gripper": ImplicitActuatorCfg(joint_names_expr=GRIPPER_JOINTS, stiffness=7000, damping=100),
        },
    )
    # ros convention: camera looks along +Z, rot (0,1,0,0) turns it straight down
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
    )


def save_frame(camera, name):
    from PIL import Image

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    np.save(os.path.join(OUT, f"{name}_depth.npy"), depth)
    Image.fromarray(rgb).save(os.path.join(OUT, f"{name}_rgb.png"))
    finite = np.isfinite(depth)
    d = np.zeros(depth.shape, dtype=np.float32)
    if finite.any():
        lo, hi = depth[finite].min(), depth[finite].max()
        d[finite] = (depth[finite] - lo) / max(hi - lo, 1e-6)
    Image.fromarray((d * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_depth.png"))
    fd = depth[finite]
    return {
        "rgb_mean": float(rgb.mean()),
        "depth_finite_frac": float(finite.mean()),
        "depth_min": float(fd.min()) if fd.size else None,
        "depth_median": float(np.median(fd)) if fd.size else None,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[scene] setup complete", flush=True)

    robot = scene["robot"]
    camera = scene["camera"]
    arm = SceneEntityCfg("robot", joint_names=list(ARM_JOINT), body_names=["hande_end"])
    arm.resolve(scene)
    grip_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    ee_id = arm.body_ids[0]
    ee_jacobi_idx = ee_id - 1 if robot.is_fixed_base else ee_id

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=sim.device,
    )
    sim_dt = sim.get_physics_dt()
    grip_target = torch.full((1, len(grip_ids)), GRIPPER_OPEN, device=sim.device)

    def step(arm_target=None):
        if arm_target is not None:
            robot.set_joint_position_target(arm_target, joint_ids=arm.joint_ids)
        robot.set_joint_position_target(grip_target, joint_ids=grip_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    def ee_pose_b():
        ee_w = robot.data.body_state_w[:, ee_id, 0:7]
        root_w = robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])

    for _ in range(50):
        step(robot.data.default_joint_pos[:, arm.joint_ids])

    results = []
    for goal in EE_GOALS:
        cmd = torch.tensor([goal], device=sim.device)
        ik.reset()
        ik.set_command(cmd)
        for _ in range(STEPS_PER_GOAL):
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm.joint_ids]
            pos_b, quat_b = ee_pose_b()
            q_des = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm.joint_ids])
            step(q_des)
        pos_b, quat_b = ee_pose_b()
        pos_err = torch.linalg.norm(pos_b - cmd[:, 0:3], dim=-1).item()
        rot_err = quat_error_magnitude(quat_b, cmd[:, 3:7]).item()
        results.append({"goal": goal, "pos_err_mm": round(pos_err * 1000, 2), "rot_err_deg": round(np.degrees(rot_err), 2)})
        print(f"[scene] goal {goal[:3]}: pos err {pos_err * 1000:.2f} mm, rot err {np.degrees(rot_err):.2f} deg", flush=True)

    frame = save_frame(camera, "arm_at_goal")

    hold = robot.data.joint_pos[:, arm.joint_ids].clone()
    grip = {}
    for name, val in (("closed", 0.0), ("open", GRIPPER_OPEN)):
        grip_target[:] = val
        for _ in range(GRIPPER_STEPS):
            step(hold)
        grip[name] = [round(v, 4) for v in robot.data.joint_pos[0, grip_ids].tolist()]
        print(f"[scene] gripper {name}: {grip[name]} (target {val})", flush=True)

    checks = {
        "reach": all(r["pos_err_mm"] < POS_TOL * 1000 for r in results),
        "gripper_closes": max(grip["closed"]) < 0.005,
        "gripper_opens": min(grip["open"]) > GRIPPER_OPEN - 0.005,
        "camera": frame["rgb_mean"] > 10 and frame["depth_finite_frac"] > 0.9,
    }
    report = {"goals": results, "gripper": grip, "frame": frame, "checks": checks}
    with open(os.path.join(OUT, "scene_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    ok = all(checks.values())
    print(f"[scene] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


t0 = time.time()
ok = False
try:
    ok = main()
except Exception:
    traceback.print_exc()
finally:
    print(f"[scene] total {time.time() - t0:.1f}s", flush=True)
    app.close()
sys.exit(0 if ok else 1)
