"""Пробуем один нормальный схват: рукой рулит IK, пальцами отдельно.
Сначала щёлкаем захватом в воздухе и смотрим, насколько он реально раскрывается,
потом подъезжаем к объекту, опускаемся, жмём, тащим вверх на 10+ см и держим 2 с.

    bash server/run.sh grasp/grasp_check.py <gpu> [--obj tetrapak|can|chips] [--side] [--video]
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--obj", default="tetrapak", choices=["tetrapak", "can", "chips"])
parser.add_argument("--side", action="store_true", help="сбоку, а не сверху")
parser.add_argument("--video", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from common import (
    ARM_JOINT, CAMERA_SIDE, CAMERA_TOP, CLOSED, FINGERS, GROUND, LIGHT, OPEN, ROBOT, SIDE, TABLE, UP,
    grab, obj_cfg, save_frame, save_gif,
)

OBJ_XY = (0.0, 0.5)
# где верх у объекта на столе: тетрапак и банка стоят, чипсы лежат (стоя они всё равно падают)
OBJ_TOP = {"tetrapak": 0.189, "can": 0.038, "chips": 0.044}
OBJ_ROT = {"tetrapak": UP, "can": UP, "chips": SIDE}
OBJ_Z0 = {"tetrapak": 0.005, "can": 0.005, "chips": 0.03}

# схват смотрит вниз (его z в -z мира). Вывел руками, надо глянуть, правда ли
QUAT_TOP = (0.0, 1.0, 0.0, 0.0)
# схват боком, смотрит от робота (+y), такие же цели были в scene_check
QUAT_SIDE = (0.70710678, -0.70710678, 0.0, 0.0)

GRIP_DEPTH = 0.03  # насколько пальцы залезают на объект сверху
PRE = 0.12  # с какого расстояния подъезжаем
LIFT = 0.15
LIFT_OK = 0.10

# фазы расписал сам, подсмотрел у стейт-машины MetaIsaacGrasp (air_env_base: reach/approach/grasp/lift),
# только без warp и под один объект. (имя, сколько шагов), шаг 0.01 с
PHASES = [
    ("settle", 50),
    ("test_close", 80),
    ("test_open", 80),
    ("pregrasp", 300),
    ("approach", 200),
    ("close", 100),
    ("lift", 200),
    ("hold", 200),
]


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground = GROUND
    light = LIGHT
    table = TABLE
    obj = obj_cfg(args.obj, (*OBJ_XY, OBJ_Z0[args.obj]), OBJ_ROT[args.obj])
    robot = ROBOT
    camera_top = CAMERA_TOP
    camera = CAMERA_SIDE


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("setup done", flush=True)

    robot, obj, cam = scene["robot"], scene["obj"], scene["camera"]
    dev = sim.device
    arm_ids, _ = robot.find_joints(list(ARM_JOINT), preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGERS, preserve_order=True)
    ee_id = robot.find_bodies("hande_end")[0][0]
    finger_bodies = robot.find_bodies(["hande_left_finger", "hande_right_finger"], preserve_order=True)[0]
    ee_jacobi_idx = ee_id - 1  # база прибита, поэтому -1

    limits = getattr(robot.data, "joint_pos_limits", None)
    if limits is None:
        limits = robot.data.joint_limits
    print(f"finger limits {limits[0, finger_ids].tolist()}", flush=True)

    # IK и ee_pose стащил из run_simulator в scene_check, просто завернул в функцию
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, dev
    )

    def ee_pose():
        root = robot.data.root_state_w[:, 0:7]
        ee = robot.data.body_state_w[:, ee_id, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    # дальше уже своё
    def finger_gap():
        p = robot.data.body_pos_w[0, finger_bodies]
        return (p[0] - p[1]).norm().item() * 1000

    # цели в координатах базы робота, но база стоит в нуле, так что это то же самое, что мир
    top = OBJ_TOP[args.obj]
    if args.side:
        quat = torch.tensor(QUAT_SIDE, device=dev)
        grasp = torch.tensor([OBJ_XY[0], OBJ_XY[1], top / 2], device=dev)
        pre = grasp - torch.tensor([0.0, PRE, 0.0], device=dev)
    else:
        quat = torch.tensor(QUAT_TOP, device=dev)
        grasp = torch.tensor([OBJ_XY[0], OBJ_XY[1], max(top - GRIP_DEPTH, 0.015)], device=dev)
        pre = grasp + torch.tensor([0.0, 0.0, PRE], device=dev)
    lifted = grasp + torch.tensor([0.0, 0.0, LIFT], device=dev)

    fingers = torch.full((1, 2), OPEN, device=dev)
    frames, obj_z0, gap_open, gap_closed = [], None, None, None
    count = 0

    for name, n in PHASES:
        ee_pos, _ = ee_pose()
        start = ee_pos[0].clone()
        if name == "pregrasp":
            target_from, target_to = start, pre
        elif name == "approach":
            target_from, target_to = pre, grasp
        elif name == "lift":
            target_from, target_to = grasp, lifted
        else:
            target_from = target_to = None
        if name in ("test_close", "close"):
            fingers[:] = CLOSED
        elif name in ("settle", "test_open", "pregrasp"):
            fingers[:] = OPEN
        if name == "approach":
            obj_z0 = obj.data.root_pos_w[0, 2].item()

        for i in range(n):
            if target_from is not None:
                # тащим цель плавно по прямой, а то если сразу кинуть в конечную точку, рука дёргается
                a = min(1.0, (i + 1) / (0.7 * n))
                cmd = torch.cat([target_from + a * (target_to - target_from), quat]).unsqueeze(0)
                ik.set_command(cmd)
                ee_pos, ee_quat = ee_pose()
                # шаг IK как в scene_check, только по 6 суставам руки, пальцы не трогаем, они ниже
                jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids]
                arm_des = ik.compute(ee_pos, ee_quat, jac, robot.data.joint_pos[:, arm_ids])
                robot.set_joint_position_target(arm_des, joint_ids=arm_ids)
            elif count == 0:
                robot.set_joint_position_target(robot.data.joint_pos[:, arm_ids].clone(), joint_ids=arm_ids)
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            count += 1
            if args.video and count % 5 == 0:
                frames.append(grab(cam))

        ee_pos, _ = ee_pose()
        q = robot.data.joint_pos[0, finger_ids].tolist()
        oz = obj.data.root_pos_w[0, 2].item()
        err = (ee_pos[0] - target_to).norm().item() * 1000 if target_to is not None else float("nan")
        print(
            f"[{name:10s}] step {count:4d}  ee err {err:6.1f} mm  fingers {q[0]:.4f} {q[1]:.4f}  "
            f"gap {finger_gap():6.1f} mm  obj z {oz:+.3f}",
            flush=True,
        )
        if name == "settle":
            save_frame(scene["camera_top"])
        if name == "test_close":
            gap_closed = finger_gap()
        if name == "test_open":
            gap_open = finger_gap()

    rise = obj.data.root_pos_w[0, 2].item() - obj_z0
    held = robot.data.joint_pos[0, finger_ids]
    print(f"gripper stroke: gap open {gap_open:.1f} mm, closed {gap_closed:.1f} mm "
          f"-> ход {abs(gap_open - gap_closed):.1f} mm", flush=True)
    print(f"object rise {rise * 100:.1f} cm, fingers at {held.tolist()}", flush=True)

    if frames:
        save_gif(frames, f"grasp_{args.obj}{'_side' if args.side else ''}.gif", ms=50)
    print("PASS" if rise >= LIFT_OK else "FAIL", flush=True)


if __name__ == "__main__":
    main()
    app.close()
