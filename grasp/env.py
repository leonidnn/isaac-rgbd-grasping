"""Среда для схвата. На столе один объект, сверху снимаем RGB-D, потом хватаем по позе (x, y, z, yaw, сверху/сбоку).
Isaac Sim запускаем один раз, а дальше гоняем reset() / step() сколько влезет.
Импортить после make_app(), как и common, иначе упадёт.

    env = GraspEnv(seed=0)
    obs = env.reset()                 # obs["rgb"], obs["depth"], obs["obj"], obs["true"] - true это читерство, только для отладки
    res = env.step(x, y, z, yaw, side=False)   # res["success"], res["rise"], res["knocked"]
"""

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from common import ARM_JOINT, CAMERA_TOP, CLOSED, FINGERS, GROUND, LIGHT, OPEN, ROBOT, SIDE, TABLE, UP, obj_cfg

OBJECTS = ["tetrapak", "can", "chips"]

# как можно положить каждый объект. Банку на ребро не ставим - катится, чипсы стоя не ставим - падают (см. 2_objects)
# поза: (кватернион до поворота по yaw, с какой высоты роняем)
POSES = {
    "tetrapak": {"up": (UP, 0.002), "side": (SIDE, 0.035)},
    "can": {"up": (UP, 0.002)},
    "chips": {"face_up": (SIDE, 0.03), "face_down": ((0.70710678, -0.70710678, 0.0, 0.0), 0.03)},
}
# где у объекта середина в его осях (у меша ноль на дне). Это для оракула, чтоб целился в середину
CENTER = {"tetrapak": (0.0, 0.0, 0.0945), "can": (0.0, 0.0, 0.019), "chips": (0.0, 0.0, 0.11)}
# половинки размеров по x, y, z, списал из вывода конвертера. Чипсы уже растянуты в 1.8
HALF = {"tetrapak": (0.030, 0.030, 0.0945), "can": (0.040, 0.040, 0.019), "chips": (0.074, 0.024, 0.1075)}

# куда кидаем объект: квадрат перед роботом, куда рука вроде дотягивается (env_check покажет, так или нет)
X_RANGE = (-0.2, 0.2)
Y_RANGE = (0.35, 0.65)
# лишние объекты валяются на полу подальше от стола, в камеру не попадают
PARK = {"tetrapak": (2.0, -0.5, -1.0), "can": (2.0, 0.0, -1.0), "chips": (2.0, 0.5, -1.0)}

# как повёрнута кисть, пока не крутили по yaw. Пальцы торчат по y кисти, на этом уже обжёгся (results/3_grasp)
QUAT_TOP = (0.70710678, -0.70710678, 0.0, 0.0)
QUAT_SIDE = (1.0, 0.0, 0.0, 0.0)
HAND_FORWARD = (0.0, 1.0, 0.0)  # куда торчат пальцы в осях кисти

PRE = 0.12
LIFT = 0.15
LIFT_OK = 0.10
KNOCK = 0.02  # если объект уехал больше чем на 2 см ещё до того, как сжали - значит снесли его

# фазы схвата, шаг 0.01 с. Урезал по сравнению с grasp_check и выкинул щёлканье пальцами в воздухе
PHASES = [("pregrasp", 200), ("approach", 100), ("close", 60), ("lift", 100), ("hold", 100)]
SETTLE = 50
WARMUP_RENDERS = 3  # первые кадры RTX бывают кривые, пару раз рендерим вхолостую


@configclass
class EnvSceneCfg(InteractiveSceneCfg):
    ground = GROUND
    light = LIGHT
    table = TABLE
    robot = ROBOT
    camera = CAMERA_TOP
    tetrapak = obj_cfg("tetrapak", PARK["tetrapak"], prim="tetrapak")
    can = obj_cfg("can", PARK["can"], prim="can")
    chips = obj_cfg("chips", PARK["chips"], prim="chips")


def _q(t, dev):
    return torch.tensor(t, dtype=torch.float32, device=dev).unsqueeze(0)


def yaw_q(yaw, dev):
    z = torch.zeros(1, device=dev)
    return quat_from_euler_xyz(z, z, torch.tensor([yaw], dtype=torch.float32, device=dev))


class GraspEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
        self.scene = InteractiveScene(EnvSceneCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()
        self.dev = self.sim.device
        self.dt = self.sim.get_physics_dt()

        self.robot = self.scene["robot"]
        self.cam = self.scene["camera"]
        self.objs = {n: self.scene[n] for n in OBJECTS}
        self.arm_ids, _ = self.robot.find_joints(list(ARM_JOINT), preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(FINGERS, preserve_order=True)
        self.ee_id = self.robot.find_bodies("hande_end")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, self.dev
        )
        self.obj = None
        self.z0 = None
        self.xy0 = None

    # --- всякая мелочь

    def _ee(self):
        root = self.robot.data.root_state_w[:, 0:7]
        ee = self.robot.data.body_state_w[:, self.ee_id, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    def _tick(self, n=1, render=False):
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.dt)

    def _place(self, name, pos, quat):
        o = self.objs[name]
        o.write_root_pose_to_sim(torch.cat([_q(pos, self.dev), quat], dim=1))
        o.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.dev))
        o.reset()

    # --- сам эпизод

    def reset(self, obj=None, pose=None):
        # робота домой, пальцы раскрыть
        r = self.robot
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(), torch.zeros_like(r.data.default_joint_vel))
        r.set_joint_position_target(r.data.default_joint_pos.clone())
        r.reset()
        self.ik.reset()

        # всё на пол, а один, который выпал, на стол
        for n in OBJECTS:
            self._place(n, PARK[n], _q(UP, self.dev))
        self.obj = obj or OBJECTS[self.rng.integers(len(OBJECTS))]
        pose = pose or list(POSES[self.obj])[self.rng.integers(len(POSES[self.obj]))]
        q_pose, z = POSES[self.obj][pose]
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        x, y = float(self.rng.uniform(*X_RANGE)), float(self.rng.uniform(*Y_RANGE))
        self._place(self.obj, (x, y, z), quat_mul(yaw_q(yaw, self.dev), _q(q_pose, self.dev)))

        self._tick(SETTLE)
        o = self.objs[self.obj].data
        self.z0 = o.root_pos_w[0, 2].item()
        self.xy0 = o.root_pos_w[0, :2].clone()

        # рендерим только здесь, один кадр на эпизод, а то дорого
        for _ in range(WARMUP_RENDERS):
            self.sim.render()
        self.scene.update(self.dt)
        rgb = self.cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = np.nan_to_num(self.cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy(), posinf=0.0)
        return {
            "rgb": rgb,
            "depth": depth,
            "obj": self.obj,
            # где объект на самом деле - агенту это давать нельзя, только оракулу и в логи
            "true": {"pose": pose, "yaw": yaw, "pos": o.root_pos_w[0].tolist(), "quat": o.root_quat_w[0].tolist()},
        }

    def hand_quat(self, yaw, side=False):
        return quat_mul(yaw_q(yaw, self.dev), _q(QUAT_SIDE if side else QUAT_TOP, self.dev))[0]

    def step(self, x, y, z, yaw, side=False):
        quat = self.hand_quat(yaw, side)
        grasp = torch.tensor([x, y, z], dtype=torch.float32, device=self.dev)
        # подъезжаем с обратной стороны от пальцев: сверху это просто повыше, сбоку - со стороны кисти
        fwd = quat_apply(quat.unsqueeze(0), _q(HAND_FORWARD, self.dev))[0]
        pre = grasp - PRE * fwd
        lifted = grasp + torch.tensor([0.0, 0.0, LIFT], device=self.dev)
        fingers = torch.full((1, 2), OPEN, device=self.dev)
        obj = self.objs[self.obj].data
        knocked = False

        for name, n in PHASES:
            ee_pos, _ = self._ee()
            seg = {"pregrasp": (ee_pos[0].clone(), pre), "approach": (pre, grasp), "lift": (grasp, lifted)}.get(name)
            if name == "close":
                fingers[:] = CLOSED
                # перед тем как сжать, смотрим, не снесли ли объект
                knocked = (obj.root_pos_w[0, :2] - self.xy0).norm().item() > KNOCK
            for i in range(n):
                if seg is not None:
                    a = min(1.0, (i + 1) / (0.7 * n))
                    self.ik.set_command(torch.cat([seg[0] + a * (seg[1] - seg[0]), quat]).unsqueeze(0))
                    ee_pos, ee_quat = self._ee()
                    jac = self.robot.root_physx_view.get_jacobians()[:, self.ee_id - 1, :, self.arm_ids]
                    arm = self.ik.compute(ee_pos, ee_quat, jac, self.robot.data.joint_pos[:, self.arm_ids])
                    self.robot.set_joint_position_target(arm, joint_ids=self.arm_ids)
                self.robot.set_joint_position_target(fingers, joint_ids=self.finger_ids)
                self._tick()

        ee_pos, _ = self._ee()
        rise = obj.root_pos_w[0, 2].item() - self.z0
        return {
            "success": rise >= LIFT_OK,
            "rise": rise,
            "knocked": knocked,
            "ee_err": (ee_pos[0] - lifted).norm().item(),
        }

    # --- оракул: подглядывает, где объект на самом деле, и целится в середину. Чисто проверить, что сама среда работает

    def oracle(self, yaw_offset=0.0, side=False):
        o = self.objs[self.obj].data
        q = o.root_quat_w
        center = (o.root_pos_w + quat_apply(q, _q(CENTER[self.obj], self.dev)))[0]
        # где верх: от середины вверх на сумму проекций полуосей на вертикаль
        axes = torch.eye(3, device=self.dev) * torch.tensor(HALF[self.obj], device=self.dev)
        top = center[2] + quat_apply(q.expand(3, 4), axes)[:, 2].abs().sum()
        # yaw объекта - куда смотрит его ось x, если глядеть сверху
        ax = quat_apply(q, _q((1.0, 0.0, 0.0), self.dev))[0]
        yaw = math.atan2(ax[1].item(), ax[0].item()) + yaw_offset
        if side:
            return center[0].item(), center[1].item(), center[2].item(), yaw, True
        z = max(top.item() - 0.03, 0.015)
        return center[0].item(), center[1].item(), z, yaw, False
