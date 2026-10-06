"""Погонять моего агента в Isaac Sim у себя на компе. Нужны Isaac Sim 4.5 + Isaac Lab 2.0 и RTX-видюха.

    python agent/sim.py fast      # быстро: 8 столов, рука телепортом встаёт над точкой (так агент учился)
    python agent/sim.py arm       # рука честно едет из домашней позы к объекту, один стол

Опции: --obj tetrapak can chips, --n (fast: раундов по 8 попыток, по умолч. 10; arm: эпизодов на объект, по умолч. 5),
--seed, --video (gif в первых N раундах/эпизодах), --out (по умолч. agent/out/<режим>_<время>).
Кладёт log.txt (SR по объектам, 95% интервал Уилсона), scenes.png (куда целится, тепловая карта), gif сбоку.

Тут собрано из моих файлов репозитория в один: сцена (grasp/common.py, grasp/env.py), кинематика UR10e (grasp/planner.py),
быстрая среда (results/6_rl/2_depth/fly_env.py). Подъезд в режиме arm - простой IK по прямой, как у оракула в
results/4_env (там 20/20), а не мой планировщик с лесенкой - он большой. Сеть и выбор схвата - в agent.py.
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# на сервере лаборатории, где я всё учил, видюхи общие, и Isaac Sim там можно запускать только через server/run.sh
if os.path.isdir(os.path.expanduser("~/grasp_task")):
    sys.exit("это для своего компа. На сервере лаборатории запускать только через server/run.sh")

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["fast", "arm"])
parser.add_argument("--obj", nargs="*", default=["tetrapak", "can", "chips"])
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--video", type=int, default=1)
parser.add_argument("--out", default=None)
args = parser.parse_args()
OUT = args.out or os.path.join(HERE, "out", f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUT, exist_ok=True)

# ---------------- запуск Isaac Sim, headless (камеры включаю так же, как Isaac Lab по --enable_cameras)
from isaacsim import SimulationApp

ASSET_ROOT = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480,
                     "extra_args": [f"--/persistent/isaac/asset_root/default={ASSET_ROOT}"]})
import carb

_s = carb.settings.get_settings()
for key in ("/isaaclab/cameras_enabled", "/isaaclab/render/offscreen", "/physics/fabricUpdateTransformations"):
    _s.set_bool(key, True)
for key in ("/isaaclab/render/active_viewport", "/isaaclab/render/rtx_sensors"):
    _s.set_bool(key, False)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_euler_xyz, quat_from_matrix, quat_mul, subtract_frame_transforms

import agent as A

# ---------------- сцена (из grasp/common.py и grasp/env.py)

USD_DIR = os.path.join(ROOT, "assets", "usd")
ROBOT_USD = os.path.join(ROOT, "MetaIsaacGrasp", "models", "ur10e_with_hand_e_and_camera_mount.usd")
OBJECTS = ["tetrapak", "can", "chips"]
UP, SIDE = (1.0, 0.0, 0.0, 0.0), (0.70710678, 0.70710678, 0.0, 0.0)
# как кладу каждый объект: (поворот до yaw, с какой высоты роняю). Банку только стоя, чипсы только лёжа
POSES = {
    "tetrapak": {"up": (UP, 0.002), "side": (SIDE, 0.035)},
    "can": {"up": (UP, 0.002)},
    "chips": {"face_up": (SIDE, 0.03), "face_down": ((0.70710678, -0.70710678, 0.0, 0.0), 0.03)},
}
CENTER = {"tetrapak": (0.0, 0.0, 0.0945), "can": (0.0, 0.0, 0.019), "chips": (0.0, 0.0, 0.11)}  # середина в осях объекта
X_RANGE, Y_RANGE = (-0.2, 0.2), (0.35, 0.65)  # куда кидаю объект (его начало координат), метры
PARK = {"tetrapak": (1.0, 0.0, -1.0), "can": (1.0, 0.4, -1.0), "chips": (1.0, 0.8, -1.0)}  # лишние - на пол между столами
QUAT_TOP = (0.70710678, -0.70710678, 0.0, 0.0)  # кисть смотрит вниз; пальцы Hand-E торчат по y кисти, не по z
OPEN, CLOSED = 0.0425, 0.0
LIFT, LIFT_OK, ABOVE = 0.15, 0.10, 0.10
ARM_JOINT = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -np.pi * 0.75 + 0.1, "elbow_joint": np.pi * 0.75,
             "wrist_1_joint": -np.pi / 2 - 0.6, "wrist_2_joint": -np.pi / 2, "wrist_3_joint": np.pi}
FINGERS = ["hande_left_finger_joint", "hande_right_finger_joint"]
CAM_W, CAM_H, CAM_Z, CAM_Y = 320, 240, 1.2, 0.5
SETTLE, WARMUP = 35, 10  # RTX отдаёт кадр с опозданием: после перестановки объектов надо 8+ рендеров
PHASES = [("descent", 70), ("close", 60), ("lift", 70), ("hold", 100)]
PREGRASP = 200  # arm: сколько шагов едем из домашней позы в точку над объектом

_pin = sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0))
_box = lambda size, pos, color: sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))


def obj_cfg(name):
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD_DIR, name, f"{name}.usd")),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=PARK[name]))


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=_box((60.0, 60.0, 0.02), None, (0.4, 0.4, 0.4)),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=_box((1.4, 1.4, 1.05), None, (0.55, 0.45, 0.35)),
                         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.4, -0.525)))  # верх стола на z=0
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD, rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=50.0, linear_damping=2, angular_damping=2, max_contact_impulse=float("inf"))),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={**ARM_JOINT, **{j: OPEN for j in FINGERS}}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=list(ARM_JOINT), velocity_limit_sim=50.0, effort_limit_sim=1e4, stiffness=5e3, damping=400.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=FINGERS, velocity_limit_sim=0.15, effort_limit_sim=100.0, stiffness=2000.0, damping=100.0),
        },
    )
    camera = CameraCfg(prim_path="{ENV_REGEX_NS}/CameraTop", update_period=0.0, width=CAM_W, height=CAM_H,
                       data_types=["rgb", "distance_to_image_plane"], spawn=_pin,
                       offset=CameraCfg.OffsetCfg(pos=(0.0, CAM_Y, CAM_Z), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"))
    tetrapak, can, chips = obj_cfg("tetrapak"), obj_cfg("can"), obj_cfg("chips")


# ---------------- кинематика UR10e по DH (из grasp/planner.py) - чтоб сразу поставить руку над точкой

DH_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
DH_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def dh_fk(q, T_pre, T_post):
    T = torch.eye(4, dtype=q.dtype, device=q.device).repeat(q.shape[0], 1, 1)
    for i in range(6):
        ct, st, ca, sa = torch.cos(q[:, i]), torch.sin(q[:, i]), math.cos(DH_ALPHA[i]), math.sin(DH_ALPHA[i])
        A_ = torch.zeros_like(T)
        A_[:, 0, 0], A_[:, 0, 1], A_[:, 0, 2], A_[:, 0, 3] = ct, -st * ca, st * sa, DH_A[i] * ct
        A_[:, 1, 0], A_[:, 1, 1], A_[:, 1, 2], A_[:, 1, 3] = st, ct * ca, -ct * sa, DH_A[i] * st
        A_[:, 2, 1], A_[:, 2, 2], A_[:, 2, 3], A_[:, 3, 3] = sa, ca, DH_D[i], 1.0
        T = T @ A_
    return T_pre @ T @ T_post


def rot_err(R, R_goal):
    qe = quat_from_matrix(R_goal @ R.transpose(1, 2))
    qe = torch.where(qe[:, :1] < 0, -qe, qe)
    s = qe[:, 1:].norm(dim=1, keepdim=True)
    return torch.where(s > 1e-9, qe[:, 1:] / s.clamp(min=1e-9) * 2 * torch.atan2(s, qe[:, :1]), 2 * qe[:, 1:])


class Kin:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo.double(), hi.double()
        self.T_pre = self.T_post = None

    def fk(self, q):
        return dh_fk(q, self.T_pre, self.T_post)

    def jac(self, q, eps=1e-6):
        T0, cols = self.fk(q), []
        for i in range(6):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            T1 = self.fk(q + dq)
            cols.append(torch.cat([(T1[:, :3, 3] - T0[:, :3, 3]) / eps, rot_err(T0[:, :3, :3], T1[:, :3, :3]) / eps], 1))
        return torch.stack(cols, 2)

    def calibrate(self, q0, T_ee, jac_sim):
        """база DH у UR то ли повёрнута на 180, то ли нет; кисть подгоняю так, чтоб в домашней позе сошлось точно"""
        best = None
        for flip in (0.0, math.pi):
            R = torch.eye(4, dtype=torch.float64, device=q0.device)
            R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(flip), -math.sin(flip), math.sin(flip), math.cos(flip)
            self.T_pre, self.T_post = R, torch.eye(4, dtype=torch.float64, device=q0.device)
            self.T_post = torch.linalg.inv(self.fk(q0)[0]) @ T_ee
            diff = (self.jac(q0)[0] - jac_sim.double()).abs().max().item()
            if best is None or diff < best[0]:
                best = (diff, self.T_pre, self.T_post)
        _, self.T_pre, self.T_post = best

    def ik(self, pos, quat, seed, iters=60, lam=0.05):
        q, R_goal = seed.double().clone(), matrix_from_quat(quat.double())
        eye = torch.eye(6, dtype=torch.float64, device=q.device)
        for _ in range(iters):
            T = self.fk(q)
            e = torch.cat([pos.double() - T[:, :3, 3], rot_err(T[:, :3, :3], R_goal)], 1)
            J = self.jac(q)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam**2 * eye, e.unsqueeze(-1))).squeeze(-1)
            q = torch.clamp(q + dq.clamp(-0.3, 0.3), self.lo, self.hi)
        return q, (pos.double() - self.fk(q)[:, :3, 3]).norm(dim=1)


def yaw_quat(yaw):
    z = torch.zeros_like(yaw)
    return quat_from_euler_xyz(z, z, yaw)


# ---------------- среда (из results/6_rl/2_depth/fly_env.py)


class Env:
    def __init__(self, num_envs, seed, video):
        self.N, self.rng = num_envs, np.random.default_rng(seed)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
        cfg = SceneCfg(num_envs=num_envs, env_spacing=2.0)
        if video:  # боковая камера только у первого стола
            cfg.camera_side = CameraCfg(prim_path="/World/envs/env_0/CameraSide", update_period=0.0, width=320, height=240,
                                        data_types=["rgb"], spawn=_pin,
                                        offset=CameraCfg.OffsetCfg(pos=(0.9, 0.5, 0.3), rot=(0.0, 0.0, 0.0, 1.0), convention="world"))
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.dev, self.dt, self.origins = self.sim.device, self.sim.get_physics_dt(), self.scene.env_origins
        self.robot, self.cam = self.scene["robot"], self.scene["camera"]
        self.side = self.scene["camera_side"] if video else None
        self.objs = {n: self.scene[n] for n in OBJECTS}
        self.arm_ids, _ = self.robot.find_joints(list(ARM_JOINT), preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(FINGERS, preserve_order=True)
        self.ee_id = self.robot.find_bodies("hande_end")[0][0]
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), self.N, self.dev)
        self.rec, self.n_tick = None, 0

        r = self.robot
        for _ in range(5):
            r.set_joint_position_target(r.data.default_joint_pos.clone())
            self._tick()
        lim = getattr(r.data, "joint_pos_limits", None)
        lim = (r.data.joint_limits if lim is None else lim)[0, self.arm_ids]
        self.kin = Kin(lim[:, 0], lim[:, 1])
        p, q = self._ee()
        T_ee = torch.eye(4, dtype=torch.float64, device=self.dev)
        T_ee[:3, :3], T_ee[:3, 3] = matrix_from_quat(q[:1].double())[0], p[0].double()
        self.kin.calibrate(r.data.joint_pos[:1, self.arm_ids].double(), T_ee, r.root_physx_view.get_jacobians()[0, self.ee_id - 1, :, self.arm_ids])
        # затравка IK: кисть вниз над серединой стола, из кучи случайных стартов
        lo, hi = self.kin.lo.clamp(min=-math.pi), self.kin.hi.clamp(max=math.pi)
        seeds = lo + torch.rand(128, 6, dtype=torch.float64, device=self.dev) * (hi - lo)
        seeds[0] = r.data.joint_pos[0, self.arm_ids].double()
        qs, err = self.kin.ik(torch.tensor([[0.0, 0.5, 0.25]], device=self.dev).expand(128, 3),
                              torch.tensor([QUAT_TOP], device=self.dev).expand(128, 4), seeds, iters=300)
        # из точных решений беру ближайшее к домашней позе
        self.q_seed = qs[((err > 0.002) * 1e3 + (qs - seeds[0]).norm(dim=1)).argmin()]

        xs = torch.linspace(A.WS_X[0], A.WS_X[1], A.IMG, device=self.dev)
        ys = torch.linspace(A.WS_Y[1], A.WS_Y[0], A.IMG, device=self.dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        fx = CAM_W * 18.0 / 20.955
        u, v = CAM_W / 2 + gx * fx / CAM_Z, CAM_H / 2 - (gy - CAM_Y) * fx / CAM_Z
        self.grid = torch.stack([u / CAM_W * 2 - 1, v / CAM_H * 2 - 1], -1).unsqueeze(0)

    def _ee(self):
        root, ee = self.robot.data.root_state_w[:, 0:7], self.robot.data.body_state_w[:, self.ee_id, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    def _tick(self, n=1):
        for _ in range(n):
            self.n_tick += 1
            frame = self.rec is not None and self.n_tick % 10 == 0
            self.scene.write_data_to_sim()
            self.sim.step(render=frame)
            self.scene.update(self.dt)
            if frame:
                self.rec.append(Image.fromarray(self.side.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)))

    def _obj_pos(self, k):
        return self.objs[self.obj[k]].data.root_pos_w[k]

    def reset(self, objs):
        """каждому столу свой объект, случайное место, yaw, стоит/лежит; рука домой. Вернёт карты высот [N,128,128]"""
        r = self.robot
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(), torch.zeros_like(r.data.default_joint_vel))
        r.set_joint_position_target(r.data.default_joint_pos.clone())
        r.reset()
        self.obj = objs
        for name in OBJECTS:
            pos = torch.tensor(PARK[name], device=self.dev).repeat(self.N, 1)
            quat = torch.tensor([UP], device=self.dev).repeat(self.N, 1)
            for i in range(self.N):
                if objs[i] == name:
                    pose = list(POSES[name])[self.rng.integers(len(POSES[name]))]
                    q_pose, z = POSES[name][pose]
                    pos[i] = torch.tensor([self.rng.uniform(*X_RANGE), self.rng.uniform(*Y_RANGE), z], device=self.dev)
                    quat[i] = quat_mul(yaw_quat(torch.tensor([self.rng.uniform(-math.pi, math.pi)], device=self.dev, dtype=torch.float32)),
                                       torch.tensor([q_pose], device=self.dev))[0]
            o = self.objs[name]
            o.write_root_pose_to_sim(torch.cat([pos + self.origins, quat], 1))
            o.write_root_velocity_to_sim(torch.zeros(self.N, 6, device=self.dev))
            o.reset()
        self._tick(SETTLE)
        self.z0 = torch.stack([self._obj_pos(k)[2] for k in range(self.N)])
        for _ in range(WARMUP):
            self.sim.render()
        self.scene.update(self.dt)
        depth = torch.nan_to_num(self.cam.data.output["distance_to_image_plane"][..., 0], posinf=0.0)
        h = torch.where(depth > 0, CAM_Z - depth, torch.zeros_like(depth)).unsqueeze(1)
        h = F.max_pool2d(F.grid_sample(h, self.grid.expand(self.N, -1, -1, -1), align_corners=False), 5, stride=1, padding=2)
        return h[:, 0].clamp(0, 0.25)

    def true_xy(self):
        out = []
        for k in range(self.N):
            q = self.objs[self.obj[k]].data.root_quat_w[k:k + 1]
            c = self._obj_pos(k) - self.origins[k] + quat_apply(q, torch.tensor([CENTER[self.obj[k]]], device=self.dev))[0]
            out.append((c[0].item(), c[1].item()))
        return out

    def grasp(self, picks, teleport):
        """picks - выход agent.choose. teleport: рука сразу над точкой (fast), иначе едет из домашней позы (arm)"""
        N, dev, r = self.N, self.dev, self.robot
        g = [p if p is not None else (0.0, 0.5, A.MIN_Z, 0.0) for p in picks]
        grasp = torch.tensor([[p[0], p[1], p[2]] for p in g], device=dev)
        yaw = torch.tensor([p[3] for p in g], device=dev)
        pre, lifted = grasp + torch.tensor([0.0, 0.0, ABOVE], device=dev), grasp + torch.tensor([0.0, 0.0, LIFT], device=dev)
        top = torch.tensor([QUAT_TOP], device=dev).expand(N, 4)
        quat = quat_mul(yaw_quat(yaw), top)
        fingers = torch.full((N, 2), OPEN, device=dev)
        phases = list(PHASES)
        if teleport:
            # пальцы симметричные: пробую yaw и yaw+180, беру что ближе к затравке
            quat2 = quat_mul(yaw_quat(yaw + math.pi), top)
            seed = self.q_seed.unsqueeze(0).expand(2 * N, 6)
            q, err = self.kin.ik(torch.cat([pre, pre]), torch.cat([quat, quat2]), seed)
            pick = (err[N:] * 100 + (q[N:] - seed[N:]).norm(dim=1) * 0.01) < (err[:N] * 100 + (q[:N] - seed[:N]).norm(dim=1) * 0.01)
            q_arm = torch.where(pick.unsqueeze(1), q[N:], q[:N]).float()
            quat = torch.where(pick.unsqueeze(1), quat2, quat)
            full = r.data.joint_pos.clone()
            full[:, self.arm_ids], full[:, self.finger_ids] = q_arm, OPEN
            r.write_joint_state_to_sim(full, torch.zeros_like(full))
            r.set_joint_position_target(full)
        else:
            phases = [("pregrasp", PREGRASP)] + phases
        self.ik.reset()
        start = self._ee()[0].clone()
        cmd = r.data.joint_pos[:, self.arm_ids]
        for name, n in phases:
            seg = {"pregrasp": (start, pre), "descent": (pre if not teleport else start, grasp), "lift": (grasp, lifted)}.get(name)
            if name == "close":
                fingers[:] = CLOSED
            for s in range(n):
                if seg is not None:
                    a = min(1.0, (s + 1) / (0.7 * n))
                    self.ik.set_command(torch.cat([seg[0] + a * (seg[1] - seg[0]), quat], 1))
                    ee_p, ee_q = self._ee()
                    jac = r.root_physx_view.get_jacobians()[:, self.ee_id - 1, :, self.arm_ids]
                    cmd = self.ik.compute(ee_p, ee_q, jac, r.data.joint_pos[:, self.arm_ids])
                r.set_joint_position_target(cmd, joint_ids=self.arm_ids)
                r.set_joint_position_target(fingers, joint_ids=self.finger_ids)
                self._tick()
        rise = torch.stack([self._obj_pos(k)[2] for k in range(N)]) - self.z0
        return [bool(rise[k] >= LIFT_OK) and picks[k] is not None for k in range(N)], rise


# ---------------- прогон


def main():
    log = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    fast = args.mode == "fast"
    env = Env(num_envs=8 if fast else 1, seed=args.seed, video=args.video > 0)
    net = A.load(dev=env.dev)
    n = args.n or (10 if fast else 5)
    rounds = n if fast else n * len(args.obj)
    say(f"режим {args.mode}, {rounds} раундов по {env.N} попыток, результаты в {OUT}")
    stats, shots = {o: [0, 0] for o in args.obj}, []
    for rnd in range(rounds):
        t = time.time()
        objs = [args.obj[(rnd * env.N + k) % len(args.obj)] for k in range(env.N)] if fast else [args.obj[rnd // n]]
        heights = env.reset(objs)
        picks = A.choose(net, heights)
        env.rec = [] if (rnd if fast else rnd % n) < args.video else None
        ok, rise = env.grasp(picks, teleport=fast)
        if env.rec:
            env.rec[0].save(os.path.join(OUT, f"{rnd:02d}_{objs[0]}_{'ok' if ok[0] else 'fail'}.gif"), save_all=True, append_images=env.rec[1:], duration=100, loop=0)
        env.rec = None
        if (fast and rnd == 0) or (not fast and rnd % n == 0):
            shots.append((heights.cpu(), picks, [f"{A.RU[o]}: {'взял' if ok[k] else 'не взял'}" for k, o in enumerate(objs)], env.true_xy()))
        for k, o in enumerate(objs):
            stats[o][0] += ok[k]
            stats[o][1] += 1
        say(f"раунд {rnd}: " + "  ".join(f"{o} {'OK' if ok[k] else '-'} ({rise[k] * 100:.0f} см)" for k, o in enumerate(objs)) + f"  {time.time() - t:.0f} с")
    A.scenes_png(torch.cat([s[0] for s in shots]), sum([s[1] for s in shots], []), sum([s[2] for s in shots], []),
                 os.path.join(OUT, "scenes.png"), true_xy=sum([s[3] for s in shots], []))
    say("SR (95% Уилсон):")
    for o, (k, m) in stats.items():
        lo, hi = A.wilson(k, m)
        say(f"  {A.RU[o]:9s} {k}/{m} = {k / max(m, 1):.0%}  [{lo:.0%}, {hi:.0%}]")
    log.close()


if __name__ == "__main__":
    main()
    app.close()
