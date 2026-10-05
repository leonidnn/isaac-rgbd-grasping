"""Быстрая среда для обучения: много столов сразу, рука телепортом встаёт на 10 см над точкой схвата и дальше
только спуск, сжатие, подъём, держим. Никакой лесенки и планировщика - это для финальной проверки, тут дорого.

Агент видит кусок кадра над рабочей зоной стола 128x128: RGB + карта высот. Действие - точка на этой картинке
(x, y в метрах) и yaw. Глубину схвата не выбирает: верх под точкой минус 3 см, как у оракула.

    env = FlyEnv(num_envs=8)
    obs = env.reset()                 # obs["img"] [N,4,128,128] uint8, obs["obj"] список имён
    r, info = env.grasp(x, y, yaw)    # всё тензоры [N], r - 0/1

Импортить после make_app(). Куски взяты из env.py (позы объектов, фазы) и top_plan.py (IK).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

import planner as P
from common import ARM_JOINT, CAMERA_SIDE, CAMERA_TOP, CLOSED, FINGERS, LIGHT, OPEN, ROBOT, TABLE, obj_cfg
from env import CENTER, HALF, LIFT, LIFT_OK, OBJECTS, POSES, QUAT_TOP, X_RANGE, Y_RANGE

# рабочая зона, которую видит агент: квадрат 70x70 см на столе. Объекты кидаются в X_RANGE/Y_RANGE (это их начало
# координат), а центр лежачего тетрапака/чипсов от него ещё на ~10 см, поэтому с запасом
WS_X = (-0.35, 0.35)
WS_Y = (0.15, 0.85)
IMG = 128
N_YAW = 8  # yaw через 22.5 градуса, 0..180 (пальцы симметричные, 180 = то же самое)

# верхняя камера как в env.py, 640x480. Пробовал 256x192 - RGB выходил почти белый, банку не видно (fly_check_4envs)
CAM_W, CAM_H = 640, 480
FX = CAM_W * 18.0 / 20.955
CAM_Z = 1.2
CAM_Y = 0.5

ABOVE = 0.10
GRIP_DEPTH = 0.03
MIN_Z = 0.015
SETTLE = 50
PHASES = [("descent", 100), ("close", 60), ("lift", 100), ("hold", 100)]
IK_ITERS = 60
IK_POS_TOL = 0.005  # не дотянулся на 5 мм - считаю недостижимым, награда 0

# свалка лишних объектов: между столами, на полу. Камера туда не смотрит (закрывает стол)
PARK = {"tetrapak": (1.0, 0.0, -1.0), "can": (1.0, 0.4, -1.0), "chips": (1.0, 0.8, -1.0)}

# пол побольше, чем в common: столов много, а там пол 6x6
GROUND_BIG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.CuboidCfg(
        size=(60.0, 60.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
)


@configclass
class FlySceneCfg(InteractiveSceneCfg):
    ground = GROUND_BIG
    light = LIGHT
    table = TABLE
    robot = ROBOT
    camera = CAMERA_TOP.replace(width=CAM_W, height=CAM_H)
    tetrapak = obj_cfg("tetrapak", PARK["tetrapak"], prim="tetrapak")
    can = obj_cfg("can", PARK["can"], prim="can")
    chips = obj_cfg("chips", PARK["chips"], prim="chips")


def yaw_quat(yaw):
    z = torch.zeros_like(yaw)
    return quat_from_euler_xyz(z, z, yaw)


def ik_batch(plan, pos, quat, seed, iters=IK_ITERS, lam=0.05):
    """то же, что planner.ik, только у каждой строки своя цель. pos [M,3], quat [M,4], seed [M,6]"""
    q = seed.double().clone()
    R_goal = matrix_from_quat(quat.double())
    p_goal = pos.double()
    eye = torch.eye(6, dtype=torch.float64, device=q.device)
    for _ in range(iters):
        T = plan.fk(q)
        e = torch.cat([p_goal - T[:, :3, 3], P.rot_err(T[:, :3, :3], R_goal)], dim=1)
        J = plan.jac(q)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam**2 * eye, e.unsqueeze(-1))).squeeze(-1)
        q = torch.clamp(q + dq.clamp(-0.3, 0.3), plan.lo, plan.hi)
    T = plan.fk(q)
    return q, (p_goal - T[:, :3, 3]).norm(dim=1)


class FlyEnv:
    def __init__(self, num_envs=8, seed=0, video=False):
        self.N = num_envs
        self.rng = np.random.default_rng(seed)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
        cfg = FlySceneCfg(num_envs=num_envs, env_spacing=2.0)
        if video:
            # боковая камера только у первого стола, чисто для видео
            cfg.camera_side = CAMERA_SIDE.replace(prim_path="/World/envs/env_0/CameraSide", width=320, height=240)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.dev = self.sim.device
        self.dt = self.sim.get_physics_dt()
        self.origins = self.scene.env_origins  # [N,3], у каждого стола своё начало

        self.robot = self.scene["robot"]
        self.cam = self.scene["camera"]
        self.side = self.scene["camera_side"] if video else None
        self.objs = {n: self.scene[n] for n in OBJECTS}
        self.arm_ids, _ = self.robot.find_joints(list(ARM_JOINT), preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(FINGERS, preserve_order=True)
        self.ee_id = self.robot.find_bodies("hande_end")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), self.N, self.dev
        )
        self.rec = None  # пока пишем видео - список кадров
        self.n_tick = 0

        # кинематика планировщика под сим, как в top_plan
        r = self.robot
        for _ in range(5):
            r.set_joint_position_target(r.data.default_joint_pos.clone())
            self._tick()
        lim = getattr(r.data, "joint_pos_limits", None)
        lim = (r.data.joint_limits if lim is None else lim)[0, self.arm_ids]
        self.plan = P.Planner(lim[:, 0], lim[:, 1], self.dev)
        ee_pos, ee_quat = self._ee()
        jac = r.root_physx_view.get_jacobians()[0, self.ee_id - 1, :, self.arm_ids]
        diff = self.plan.calibrate(r.data.joint_pos[0, self.arm_ids], ee_pos[0], ee_quat[0], jac)
        if diff > 0.05:
            raise RuntimeError(f"кинематика планировщика не сошлась с симом, якобиан расходится на {diff:.3f}")

        # затравка для IK: кисть вниз над серединой стола. Ищу один раз из кучи стартов
        q_home = r.data.joint_pos[0, self.arm_ids].double()
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        rnd = torch.rand(127, 6, dtype=torch.float64, device=self.dev, generator=gen)
        lo, hi = self.plan.lo.clamp(min=-math.pi), self.plan.hi.clamp(max=math.pi)
        seeds = torch.cat([q_home.unsqueeze(0), lo + rnd * (hi - lo)])
        goal = torch.tensor([[0.0, 0.5, 0.25]], dtype=torch.float64, device=self.dev).expand(128, 3)
        qt = torch.tensor([QUAT_TOP], dtype=torch.float64, device=self.dev).expand(128, 4)
        q, err = ik_batch(self.plan, goal, qt, seeds, iters=300)
        ok = (err < 0.002) & (self.plan.lowest(q) > 0.05)
        if not ok.any():
            raise RuntimeError("не нашёл затравку для IK")
        q, m = q[ok], self.plan.manip(q[ok])
        good = q[m >= 0.7 * m.max()]
        self.q_seed = good[(good - q_home).norm(dim=1).argmin()]
        print(f"[fly] затравка IK {[round(v, 2) for v in self.q_seed.tolist()]}, manip {self.plan.manip(self.q_seed.unsqueeze(0)).item():.3f}", flush=True)

        # сетка рабочей зоны -> координаты для grid_sample по кадру камеры (строка 0 - дальний край стола, +y)
        xs = torch.linspace(WS_X[0], WS_X[1], IMG, device=self.dev)
        ys = torch.linspace(WS_Y[1], WS_Y[0], IMG, device=self.dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u = CAM_W / 2 + gx * FX / CAM_Z
        v = CAM_H / 2 - (gy - CAM_Y) * FX / CAM_Z
        self.grid = torch.stack([u / CAM_W * 2 - 1, v / CAM_H * 2 - 1], dim=-1).unsqueeze(0)  # [1,IMG,IMG,2]
        self.obj = ["tetrapak"] * self.N

    # --- мелочь

    def _ee(self):
        root = self.robot.data.root_state_w[:, 0:7]
        ee = self.robot.data.body_state_w[:, self.ee_id, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    def _tick(self, n=1):
        for _ in range(n):
            self.n_tick += 1
            frame = self.rec is not None and self.n_tick % 10 == 0
            self.scene.write_data_to_sim()
            self.sim.step(render=frame)
            self.scene.update(self.dt)
            if frame:
                self.rec.append(self.side.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

    # --- эпизод

    def reset(self, objs=None):
        """каждому столу случайный объект (или заданный), место, yaw, стоит/лежит. Рука домой"""
        r = self.robot
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(), torch.zeros_like(r.data.default_joint_vel))
        r.set_joint_position_target(r.data.default_joint_pos.clone())
        r.reset()

        self.obj = objs or [OBJECTS[i] for i in self.rng.integers(len(OBJECTS), size=self.N)]
        self.pose = []
        for name in OBJECTS:
            pos = torch.tensor(PARK[name], device=self.dev).repeat(self.N, 1)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.dev).repeat(self.N, 1)
            for i in range(self.N):
                if self.obj[i] != name:
                    continue
                poses = list(POSES[name])
                pname = poses[self.rng.integers(len(poses))]
                q_pose, z = POSES[name][pname]
                yaw = torch.tensor([self.rng.uniform(-math.pi, math.pi)], device=self.dev, dtype=torch.float32)
                pos[i] = torch.tensor([self.rng.uniform(*X_RANGE), self.rng.uniform(*Y_RANGE), z], device=self.dev)
                quat[i] = quat_mul(yaw_quat(yaw), torch.tensor([q_pose], device=self.dev))[0]
            o = self.objs[name]
            o.write_root_pose_to_sim(torch.cat([pos + self.origins, quat], dim=1))
            o.write_root_velocity_to_sim(torch.zeros(self.N, 6, device=self.dev))
            o.reset()
        self._tick(SETTLE)

        self.z0 = torch.stack([self.objs[self.obj[i]].data.root_pos_w[i, 2] for i in range(self.N)])
        self.sim.render()
        self.sim.render()  # первые кадры RTX бывают кривые, как в env.py
        self.scene.update(self.dt)
        rgb = self.cam.data.output["rgb"][..., :3].permute(0, 3, 1, 2).float()  # [N,3,H,W]
        depth = torch.nan_to_num(self.cam.data.output["distance_to_image_plane"][..., 0], posinf=0.0)
        height = torch.where(depth > 0, CAM_Z - depth, torch.zeros_like(depth)).unsqueeze(1)
        g = self.grid.expand(self.N, -1, -1, -1)
        rgb = F.grid_sample(rgb, g, align_corners=False)
        # высота - максимум в окошке 5x5, чтоб схват целился в верх, а не в край
        height = F.max_pool2d(F.grid_sample(height, g, align_corners=False), 5, stride=1, padding=2)
        self.height = height[:, 0].clamp(0, 0.25)  # [N,IMG,IMG] метры над столом
        img = torch.cat([rgb, self.height.unsqueeze(1) / 0.25 * 255], dim=1).clamp(0, 255).to(torch.uint8)
        return {"img": img, "obj": list(self.obj)}

    # перевод между пикселями агента и метрами
    @staticmethod
    def pix_to_xy(i, j):
        x = WS_X[0] + (j.float() + 0.5) / IMG * (WS_X[1] - WS_X[0])
        y = WS_Y[1] - (i.float() + 0.5) / IMG * (WS_Y[1] - WS_Y[0])
        return x, y

    @staticmethod
    def xy_to_pix(x, y):
        j = ((x - WS_X[0]) / (WS_X[1] - WS_X[0]) * IMG).long().clamp(0, IMG - 1)
        i = ((WS_Y[1] - y) / (WS_Y[1] - WS_Y[0]) * IMG).long().clamp(0, IMG - 1)
        return i, j

    def grasp(self, x, y, yaw):
        """x, y, yaw - тензоры [N] в осях стола. Вернёт награду [N] (0/1) и всякое для лога"""
        N, dev = self.N, self.dev
        i, j = self.xy_to_pix(x, y)
        top = self.height[torch.arange(N, device=dev), i, j]
        z = (top - GRIP_DEPTH).clamp(min=MIN_Z)
        grasp = torch.stack([x, y, z], dim=1)
        pre = grasp + torch.tensor([0.0, 0.0, ABOVE], device=dev)
        lifted = grasp + torch.tensor([0.0, 0.0, LIFT], device=dev)
        quat = quat_mul(yaw_quat(yaw), torch.tensor([QUAT_TOP], device=dev).expand(N, 4))

        # IK для точки над схватом, пробую yaw и yaw+180 (пальцам всё равно), беру что точнее и ближе к затравке
        quat2 = quat_mul(yaw_quat(yaw + math.pi), torch.tensor([QUAT_TOP], device=dev).expand(N, 4))
        seed = self.q_seed.unsqueeze(0).expand(2 * N, 6)
        q, err = ik_batch(self.plan, torch.cat([pre, pre]), torch.cat([quat, quat2]), seed)
        cost = err * 100 + (q - seed).norm(dim=1) * 0.01
        pick = (cost[N:] < cost[:N])
        q_arm = torch.where(pick.unsqueeze(1), q[N:], q[:N]).float()
        quat = torch.where(pick.unsqueeze(1), quat2, quat)
        reach = torch.where(pick, err[N:], err[:N]) < IK_POS_TOL

        # телепорт: рука сразу над точкой, пальцы раскрыты
        r = self.robot
        full = r.data.joint_pos.clone()
        full[:, self.arm_ids] = q_arm
        full[:, self.finger_ids] = OPEN
        r.write_joint_state_to_sim(full, torch.zeros_like(full))
        r.set_joint_position_target(full)
        self.ik.reset()
        ee_pos, _ = self._ee()
        start = ee_pos.clone()
        xy0 = torch.stack([self.objs[self.obj[k]].data.root_pos_w[k, :2] for k in range(N)])

        fingers = torch.full((N, 2), OPEN, device=dev)
        cmd = q_arm
        knocked = torch.zeros(N, dtype=torch.bool, device=dev)
        for name, n in PHASES:
            seg = {"descent": (start, grasp), "lift": (grasp, lifted)}.get(name)
            if name == "close":
                fingers[:] = CLOSED
                xy = torch.stack([self.objs[self.obj[k]].data.root_pos_w[k, :2] for k in range(N)])
                knocked = (xy - xy0).norm(dim=1) > 0.02
            for s in range(n):
                if seg is not None:
                    a = min(1.0, (s + 1) / (0.7 * n))
                    self.ik.set_command(torch.cat([seg[0] + a * (seg[1] - seg[0]), quat], dim=1))
                    ee_pos, ee_quat = self._ee()
                    jac = r.root_physx_view.get_jacobians()[:, self.ee_id - 1, :, self.arm_ids]
                    cmd = self.ik.compute(ee_pos, ee_quat, jac, r.data.joint_pos[:, self.arm_ids])
                r.set_joint_position_target(cmd, joint_ids=self.arm_ids)
                r.set_joint_position_target(fingers, joint_ids=self.finger_ids)
                self._tick()

        zf = torch.stack([self.objs[self.obj[k]].data.root_pos_w[k, 2] for k in range(N)])
        rise = zf - self.z0
        reward = ((rise >= LIFT_OK) & reach).float()
        return reward, {"rise": rise, "reach": reach, "knocked": knocked, "z": z}

    # --- оракул: настоящий центр и yaw объекта, как в env.oracle. Чисто проверить, что среда берёт
    def oracle(self):
        xs, ys, yaws = [], [], []
        for k in range(self.N):
            o = self.objs[self.obj[k]].data
            q = o.root_quat_w[k:k + 1]
            from isaaclab.utils.math import quat_apply

            c = o.root_pos_w[k] - self.origins[k] + quat_apply(q, torch.tensor([CENTER[self.obj[k]]], device=self.dev))[0]
            ax = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=self.dev))[0]
            xs.append(c[0])
            ys.append(c[1])
            yaws.append(torch.atan2(ax[1], ax[0]))
        return torch.stack(xs), torch.stack(ys), torch.stack(yaws)
