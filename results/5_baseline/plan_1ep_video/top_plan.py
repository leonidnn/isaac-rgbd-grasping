"""Схват сверху через планировщик (идея 04.10):
1) цель фазы 1 - точка на 10 см выше схвата, кисть вниз с нужным yaw. IK пачкой даёт кучу решений в суставах
2) сортирую их по близости к текущим углам. Для каждого строю лесенку (суставы по одному) в двух порядках,
   ставлю на неё 50 точек и проверяю: стол, объект, рука сама в себя. Первое чистое - едем. Ни одного - провал
3) едем лесенкой: каждый сустав по очереди, 1 рад/с, между суставами пауза 0.5 с. Потом пауза 0.5 с
4) фаза 2: спуск строго вниз на 10 см обычным IK по шажку (все суставы вместе, кисть остаётся вертикально)
5) дальше как в env.py: сжать, поднять, держать

Кинематика, IK и шарики - из planner.py (v9), исполнение лесенки - из grasp_check.py v9.
Объект для проверки столкновений берётся из картинки (коробка по маске глубины), в симулятор не подглядываем.
Импортить после make_app().
"""

import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import planner as P
import vision
from common import CAMERA_SIDE, CAMERA_TOP, GROUND, LIGHT, ROBOT, TABLE, grab, obj_cfg
from env import CLOSED, LIFT, LIFT_OK, KNOCK, OBJECTS, OPEN, PARK, GraspEnv

ABOVE = 0.10  # фаза 1 приезжает на 10 см выше схвата
JOINT_SPEED = 1.0  # рад/с
JOINT_PAUSE = 50  # 0.5 с между суставами
PHASE_PAUSE = 50  # 0.5 с между фазами
CHECK_N = 50  # столько точек на лесенке проверяю
ORDERS = [(0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)]  # база->запястье; сначала плечо/локоть, потом база
DESCENT = 100  # спуск, шагов
AFTER = [("close", 60), ("lift", 100), ("hold", 100)]  # как в env.py
CONTACT_MIN = 0.5  # Н, меньше - шум

# самостолкновения: базу (колонну до плеча) тоже шариками, радиус на глаз
BASE_R = 0.08


@configclass
class PlanSceneCfg(InteractiveSceneCfg):
    ground = GROUND
    light = LIGHT
    table = TABLE
    # датчики контактов включаю, чтоб видеть, правда ли рука ни во что не упёрлась (модель шариков уже врала в v9)
    robot = ROBOT.replace(spawn=ROBOT.spawn.replace(activate_contact_sensors=True))
    camera = CAMERA_TOP
    tetrapak = obj_cfg("tetrapak", PARK["tetrapak"], prim="tetrapak")
    can = obj_cfg("can", PARK["can"], prim="can")
    chips = obj_cfg("chips", PARK["chips"], prim="chips")
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=1)


# --- проверка лесенки


def staircase(q_from, q_to, order):
    pts = [q_from.clone()]
    q = q_from.clone()
    for j in order:
        q = q.clone()
        q[j] = q_to[j]
        pts.append(q)
    return torch.stack(pts)


def sample(path, n=CHECK_N):
    """ровно n точек по лесенке, равномерно по сумме поворотов"""
    seg = (path[1:] - path[:-1]).abs().sum(dim=1)
    cum = torch.cat([torch.zeros(1, dtype=path.dtype, device=path.device), seg.cumsum(0)])
    if cum[-1] < 1e-9:
        return path[:1].expand(n, -1)
    s = torch.linspace(0, cum[-1].item(), n, dtype=path.dtype, device=path.device)
    k = (torch.searchsorted(cum, s, right=True) - 1).clamp(0, len(seg) - 1)
    a = ((s - cum[k]) / seg[k].clamp(min=1e-9)).unsqueeze(1)
    return path[k] + a * (path[k + 1] - path[k])


def self_clearance(plan, q):
    """рука сама в себя: шарики несоседних звеньев (плюс колонна базы). [N] - минимальный запас, <0 - врезались"""
    p, r, _ = plan.points(q)  # [N, 6*PTS, 3], 6 отрезков по PTS шариков: плечо-локоть ... фланец-кисть
    N = q.shape[0]
    seg_id = torch.arange(6, device=q.device).repeat_interleave(P.PTS)
    # колонна базы: от нуля до плеча (первая точка DH)
    org0 = (plan.T_pre @ P.dh_frames(q)[0])[:, :3, 3]
    t = torch.linspace(0, 1, P.PTS, dtype=q.dtype, device=q.device).view(1, -1, 1)
    base = t * org0.unsqueeze(1)
    p = torch.cat([base, p], dim=1)
    r = torch.cat([torch.full((P.PTS,), BASE_R, dtype=q.dtype, device=q.device), r])
    seg_id = torch.cat([torch.full((P.PTS,), -1, device=q.device), seg_id])
    # пары только несоседних кусков: соседние и так касаются в суставе
    far = (seg_id.view(-1, 1) - seg_id.view(1, -1)).abs() >= 2
    d = torch.cdist(p, p) - (r.view(-1, 1) + r.view(1, -1))
    d = torch.where(far.unsqueeze(0), d, torch.full_like(d, 1e3))
    return d.view(N, -1).min(dim=1).values - P.MARGIN


def check(plan, qs):
    """qs [M,6] -> (чисто ли, что задели): стол / объект / сам в себя"""
    p, r, _ = plan.points(qs)
    if (p[..., 2] - r - P.MARGIN).min().item() < 0:
        return False, "стол"
    if plan.clearance(qs, obj=True).min().item() < 0:
        return False, "объект"
    if self_clearance(plan, qs).min().item() < 0:
        return False, "сам в себя"
    return True, ""


class PlanEnv(GraspEnv):
    def __init__(self, seed=0, video=False):
        # то же, что GraspEnv.__init__, только сцена с датчиками контактов (env.py не трогаю)
        import numpy as np
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.rng = np.random.default_rng(seed)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
        cfg = PlanSceneCfg(num_envs=1, env_spacing=2.0)
        if video:
            cfg.camera_side = CAMERA_SIDE  # сбоку, чисто для видео
        self.scene = InteractiveScene(cfg)
        self.side = self.scene["camera_side"] if video else None
        self.rec = None  # список кадров, пока пишем видео, иначе None
        self.n_drive = 0
        self.sim.reset()
        self.dev = self.sim.device
        self.dt = self.sim.get_physics_dt()
        self.robot = self.scene["robot"]
        self.cam = self.scene["camera"]
        self.objs = {n: self.scene[n] for n in OBJECTS}
        from common import ARM_JOINT, FINGERS

        self.arm_ids, _ = self.robot.find_joints(list(ARM_JOINT), preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(FINGERS, preserve_order=True)
        self.ee_id = self.robot.find_bodies("hande_end")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, self.dev
        )
        self.obj = None
        self.z0 = None
        self.xy0 = None
        self.sens = self.scene["contacts"]
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

        # подгоняю кинематику планировщика под сим в домашней позе, как в grasp_check v9
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
        q0 = r.data.joint_pos[0, self.arm_ids].double().unsqueeze(0)
        print(f"[plan] домашняя поза: запас сам-в-себя по модели {self_clearance(self.plan, q0).item() * 1000:.0f} мм", flush=True)

    def set_obstacle(self, depth):
        """коробка объекта по маске глубины (то, что видит камера), а не из симулятора"""
        m, x, y, z = vision.mask(depth)
        self.plan.boxes = []
        if m.sum() < vision.MIN_PIX:
            return None
        lo = (float(x[m].min()), float(y[m].min()), 0.0)
        hi = (float(x[m].max()), float(y[m].max()), float(z[m].max()))
        cen = tuple((a + b) / 2 for a, b in zip(lo, hi))
        half = tuple((b - a) / 2 for a, b in zip(lo, hi))
        self.plan.add_box(cen, half)
        return cen, half

    def _plan(self, pos, quat):
        plan, t0 = self.plan, time.time()
        q_ref = self.robot.data.joint_pos[0, self.arm_ids].double()
        rnd = torch.rand(63, 6, dtype=torch.float64, device=self.dev, generator=self.gen)
        lo, hi = plan.lo.clamp(min=-math.pi), plan.hi.clamp(max=math.pi)
        seeds = torch.cat([q_ref.unsqueeze(0), lo + rnd * (hi - lo)])
        q, pe, re = plan.ik(pos.double(), quat.double(), seeds)
        ok = (pe < P.POS_TOL) & (re < P.ROT_TOL)
        sols = q[ok]
        info = {"ik_ok": int(ok.sum()), "tried": 0, "hits": {}, "plan_s": 0.0}
        if len(sols) == 0:
            info["plan_s"] = time.time() - t0
            return None, info
        sols = sols[(sols - q_ref).norm(dim=1).argsort()]
        for k in range(len(sols)):
            for oi, order in enumerate(ORDERS):
                info["tried"] += 1
                path = staircase(q_ref, sols[k], order)
                free, what = check(plan, sample(path))
                if free:
                    info.update(sol=k, order=oi, dist=(sols[k] - q_ref).norm().item(), plan_s=time.time() - t0)
                    return path, info
                info["hits"][what] = info["hits"].get(what, 0) + 1
        info["plan_s"] = time.time() - t0
        return None, info

    def _drive(self, cmd, fingers):
        self.robot.set_joint_position_target(cmd.float().unsqueeze(0), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(fingers, joint_ids=self.finger_ids)
        # пишем видео - каждый 5-й шаг рендерим и берём кадр с боковой камеры (как в grasp_check --video)
        self.n_drive += 1
        frame = self.rec is not None and self.n_drive % 5 == 0
        self._tick(render=frame)
        if frame:
            self.rec.append(grab(self.side))
        f = self.sens.data.net_forces_w[0].norm(dim=-1)
        if f.max().item() > self.contact[1]:
            self.contact = (self.sens.body_names[int(f.argmax())], f.max().item())

    def step_plan(self, x, y, z, yaw):
        quat = self.hand_quat(yaw)
        grasp = torch.tensor([x, y, z], dtype=torch.float32, device=self.dev)
        pre = grasp + torch.tensor([0.0, 0.0, ABOVE], device=self.dev)
        lifted = grasp + torch.tensor([0.0, 0.0, LIFT], device=self.dev)
        fingers = torch.full((1, 2), OPEN, device=self.dev)
        obj = self.objs[self.obj].data
        self.contact = ("", 0.0)  # самый сильный контакт руки до сжатия пальцев

        path, info = self._plan(pre, quat)
        res = {"success": False, "rise": 0.0, "knocked": False, "ee_err": float("nan"), **info}
        if path is None:
            res["fail"] = "нет пути"
            return res

        # фаза 1: лесенкой, суставы по одному
        cmd = path[0]
        for k in range(len(path) - 1):
            d = (path[k + 1] - path[k]).abs().max().item()
            if d < 1e-4:
                continue
            m = max(5, math.ceil(d / JOINT_SPEED / self.dt))
            for i in range(m):
                cmd = path[k] + (i + 1) / m * (path[k + 1] - path[k])
                self._drive(cmd, fingers)
            for _ in range(JOINT_PAUSE):
                self._drive(cmd, fingers)
        for _ in range(PHASE_PAUSE - JOINT_PAUSE):  # после последнего сустава уже 0.5 с отстояли
            self._drive(cmd, fingers)
        ee_pos, _ = self._ee()
        res["pre_err"] = (ee_pos[0] - pre).norm().item()

        # фаза 2: вниз по прямой, IK по шажку как в env.step
        self.ik.reset()
        start = ee_pos[0].clone()
        res["knocked"] = False
        for name, n in [("descent", DESCENT)] + AFTER:
            seg = {"descent": (start, grasp), "lift": (grasp, lifted)}.get(name)
            if name == "close":
                fingers[:] = CLOSED
                res["knocked"] = (obj.root_pos_w[0, :2] - self.xy0).norm().item() > KNOCK
                res["contact_before_close"] = self.contact
            for i in range(n):
                if seg is not None:
                    a = min(1.0, (i + 1) / (0.7 * n))
                    self.ik.set_command(torch.cat([seg[0] + a * (seg[1] - seg[0]), quat]).unsqueeze(0))
                    ee_pos, ee_quat = self._ee()
                    jac = self.robot.root_physx_view.get_jacobians()[:, self.ee_id - 1, :, self.arm_ids]
                    cmd = self.ik.compute(ee_pos, ee_quat, jac, self.robot.data.joint_pos[:, self.arm_ids])[0]
                self._drive(cmd, fingers)
            if name == "descent":
                ee_pos, _ = self._ee()
                res["grasp_err"] = (ee_pos[0] - grasp).norm().item()

        ee_pos, _ = self._ee()
        rise = obj.root_pos_w[0, 2].item() - self.z0
        res.update(success=rise >= LIFT_OK, rise=rise, ee_err=(ee_pos[0] - lifted).norm().item())
        return res
