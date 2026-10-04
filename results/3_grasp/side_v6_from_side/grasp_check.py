"""Пробуем один нормальный схват: рукой рулит мой планировщик (planner.py), пальцами отдельно.
Сначала щёлкаем захватом в воздухе (видно по суставам пальцев, что открывается и закрывается),
потом подъезжаем к объекту, жмём, тащим вверх на 10+ см и держим 2 с.

v5: никакого IK по шажку. Планировщик сразу решает IK для каждой точки, берёт решение, ближайшее к тому,
где рука сейчас, и едет по прямой в углах суставов. Плюс стартуем не из домашней позы MetaIsaacGrasp,
а из позы, которую сам выбрал подальше от сингулярностей.
v6: сбоку подходим не со стороны робота (туда рука не дотягивается), а слева или справа от объекта.

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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from common import (
    ARM_JOINT, CAMERA_SIDE, CAMERA_TOP, CLOSED, FINGERS, GROUND, LIGHT, OPEN, ROBOT, SIDE, TABLE, UP,
    grab, obj_cfg, save_frame, save_gif,
)
from planner import Planner

OBJ_XY = (0.0, 0.5)
# где верх у объекта на столе: тетрапак и банка стоят, чипсы лежат (стоя они всё равно падают)
OBJ_TOP = {"tetrapak": 0.189, "can": 0.038, "chips": 0.044}
OBJ_ROT = {"tetrapak": UP, "can": UP, "chips": SIDE}
OBJ_Z0 = {"tetrapak": 0.005, "can": 0.005, "chips": 0.03}

# пальцы у hande_end торчат по его оси y, а не z, как в URDF (проверил прогоном, results/3_grasp).
# сверху: -90 вокруг X, y кисти уходит в -z мира
QUAT_TOP = (0.70710678, -0.70710678, 0.0, 0.0)
# сбоку: кисть как есть, y кисти смотрит от робота (+y мира). Так подходить нельзя - запястье с захватом
# не влезает между базой и объектом, IK промахивается на 10 см (v5). Оставил как базу, от неё крутим ниже
QUAT_SIDE = (1.0, 0.0, 0.0, 0.0)
# v6: подходим к объекту сбоку, а не со стороны робота: пальцы вдоль оси x мира. Это QUAT_SIDE, повёрнутый на
# -90 вокруг z (пальцы в +x, рука слева от объекта) или на +90 (пальцы в -x, рука справа). С какой стороны
# заходить, решает планировщик - где решение ближе к стартовой позе
SIDE_YAWS = (-0.70710678, 0.70710678)  # sin(угол/2) для поворота на -90 и +90 вокруг z
# кисть, повёрнутая на 180 вокруг своей оси y (вдоль пальцев), хватает точно так же - пальцы симметричные.
# планировщику даю оба варианта, пусть сам выберет, какой ближе
FLIP = (0.0, 0.0, 1.0, 0.0)

GRIP_DEPTH = 0.03  # насколько пальцы залезают на объект сверху
# задача 1: встаём напротив объекта, будто он на PRE ближе к роботу (пальцы уже параллельно ему, как при схвате).
# задача 2: оттуда двигаемся к объекту и хватаем
PRE = 0.10
LIFT = 0.15
LIFT_OK = 0.10

# стартовая поза: кисть смотрит вниз над столом, чуть ближе к роботу, чем объект. Суставы для неё ищет планировщик,
# из решений беру не самое близкое к домашней позе, а среди хорошо обусловленных (manip не меньше 70% от лучшего)
READY_POS = (0.0, 0.35, 0.45)
READY_GOOD = 0.7

# фазы расписал сам, подсмотрел у стейт-машины MetaIsaacGrasp (air_env_base: reach/approach/grasp/lift),
# только без warp и под один объект. (имя, сколько шагов), шаг 0.01 с
PHASES = [
    ("settle", 50),
    ("test_close", 80),
    ("test_open", 80),
    ("pregrasp", 400),
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
    ee_jacobi_idx = ee_id - 1  # база прибита, поэтому -1

    limits = getattr(robot.data, "joint_pos_limits", None)
    if limits is None:
        limits = robot.data.joint_limits
    print(f"finger limits {limits[0, finger_ids].tolist()}", flush=True)

    # ee_pose стащил из run_simulator в scene_check
    def ee_pose():
        root = robot.data.root_state_w[:, 0:7]
        ee = robot.data.body_state_w[:, ee_id, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    # пару шагов постоять дома, чтоб у руки обновились позы и якобиан, а то сразу после reset хз что там лежит
    for _ in range(5):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    # --- планировщик: подгоняем мою кинематику под сим, пока рука стоит в домашней позе
    plan = Planner(limits[0, arm_ids, 0], limits[0, arm_ids, 1], dev)
    q_home = robot.data.joint_pos[0, arm_ids].clone()
    ee_pos, ee_quat = ee_pose()
    jac_sim = robot.root_physx_view.get_jacobians()[0, ee_jacobi_idx, :, arm_ids]
    diff = plan.calibrate(q_home, ee_pos[0], ee_quat[0], jac_sim)
    if diff > 0.05:
        print(f"СТОП: моя кинематика не совпала с симом (якобиан расходится на {diff:.3f}), дальше смысла нет", flush=True)
        print("FAIL", flush=True)
        return
    # симовский якобиан тоже, чтоб проверить, правда ли домашняя поза в сингулярности (в v3/v4 был минимум 0 за фазу)
    m_sim = torch.sqrt(torch.clamp(torch.det(jac_sim @ jac_sim.T), min=0)).item()
    print(f"домашняя поза: manip по симу {m_sim:.4f}, по моей кинематике {plan.manip(q_home.double().unsqueeze(0)).item():.4f}", flush=True)

    gen = torch.Generator(device=dev).manual_seed(0)

    # --- стартовая поза без сингулярностей
    q_top = torch.tensor(QUAT_TOP, dtype=torch.float64, device=dev)
    ready_quats = [q_top, quat_mul(q_top.unsqueeze(0), torch.tensor([FLIP], dtype=torch.float64, device=dev))[0]]
    sols = []
    for quat in ready_quats:
        rnd = torch.rand(255, 6, dtype=torch.float64, device=dev, generator=gen)
        lo, hi = plan.lo.clamp(min=-3.1416), plan.hi.clamp(max=3.1416)
        seeds = torch.cat([q_home.double().unsqueeze(0), lo + rnd * (hi - lo)])
        q, pe, re = plan.ik(torch.tensor(READY_POS, dtype=torch.float64, device=dev), quat, seeds)
        ok = (pe < 0.002) & (re < 0.02) & (plan.lowest(q) > 0.03)
        sols.append(q[ok])
    sols = torch.cat(sols)
    if len(sols) == 0:
        print("СТОП: не нашёл ни одной стартовой позы", flush=True)
        print("FAIL", flush=True)
        return
    m = plan.manip(sols)
    good = sols[m >= READY_GOOD * m.max()]
    q_ready = good[(good - q_home.double()).norm(dim=1).argmin()]
    print(
        f"стартовая поза: решений {len(sols)}, manip лучший {m.max().item():.4f}, "
        f"у выбранной {plan.manip(q_ready.unsqueeze(0)).item():.4f}, суставы {[round(v, 3) for v in q_ready.tolist()]}",
        flush=True,
    )
    # телепортом ставим руку в стартовую позу, пока ничего не началось
    full = robot.data.joint_pos.clone()
    full[0, arm_ids] = q_ready.float()
    robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    robot.set_joint_position_target(full)
    scene.write_data_to_sim()

    # --- цели. В координатах базы робота, но база стоит в нуле, так что это то же самое, что мир
    top = OBJ_TOP[args.obj]
    if args.side:
        grasp = torch.tensor([OBJ_XY[0], OBJ_XY[1], top / 2], device=dev)
        bases = [
            quat_mul(torch.tensor([[0.70710678, 0.0, 0.0, s]], device=dev), torch.tensor([QUAT_SIDE], device=dev))[0]
            for s in SIDE_YAWS
        ]
    else:
        grasp = torch.tensor([OBJ_XY[0], OBJ_XY[1], max(top - GRIP_DEPTH, 0.015)], device=dev)
        bases = [torch.tensor(QUAT_TOP, device=dev)]
    lifted = grasp + torch.tensor([0.0, 0.0, LIFT], device=dev)

    # --- планируем всё заранее: каждая следующая точка ищется ближайшей к предыдущей, чтоб рука не перекручивалась.
    # сбоку пробую обе стороны, беру ту, где точка напротив объекта ближе к стартовой позе
    best = None
    for k, base_q in enumerate(bases):
        quats = [base_q, quat_mul(base_q.unsqueeze(0), torch.tensor([FLIP], device=dev))[0]]
        # куда торчат пальцы - вдоль этого и подъезжаем
        fwd = quat_apply(base_q.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
        pre = grasp - PRE * fwd
        side = f"сторона {k}, пальцы в {[round(v, 2) for v in fwd.tolist()]}"
        q_pre = plan.solve(pre, quats, q_ready, gen=gen, name=f"задача 1, напротив объекта ({side})")
        q_grasp = plan.solve(grasp, quats, q_pre, gen=gen, name=f"задача 2, схват ({side})") if q_pre is not None else None
        q_lift = plan.solve(lifted, quats, q_grasp, gen=gen, name=f"подъём ({side})") if q_grasp is not None else None
        if q_lift is None:
            continue
        d = (q_pre - q_ready.double()).norm().item()
        if best is None or d < best[0]:
            best = (d, k, pre, q_pre, q_grasp, q_lift)
    if best is None:
        print("СТОП: планировщик не нашёл, как доехать", flush=True)
        print("FAIL", flush=True)
        return
    _, k, pre, q_pre, q_grasp, q_lift = best
    print(f"еду со стороны {k}", flush=True)
    plan.path_check(q_ready, q_pre, "задача 1")
    plan.path_check(q_pre, q_grasp, "задача 2")
    plan.path_check(q_grasp, q_lift, "подъём")
    segs = {"pregrasp": (q_ready, q_pre, pre), "approach": (q_pre, q_grasp, grasp), "lift": (q_grasp, q_lift, lifted)}

    fingers = torch.full((1, 2), OPEN, device=dev)
    arm_cmd = q_ready.float().unsqueeze(0)
    frames, obj_z0 = [], None
    count = 0

    for name, n in PHASES:
        seg = segs.get(name)
        if name in ("test_close", "close"):
            fingers[:] = CLOSED
        elif name in ("settle", "test_open", "pregrasp"):
            fingers[:] = OPEN
        if name == "approach":
            obj_z0 = obj.data.root_pos_w[0, 2].item()

        for i in range(n):
            if seg is not None:
                # по прямой в углах суставов, за 70% фазы, остальное время рука доезжает
                a = min(1.0, (i + 1) / (0.7 * n))
                arm_cmd = (seg[0] + a * (seg[1] - seg[0])).float().unsqueeze(0)
            robot.set_joint_position_target(arm_cmd, joint_ids=arm_ids)
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
        err = (ee_pos[0] - seg[2]).norm().item() * 1000 if seg is not None else float("nan")
        # заодно сверяю мою кинематику с симом в новой позе: если тут сантиметры, то вся подгонка мимо
        fk_err = (plan.fk(robot.data.joint_pos[:, arm_ids].double())[0, :3, 3] - ee_pos[0].double()).norm().item() * 1000
        print(
            f"[{name:10s}] step {count:4d}  ee err {err:6.1f} mm  fingers {q[0]:.4f} {q[1]:.4f}  "
            f"obj z {oz:+.3f}  fk err {fk_err:.1f} mm",
            flush=True,
        )
        if name == "settle":
            save_frame(scene["camera_top"])

    rise = obj.data.root_pos_w[0, 2].item() - obj_z0
    held = robot.data.joint_pos[0, finger_ids]
    print(f"object rise {rise * 100:.1f} cm, fingers at {held.tolist()}", flush=True)

    if frames:
        save_gif(frames, f"grasp_{args.obj}{'_side' if args.side else ''}.gif", ms=50)
    print("PASS" if rise >= LIFT_OK else "FAIL", flush=True)


if __name__ == "__main__":
    main()
    app.close()
