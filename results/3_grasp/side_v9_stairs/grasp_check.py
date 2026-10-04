"""Пробуем один нормальный схват: рукой рулит мой планировщик (planner.py), пальцами отдельно.
Сначала щёлкаем захватом в воздухе (видно по суставам пальцев, что открывается и закрывается),
потом подъезжаем к объекту, жмём, тащим вверх на 10+ см и держим 2 с.

v5: никакого IK по шажку. Планировщик сразу решает IK для каждой точки, берёт решение, ближайшее к тому,
где рука сейчас, и едет по прямой в углах суставов. Плюс стартуем не из домашней позы MetaIsaacGrasp,
а из позы, которую сам выбрал подальше от сингулярностей.
v6: сбоку подходим не со стороны робота (туда рука не дотягивается), а слева или справа от объекта.
v7: планировщик знает про стол и коробку объекта: если прямая в суставах во что-то врезается, берёт следующее
по близости решение, а если чистых нет - едет костылём-проекцией вдоль препятствия.
v8: рука едет в 2 раза медленнее, и между движениями пауза 0.5 с, чтоб успевала доехать до точки.
v9: суставы крутятся по одному (сначала один, потом другой), между ними микропауза. Плюс датчики контактов
на всех телах робота - если что-то врежется, в логе будет видно, каким звеном.

    bash server/run.sh grasp/grasp_check.py <gpu> [--obj tetrapak|can|chips] [--side] [--video]
"""

import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--obj", default="tetrapak", choices=["tetrapak", "can", "chips"])
parser.add_argument("--side", action="store_true", help="сбоку, а не сверху")
parser.add_argument("--video", action="store_true")
# без камер вообще - одна физика. Для A100, где рендера нет (RT-ядер нет). Кадров и гифки тогда не будет, только логи и csv
parser.add_argument("--no-cam", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from common import (
    ARM_JOINT, CAMERA_SIDE, CAMERA_TOP, CLOSED, FINGERS, GROUND, LIGHT, OPEN, ROBOT, SIDE, TABLE, UP,
    OUT, grab, obj_cfg, save_frame, save_gif,
)
from planner import Planner, along

OBJ_XY = (0.0, 0.5)
# где верх у объекта на столе: тетрапак и банка стоят, чипсы лежат (стоя они всё равно падают)
OBJ_TOP = {"tetrapak": 0.189, "can": 0.038, "chips": 0.044}
OBJ_ROT = {"tetrapak": UP, "can": UP, "chips": SIDE}
OBJ_Z0 = {"tetrapak": 0.005, "can": 0.005, "chips": 0.03}
# полуразмеры объекта в осях мира, как он стоит на столе (чипсы лежат, у них y и z поменялись). Для планировщика,
# чтоб он знал, где коробка. Списал из env.HALF
OBJ_HALF = {"tetrapak": (0.030, 0.030, 0.0945), "can": (0.040, 0.040, 0.019), "chips": (0.074, 0.1075, 0.024)}

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
# v8: всё движение в 2 раза медленнее (было 400/200/200), а после каждого движения пауза 0.5 с - цель стоит, рука
# доезжает. По графику v7 видно, что больше всего рука отстаёт как раз на смене фазы: цель уже поехала дальше, а рука
# ещё не доехала до прошлой точки
PAUSE = 50
# v9: в pregrasp/approach/lift суставы крутятся по одному. Сколько шагов фаза - уже не фиксировано, считается по пути:
# каждый сустав едет со скоростью JOINT_SPEED, а после каждого сустава микропауза JOINT_PAUSE. Число в PHASES для этих
# фаз больше не используется. Скорость взял как в v8 примерно (там самый быстрый сустав шёл ~0.7 рад/с)
JOINT_SPEED = 0.7  # рад/с
JOINT_PAUSE = 30  # 0.3 с
PHASES = [
    ("settle", 50),
    ("test_close", 80),
    ("test_open", 80),
    ("pregrasp", 800),
    ("wait_pre", PAUSE),
    ("approach", 400),
    ("wait_grasp", PAUSE),
    ("close", 100),
    ("lift", 400),
    ("wait_lift", PAUSE),
    ("hold", 200),
]


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground = GROUND
    light = LIGHT
    table = TABLE
    obj = obj_cfg(args.obj, (*OBJ_XY, OBJ_Z0[args.obj]), OBJ_ROT[args.obj])
    # v9: у робота включаю датчики контактов, чтоб видеть, каким звеном он задел коробку (в common они выключены)
    robot = ROBOT.replace(spawn=ROBOT.spawn.replace(activate_contact_sensors=True))
    if not args.no_cam:
        camera_top = CAMERA_TOP
        camera = CAMERA_SIDE
    # контакты всех тел робота: общая сила (обо что угодно - стол, коробка, сам об себя) и отдельно сила об коробку
    contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=1, filter_prim_paths_expr=["{ENV_REGEX_NS}/obj"]
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("setup done", flush=True)

    robot, obj = scene["robot"], scene["obj"]
    cam = None if args.no_cam else scene["camera"]
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
    # объект для планировщика - коробка на столе
    h = OBJ_HALF[args.obj]
    plan.add_box((OBJ_XY[0], OBJ_XY[1], h[2]), h)

    # --- стартовая поза без сингулярностей
    q_top = torch.tensor(QUAT_TOP, dtype=torch.float64, device=dev)
    ready_quats = [q_top, quat_mul(q_top.unsqueeze(0), torch.tensor([FLIP], dtype=torch.float64, device=dev))[0]]
    sols = []
    for quat in ready_quats:
        rnd = torch.rand(255, 6, dtype=torch.float64, device=dev, generator=gen)
        lo, hi = plan.lo.clamp(min=-3.1416), plan.hi.clamp(max=3.1416)
        seeds = torch.cat([q_home.double().unsqueeze(0), lo + rnd * (hi - lo)])
        q, pe, re = plan.ik(torch.tensor(READY_POS, dtype=torch.float64, device=dev), quat, seeds)
        ok = (pe < 0.002) & (re < 0.02) & (plan.lowest(q) > 0.03) & (plan.clearance(q).min(dim=1).values >= 0)
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
        # задача 1: к объекту ничем не прикасаемся. Задача 2: пальцам можно, они его обхватывают.
        # подъём: объект уже в руке, коробку не проверяем, только стол
        r1 = plan.solve(pre, quats, q_ready.double(), gen=gen, name=f"задача 1, напротив объекта ({side})")
        r2 = plan.solve(grasp, quats, r1[0], gen=gen, name=f"задача 2, схват ({side})", fingers_ok=True) if r1 else None
        r3 = plan.solve(lifted, quats, r2[0], gen=gen, name=f"подъём ({side})", obj=False) if r2 else None
        if r3 is None:
            continue
        d = (r1[0] - q_ready.double()).norm().item()
        if best is None or d < best[0]:
            best = (d, k, pre, r1, r2, r3)
    if best is None:
        print("СТОП: планировщик не нашёл, как доехать", flush=True)
        print("FAIL", flush=True)
        return
    _, k, pre, r1, r2, r3 = best
    print(f"еду со стороны {k}", flush=True)
    plan.path_check(r1[1], "задача 1")
    plan.path_check(r2[1], "задача 2", fingers_ok=True)
    plan.path_check(r3[1], "подъём", obj=False)
    # фаза -> (путь в суставах [M,6], куда должна приехать кисть)
    segs = {"pregrasp": (r1[1], pre), "approach": (r2[1], grasp), "lift": (r3[1], lifted)}

    fingers = torch.full((1, 2), OPEN, device=dev)
    arm_cmd = q_ready.float().unsqueeze(0)
    frames, obj_z0 = [], None
    # v7b: пишу каждый шаг план и факт по суставам, чтоб потом нарисовать, отстаёт рука или нет
    rows = []
    count = 0

    # v9 метрики: кто из тел робота чего касается. Пишу все шаги, где у какого-то тела сила > CONTACT_MIN
    sens = scene["contacts"]
    body_names = sens.body_names
    CONTACT_MIN = 0.5  # Н, меньше - считаю шумом
    contact_rows = []
    first_obj_hit = None
    obj_xy_start = obj.data.root_pos_w[0, :2].clone()
    obj_moved_at = None

    def expand(path):
        """путь [M,6] -> список целей по шагам. Каждый кусок едет со скоростью JOINT_SPEED,
        после куска - микропауза, если это лесенка (кусков мало). Путь от проекции едет без пауз"""
        cmds = []
        stairs = len(path) <= 7
        for k in range(len(path) - 1):
            d = (path[k + 1] - path[k]).abs().max().item()
            if d < 1e-4:
                continue  # этот сустав крутить не надо
            m = max(5, math.ceil(d / JOINT_SPEED / 0.01))
            for i in range(m):
                cmds.append(path[k] + (i + 1) / m * (path[k + 1] - path[k]))
            if stairs:
                cmds += [path[k + 1]] * JOINT_PAUSE
        return cmds

    for name, n in PHASES:
        seg = segs.get(name)
        if name in ("test_close", "close"):
            fingers[:] = CLOSED
        elif name in ("settle", "test_open", "pregrasp"):
            fingers[:] = OPEN
        if name == "approach":
            obj_z0 = obj.data.root_pos_w[0, 2].item()
        cmds = expand(seg[0]) if seg is not None else [None] * n

        for cmd in cmds:
            if cmd is not None:
                arm_cmd = cmd.float().unsqueeze(0)
            robot.set_joint_position_target(arm_cmd, joint_ids=arm_ids)
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            count += 1
            q_act = robot.data.joint_pos[0, arm_ids]
            # запас по моей модели для того, где рука реально сейчас (в подъёме коробку не смотрим, она в руке)
            clr = plan.clearance(q_act.double().unsqueeze(0), obj=name not in ("lift", "wait_lift", "hold"), fingers_ok=name in ("approach", "wait_grasp", "close")).min().item()
            rows.append([count * 0.01, name] + arm_cmd[0].tolist() + q_act.tolist() + [clr])
            # контакты: общая сила по каждому телу и сила об коробку
            f_net = sens.data.net_forces_w[0].norm(dim=-1)  # [тела]
            f_obj = sens.data.force_matrix_w[0, :, 0].norm(dim=-1) if sens.data.force_matrix_w is not None else torch.zeros_like(f_net)
            for b in torch.nonzero((f_net > CONTACT_MIN) | (f_obj > CONTACT_MIN)).flatten().tolist():
                contact_rows.append((count * 0.01, name, body_names[b], f_net[b].item(), f_obj[b].item()))
            if first_obj_hit is None and (f_obj > CONTACT_MIN).any() and name not in ("close", "lift", "wait_lift", "hold"):
                b = int(f_obj.argmax())
                first_obj_hit = (count * 0.01, name, body_names[b], f_obj[b].item())
                print(f"!!! первый контакт с коробкой: t={count * 0.01:.2f} с, фаза {name}, тело {body_names[b]}, сила {f_obj[b].item():.1f} Н", flush=True)
            if obj_moved_at is None and (obj.data.root_pos_w[0, :2] - obj_xy_start).norm().item() > 0.001 and name not in ("close", "lift", "wait_lift", "hold"):
                obj_moved_at = (count * 0.01, name)
                print(f"!!! коробка сдвинулась больше чем на 1 мм: t={count * 0.01:.2f} с, фаза {name}", flush=True)
            if args.video and cam is not None and count % 5 == 0:
                frames.append(grab(cam))

        ee_pos, _ = ee_pose()
        q = robot.data.joint_pos[0, finger_ids].tolist()
        oz = obj.data.root_pos_w[0, 2].item()
        err = (ee_pos[0] - seg[1]).norm().item() * 1000 if seg is not None else float("nan")
        # заодно сверяю мою кинематику с симом в новой позе: если тут сантиметры, то вся подгонка мимо
        fk_err = (plan.fk(robot.data.joint_pos[:, arm_ids].double())[0, :3, 3] - ee_pos[0].double()).norm().item() * 1000
        print(
            f"[{name:10s}] step {count:4d}  ee err {err:6.1f} mm  fingers {q[0]:.4f} {q[1]:.4f}  "
            f"obj z {oz:+.3f}  fk err {fk_err:.1f} mm",
            flush=True,
        )
        if name == "settle":
            if not args.no_cam:
                save_frame(scene["camera_top"])

    rise = obj.data.root_pos_w[0, 2].item() - obj_z0
    held = robot.data.joint_pos[0, finger_ids]
    print(f"object rise {rise * 100:.1f} cm, fingers at {held.tolist()}", flush=True)

    with open(os.path.join(OUT, "joints.csv"), "w") as f:
        f.write("t,phase," + ",".join(f"cmd{i}" for i in range(6)) + "," + ",".join(f"act{i}" for i in range(6)) + ",clearance\n")
        for r in rows:
            f.write(f"{r[0]:.2f},{r[1]}," + ",".join(f"{v:.5f}" for v in r[2:]) + "\n")
    print(f"joints saved, {len(rows)} steps", flush=True)
    with open(os.path.join(OUT, "contacts.csv"), "w") as f:
        f.write("t,phase,body,force_total,force_obj\n")
        for r in contact_rows:
            f.write(f"{r[0]:.2f},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.2f}\n")
    # сводка: по каждому телу - когда первый раз чего-то коснулось (до схвата) и максимальная сила об коробку
    print(f"contacts saved, {len(contact_rows)} строк", flush=True)
    seen = {}
    for t, ph, b, fn, fo in contact_rows:
        if ph in ("close", "lift", "wait_lift", "hold"):
            continue
        s = seen.setdefault(b, [t, ph, 0.0, 0.0])
        s[2], s[3] = max(s[2], fn), max(s[3], fo)
    for b, (t, ph, fn, fo) in sorted(seen.items(), key=lambda x: x[1][0]):
        print(f"[contact] {b:28s} первый раз t={t:.2f} ({ph}), макс сила всего {fn:.1f} Н, об коробку {fo:.1f} Н", flush=True)
    print(f"итог: первый контакт с коробкой {first_obj_hit}, коробка сдвинулась {obj_moved_at}", flush=True)

    if frames:
        save_gif(frames, f"grasp_{args.obj}{'_side' if args.side else ''}.gif", ms=50)
    print("PASS" if rise >= LIFT_OK else "FAIL", flush=True)


if __name__ == "__main__":
    main()
    app.close()
