"""Мой планировщик для UR10e. Идея тупая:
1) решаем IK для конечной точки сразу целиком (не по шажку), из кучи случайных стартов, получаем несколько решений
2) берём то, которое ближе всего к тому, где рука сейчас (по углам суставов)
3) едем по прямой, но не в декартовом пространстве, а в пространстве углов. Никакой оптимизации

Своя прямая кинематика по DH, чтоб гонять IK на тысяче конфигураций разом без всякой физики.
DH у UR10e с сайта Universal Robots (стандартные DH). Сим может отличаться от DH на поворот базы и на то,
где сидит hande_end, поэтому calibrate() подгоняет это по настоящей руке и сверяет якобиан с симовским.

Импортить после make_app(), как и common.
"""

import math

import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

# DH у UR10e: d, a, alpha. Метры, радианы
DH_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
DH_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)

POS_TOL = 0.002  # решение IK считается решением, если промах меньше 2 мм
ROT_TOL = 0.02  # и поворот меньше ~1 градуса
MIN_Z = 0.03  # локоть, запястья и кисть должны быть выше стола (верх стола z=0), иначе решение выкидываем

# --- столкновения (v7). Руку заменяю шариками вдоль звеньев: на каждом отрезке между суставами DH по PTS штук.
# стол - плоскость z=0, объект - коробка. Радиусы на глаз с запасом, по фоткам UR10e и Hand-E
# ВАЖНО: отрезки идут по точкам DH, а настоящие плечо и предплечье у UR сдвинуты вбок сантиметров на 15-17,
# так что для плеча это очень приблизительно. Для запястий и захвата норм, а нас как раз они и бесят
LINK_R = (0.06, 0.05, 0.05, 0.05, 0.045)  # плечо-локоть, локоть-запястье1, запястье1-2, 2-3, 3-фланец
HAND_R = 0.045  # корпус захвата от фланца до hande_end
HAND_FREE = 0.09  # передние 9 см захвата - это пальцы, при схвате им объект задевать можно, они его и обхватывают
PTS = 6
MARGIN = 0.01  # зазор, чтоб не впритык
PATH_N = 50  # на сколько кусков режу прямую в суставах при проверке
# костыль-проекция: шаг по суставам и сколько шагов максимум
PROJ_STEP = 0.02
PROJ_MAX = 1500
PROJ_ACTIVE = 0.02  # ограничение считаю активным, если до препятствия осталось меньше 2 см


def make_T(pos, quat):
    T = torch.eye(4, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)
    T[:, :3, :3] = matrix_from_quat(quat)
    T[:, :3, 3] = pos
    return T


def rotz(a, dev):
    T = torch.eye(4, dtype=torch.float64, device=dev)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
    return T


def dh_frames(q):
    """q [N,6] -> список из 6 матриц [N,4,4], где каждое звено в осях базы DH"""
    N = q.shape[0]
    T = torch.eye(4, dtype=q.dtype, device=q.device).repeat(N, 1, 1)
    out = []
    for i in range(6):
        ct, st = torch.cos(q[:, i]), torch.sin(q[:, i])
        ca, sa = math.cos(DH_ALPHA[i]), math.sin(DH_ALPHA[i])
        A = torch.zeros(N, 4, 4, dtype=q.dtype, device=q.device)
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, DH_A[i] * ct
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, DH_A[i] * st
        A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, DH_D[i]
        A[:, 3, 3] = 1.0
        T = T @ A
        out.append(T)
    return out


def rot_err(R, R_goal):
    """насколько надо докрутить R до R_goal, вектором ось*угол в осях мира"""
    # через кватернион, а не через матрицу напрямую: на 180 градусах матричная формула даёт ноль и врёт, что всё ок
    qe = quat_from_matrix(R_goal @ R.transpose(1, 2))
    qe = torch.where(qe[:, :1] < 0, -qe, qe)  # q и -q одно и то же, берём короткий поворот
    xyz = qe[:, 1:]
    s = xyz.norm(dim=1, keepdim=True)
    ang = 2 * torch.atan2(s, qe[:, :1])
    return torch.where(s > 1e-9, xyz / s.clamp(min=1e-9) * ang, 2 * xyz)


# v9: в каком порядке крутить суставы по одному. От базы к кисти, взял просто по порядку, без всякой причины
JOINT_ORDER = (0, 1, 2, 3, 4, 5)


def staircase(q_from, q_to):
    """путь «лесенкой»: сначала крутим только один сустав, потом только следующий и т.д. [7,6]"""
    pts = [q_from.clone()]
    q = q_from.clone()
    for j in JOINT_ORDER:
        q = q.clone()
        q[j] = q_to[j]
        pts.append(q)
    return torch.stack(pts)


def densify(path, n, step=0.02):
    """путь из точек [M,6] -> точки почаще. Прямую из двух точек режу на n частей,
    а если точек больше - каждый кусок режу так, чтоб ни один сустав не прыгал больше чем на step"""
    if len(path) == 2:
        a = torch.linspace(0, 1, n, dtype=path.dtype, device=path.device).unsqueeze(1)
        return path[0] + a * (path[-1] - path[0])
    out = [path[:1]]
    for k in range(len(path) - 1):
        m = max(1, math.ceil((path[k + 1] - path[k]).abs().max().item() / step))
        a = torch.linspace(0, 1, m + 1, dtype=path.dtype, device=path.device)[1:].unsqueeze(1)
        out.append(path[k] + a * (path[k + 1] - path[k]))
    return torch.cat(out)


def along(path, a):
    """точка на пути [M,6] при a от 0 до 1, между соседними точками по прямой"""
    s = a * (len(path) - 1)
    i = min(int(s), len(path) - 2)
    return path[i] + (s - i) * (path[i + 1] - path[i])


class Planner:
    def __init__(self, lo, hi, dev):
        self.dev = dev
        self.lo, self.hi = lo.double(), hi.double()  # лимиты суставов руки [6]
        self.T_pre = torch.eye(4, dtype=torch.float64, device=dev)  # мир -> база DH
        self.T_post = torch.eye(4, dtype=torch.float64, device=dev)  # 6-е звено DH -> hande_end
        self.boxes = []  # препятствия-коробки: (центр [3], полуразмеры [3]), выровнены по осям мира
        # v10: то, что висит на запястье (кронштейн камеры, камера, корпус захвата) - точки в осях 6-го звена DH.
        # В v9 оказалось, что камерой рука упирается в стол, а в модели её не было вообще
        self.extra_local = torch.zeros(0, 3, dtype=torch.float64, device=dev)
        self.extra_r = torch.zeros(0, dtype=torch.float64, device=dev)

    # --- кинематика

    def fk(self, q):
        """q [N,6] -> поза hande_end [N,4,4] в мире (база робота в нуле)"""
        return self.T_pre @ dh_frames(q)[-1] @ self.T_post

    def lowest(self, q):
        """самая низкая точка из локтя, запястий и кисти - чтоб не лезть в стол"""
        fr = dh_frames(q)
        zs = [(self.T_pre @ fr[i])[:, 2, 3] for i in (1, 2, 3, 4, 5)] + [self.fk(q)[:, 2, 3]]
        return torch.stack(zs, dim=1).min(dim=1).values

    def jac(self, q, eps=1e-6):
        """якобиан численно, [N,6,6]: сверху скорость кисти, снизу угловая, всё в мире, как у physx"""
        T0 = self.fk(q)
        cols = []
        for i in range(6):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            T1 = self.fk(q + dq)
            lin = (T1[:, :3, 3] - T0[:, :3, 3]) / eps
            ang = rot_err(T0[:, :3, :3], T1[:, :3, :3]) / eps
            cols.append(torch.cat([lin, ang], dim=1))
        return torch.stack(cols, dim=2)

    # --- столкновения

    def add_box(self, center, half):
        self.boxes.append((torch.tensor(center, dtype=torch.float64, device=self.dev),
                           torch.tensor(half, dtype=torch.float64, device=self.dev)))

    def points(self, q):
        """шарики по руке: центры [N,P,3], радиусы [P], и какие из них - пальцы (передние HAND_FREE захвата)"""
        fr = dh_frames(q)
        org = [(self.T_pre @ f)[:, :3, 3] for f in fr]
        ee = self.fk(q)[:, :3, 3]
        segs = [(org[i], org[i + 1], LINK_R[i]) for i in range(5)] + [(org[5], ee, HAND_R)]
        t = torch.linspace(0, 1, PTS, dtype=q.dtype, device=self.dev)
        hand_len = self.T_post[:3, 3].norm()
        pts, rs, front = [], [], []
        for k, (a, b, r) in enumerate(segs):
            pts.append(a.unsqueeze(1) + t.view(1, -1, 1) * (b - a).unsqueeze(1))
            rs.append(torch.full((PTS,), r, dtype=q.dtype, device=self.dev))
            front.append((1 - t) * hand_len < HAND_FREE if k == 5 else torch.zeros(PTS, dtype=torch.bool, device=self.dev))
        if len(self.extra_local):
            # точки запястных железок: в осях 6-го звена они стоят на месте, просто переносим их в мир
            T6 = self.T_pre @ fr[5]
            pts.append((T6[:, :3, :3] @ self.extra_local.T).transpose(1, 2) + T6[:, :3, 3].unsqueeze(1))
            rs.append(self.extra_r)
            # захват смотрит вдоль оси z 6-го звена, hande_end на T_post[2,3]. Что ближе к нему чем HAND_FREE - это уже
            # пальцы и их крепления, при схвате им коробку трогать можно
            front.append(self.extra_local[:, 2] > self.T_post[2, 3] - HAND_FREE)
        return torch.cat(pts, dim=1), torch.cat(rs), torch.cat(front)

    def add_wrist_box(self, q0, lo_w, hi_w, r=0.01, n=3):
        """железка на запястье габаритом lo_w..hi_w (в мире, при позе q0). Ставлю сетку n*n*n шариков по её коробке
        и запоминаю их в осях 6-го звена. Для стола хватило бы углов (нижняя точка коробки - всегда угол),
        а сетка - чтоб и об пачку ребром не пролезла"""
        T6 = self.T_pre @ dh_frames(q0.double().unsqueeze(0))[5][0]
        lo, hi = torch.tensor(lo_w, dtype=torch.float64, device=self.dev), torch.tensor(hi_w, dtype=torch.float64, device=self.dev)
        a = torch.linspace(0, 1, n, dtype=torch.float64, device=self.dev)
        g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), dim=-1).reshape(-1, 3)
        pw = lo + g * (hi - lo)
        local = (pw - T6[:3, 3]) @ T6[:3, :3]  # мир -> оси 6-го звена
        self.extra_local = torch.cat([self.extra_local, local])
        self.extra_r = torch.cat([self.extra_r, torch.full((len(local),), r, dtype=torch.float64, device=self.dev)])

    def clearance(self, q, obj=True, fingers_ok=False):
        """сколько осталось до стола/коробок у каждого шарика, [N,P]. Меньше нуля - врезались.
        obj=False - коробки не смотрим (когда объект уже в руке). fingers_ok - пальцам объект задевать можно"""
        p, r, front = self.points(q)
        c = p[..., 2] - r  # стол
        if obj:
            for cen, half in self.boxes:
                d = (p - cen).abs() - half
                # снаружи - расстояние до коробки, внутри - минус глубина
                co = d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0) - r
                if fingers_ok:
                    co = torch.where(front, torch.ones_like(co), co)
                c = torch.minimum(c, co)
        return c - MARGIN

    def path_free(self, path, obj=True, fingers_ok=False):
        """путь из точек [M,6], между соседними режу на куски. Вернёт (чисто ли, во что врезались первым)"""
        qs = densify(path, PATH_N)
        p, r, front = self.points(qs)
        table = (p[..., 2] - r - MARGIN).min().item() < 0
        box = (self.clearance(qs, obj, fingers_ok).min().item() < 0) and not table
        return not (table or box), ("стол" if table else "коробка" if box else "")

    def clear_grad(self, q, obj, fingers_ok, eps=1e-5):
        """запасы шариков [P] и их производные по суставам [P,6], численно"""
        c0 = self.clearance(q.unsqueeze(0), obj, fingers_ok)[0]
        qs = q.unsqueeze(0) + eps * torch.eye(6, dtype=q.dtype, device=self.dev)
        G = ((self.clearance(qs, obj, fingers_ok) - c0) / eps).T
        return c0, G

    def project_path(self, q_from, q_to, obj=True, fingers_ok=False):
        """КОСТЫЛЬ. Едем к q_to мелкими шагами, но если шаг тащит какой-то шарик в стол или коробку,
        выкидываю из шага эту составляющую (проецирую шаг на то, что не уменьшает запас). Если уже залезли -
        ещё и выталкиваю. Застряли (шаг стал нулевой) или не доехали за PROJ_MAX - None"""
        q, q_to = q_from.double().clone(), q_to.double()
        path = [q.clone()]
        for _ in range(PROJ_MAX):
            d = q_to - q
            if d.abs().max() < 0.005:
                path.append(q_to.clone())
                return torch.stack(path)
            dq = d * min(1.0, PROJ_STEP / d.abs().max().item())
            c, G = self.clear_grad(q, obj, fingers_ok)
            # уже внутри - толкаем наружу по градиенту
            for j in torch.nonzero(c < 0).flatten().tolist():
                g = G[j]
                dq = dq + g * (-c[j]) / g.dot(g).clamp(min=1e-9)
            # проекция: по очереди выкидываю составляющие, которые лезут в препятствие
            act = c < PROJ_ACTIVE
            for _ in range(10):
                bad = act & (G @ dq < 0)
                if not bad.any():
                    break
                j = torch.argmin(torch.where(bad, G @ dq, torch.zeros_like(c))).item()
                g = G[j]
                dq = dq - g * g.dot(dq) / g.dot(g).clamp(min=1e-9)
            if dq.abs().max() < 1e-5:
                return None  # упёрлись, проекция ничего не оставила
            q = torch.clamp(q + dq, self.lo, self.hi)
            path.append(q.clone())
        return None

    def manip(self, q):
        J = self.jac(q)
        return torch.sqrt(torch.clamp(torch.det(J @ J.transpose(1, 2)), min=0.0))

    # --- подгонка под сим

    def calibrate(self, q0, ee_pos, ee_quat, jac_sim):
        """База DH у UR повёрнута относительно base_link то ли на 0, то ли на 180 вокруг z, хз как в нашем USD.
        Пробую оба, hande_end подгоняю так, чтоб в текущей позе всё сошлось точно, а правильный вариант
        выбираю по якобиану: он от подгонки не зависит, и если база не та, он не совпадёт с симовским."""
        q0 = q0.double().unsqueeze(0)
        T_ee = make_T(ee_pos.double().unsqueeze(0), ee_quat.double().unsqueeze(0))[0]
        best = None
        for flip in (0.0, math.pi):
            self.T_pre = rotz(flip, self.dev)
            self.T_post = torch.linalg.inv(self.T_pre @ dh_frames(q0)[-1][0]) @ T_ee
            diff = (self.jac(q0)[0] - jac_sim.double()).abs().max().item()
            print(f"[planner] база повёрнута на {math.degrees(flip):.0f}: якобиан расходится на {diff:.4f}", flush=True)
            if best is None or diff < best[0]:
                best = (diff, self.T_pre.clone(), self.T_post.clone())
        self.T_pre, self.T_post = best[1], best[2]
        # печатаю, чтоб потом можно было гонять планировщик на cpu без сима
        print(f"[planner] T_post = {[[round(v, 5) for v in row] for row in self.T_post.tolist()]}", flush=True)
        return best[0]

    # --- IK

    def ik(self, pos, quat, seeds, iters=300, lam=0.05):
        """IK сразу для пачки стартов. pos [3], quat [4]. Обычный dls, только на моём fk, а не на руке"""
        q = seeds.double().clone()
        R_goal = matrix_from_quat(quat.double().unsqueeze(0)).expand(q.shape[0], 3, 3)
        p_goal = pos.double().unsqueeze(0)
        eye = torch.eye(6, dtype=torch.float64, device=self.dev)
        for _ in range(iters):
            T = self.fk(q)
            e = torch.cat([p_goal - T[:, :3, 3], rot_err(T[:, :3, :3], R_goal)], dim=1)
            J = self.jac(q)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam**2 * eye, e.unsqueeze(-1))).squeeze(-1)
            # это внутри решателя, рука от этого не дёргается. Режу, чтоб решатель не улетал
            q = torch.clamp(q + dq.clamp(-0.3, 0.3), self.lo, self.hi)
        T = self.fk(q)
        return q, (p_goal - T[:, :3, 3]).norm(dim=1), rot_err(T[:, :3, :3], R_goal).norm(dim=1)

    def solve(self, pos, quats, q_ref, n_seeds=64, gen=None, name="", obj=True, fingers_ok=False):
        """все решения IK для pos и любого из quats. Перебираю их от ближайшего к q_ref (по углам) к дальнему
        и беру первое, где и сама поза, и прямая к ней в суставах ни во что не врезаются.
        Если чистых нет - беру ближайшее с чистой конечной позой и еду к нему костылём-проекцией.
        quats - несколько, потому что пальцы симметричные: кисть, повёрнутая на 180 вокруг своей оси, хватает так же.
        Вернёт (q, путь [M,6]) или None"""
        q_ref = q_ref.double()
        rnd = torch.rand(n_seeds - 1, 6, dtype=torch.float64, device=self.dev, generator=gen)
        lo, hi = self.lo.clamp(min=-math.pi), self.hi.clamp(max=math.pi)
        seeds = torch.cat([q_ref.unsqueeze(0), lo + rnd * (hi - lo)])
        sols = []
        for k, quat in enumerate(quats):
            q, pe, re = self.ik(pos, quat, seeds)
            low = self.lowest(q)
            p_ok, r_ok, z_ok = pe < POS_TOL, re < ROT_TOL, low > MIN_Z
            ok = p_ok & r_ok & z_ok
            # диагностика: кто кого выкинул. Фильтры независимые, так что один старт может попасть в несколько
            pr = p_ok & r_ok
            print(
                f"[planner] {name}, ориентация {k}: из {len(q)} стартов промах >2мм у {(~p_ok).sum().item()}, "
                f"поворот >1° у {(~r_ok).sum().item()}, ниже стола у {(~z_ok).sum().item()}; "
                f"сошлись по позе и повороту {pr.sum().item()}, из них низко {(pr & ~z_ok).sum().item()}; "
                f"лучший промах {pe.min().item() * 1000:.1f} мм, лучший поворот {math.degrees(re.min().item()):.2f}°"
                + (f", самая низкая точка у сошедшихся от {low[pr].min().item():.3f} до {low[pr].max().item():.3f} м" if pr.any() else ""),
                flush=True,
            )
            sols.append(q[ok])
        sols = torch.cat(sols)
        if len(sols) == 0:
            print(f"[planner] {name}: ни одного решения IK", flush=True)
            return None
        dist = (sols - q_ref).norm(dim=1)
        order = dist.argsort()
        sols, dist = sols[order], dist[order]
        # сколько разных решений нашлось - грубо, по округлению до 0.1 рад
        n_diff = len(torch.unique((sols * 10).round(), dim=0))
        end_ok = self.clearance(sols, obj, fingers_ok).min(dim=1).values >= 0
        print(f"[planner] {name}: решений {len(sols)} (разных ~{n_diff}), из них сама поза ни во что не врезается у {end_ok.sum().item()}", flush=True)

        hits = {"стол": 0, "коробка": 0}
        for k in range(len(sols)):
            if not end_ok[k]:
                continue
            # v9: рука едет лесенкой (суставы по одному), значит и проверяю лесенку, а не прямую
            path = staircase(q_ref, sols[k])
            free, what = self.path_free(path, obj, fingers_ok)
            if free:
                print(
                    f"[planner] {name}: беру решение №{k + 1} по близости (до него {dist[k].item():.2f} рад), "
                    f"путь лесенкой чистый; до него лесенка врезалась в стол {hits['стол']} раз, в коробку {hits['коробка']}; "
                    f"manip {self.manip(sols[k].unsqueeze(0)).item():.4f}",
                    flush=True,
                )
                return sols[k], path
            hits[what] += 1

        print(f"[planner] {name}: чистой лесенки нет ни у одного (стол {hits['стол']}, коробка {hits['коробка']}), пробую проекцию", flush=True)
        for k in range(len(sols)):
            if not end_ok[k]:
                continue
            path = self.project_path(q_ref, sols[k], obj, fingers_ok)
            if path is not None:
                print(f"[planner] {name}: проекцией доехал до решения №{k + 1} за {len(path)} шагов", flush=True)
                return sols[k], path
            print(f"[planner] {name}: проекция до решения №{k + 1} застряла", flush=True)
            if k >= 4:
                break  # больше пяти не пробую, долго
        print(f"[planner] {name}: никак", flush=True)
        return None

    def path_check(self, path, name="", obj=True, fingers_ok=False):
        """что будет, если ехать по пути: насколько кисть уйдёт от прямой в декарте, насколько низко опустится
        и сколько останется до препятствий. Просто печатаю, чтоб видеть"""
        qs = densify(path.double(), 30)  # для лесенки режет каждый кусок отдельно
        p = self.fk(qs)[:, :3, 3]
        seg = p[-1] - p[0]
        t = ((p - p[0]) @ seg / seg.dot(seg).clamp(min=1e-9)).clamp(0, 1)
        off = (p - (p[0] + t.unsqueeze(1) * seg)).norm(dim=1).max().item()
        print(
            f"[planner] {name}: точек пути {len(path)}, суставы сдвинутся макс на {(path[-1] - path[0]).abs().max().item():.2f} рад, "
            f"кисть уйдёт от прямой до {off * 1000:.0f} мм, самая низкая точка руки {self.lowest(qs).min().item():.3f} м, "
            f"мин запас до препятствий {self.clearance(qs, obj, fingers_ok).min().item() * 1000:.0f} мм, "
            f"manip мин {self.manip(qs).min().item():.4f}",
            flush=True,
        )
