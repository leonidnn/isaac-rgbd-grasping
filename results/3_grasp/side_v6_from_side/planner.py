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


class Planner:
    def __init__(self, lo, hi, dev):
        self.dev = dev
        self.lo, self.hi = lo.double(), hi.double()  # лимиты суставов руки [6]
        self.T_pre = torch.eye(4, dtype=torch.float64, device=dev)  # мир -> база DH
        self.T_post = torch.eye(4, dtype=torch.float64, device=dev)  # 6-е звено DH -> hande_end

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

    def solve(self, pos, quats, q_ref, n_seeds=64, gen=None, name=""):
        """все решения IK для pos и любого из quats, выбираем ближайшее к q_ref по углам.
        quats - несколько, потому что пальцы симметричные: кисть, повёрнутая на 180 вокруг своей оси, хватает так же"""
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
        best = sols[dist.argmin()]
        # сколько разных решений нашлось - грубо, по округлению до 0.1 рад
        n_diff = len(torch.unique((sols * 10).round(), dim=0))
        print(
            f"[planner] {name}: решений {len(sols)} (разных ~{n_diff}), беру ближайшее, "
            f"до него {dist.min().item():.2f} рад, manip {self.manip(best.unsqueeze(0)).item():.4f}",
            flush=True,
        )
        return best

    def path_check(self, q_from, q_to, name="", n=30):
        """что будет с кисткой, если ехать по прямой в углах: насколько она уйдёт от прямой в декарте
        и насколько низко опустится. Просто печатаю, чтоб видеть"""
        a = torch.linspace(0, 1, n, dtype=torch.float64, device=self.dev).unsqueeze(1)
        qs = q_from.double() + a * (q_to.double() - q_from.double())
        p = self.fk(qs)[:, :3, 3]
        seg = p[-1] - p[0]
        t = ((p - p[0]) @ seg / seg.dot(seg).clamp(min=1e-9)).clamp(0, 1)
        off = (p - (p[0] + t.unsqueeze(1) * seg)).norm(dim=1).max().item()
        print(
            f"[planner] {name}: суставы сдвинутся макс на {(q_to - q_from).abs().max().item():.2f} рад, "
            f"кисть уйдёт от прямой до {off * 1000:.0f} мм, самая низкая точка руки {self.lowest(qs).min().item():.3f} м, "
            f"manip мин {self.manip(qs).min().item():.4f}",
            flush=True,
        )
