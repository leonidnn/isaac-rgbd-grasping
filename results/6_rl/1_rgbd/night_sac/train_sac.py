"""SAC: актор смотрит на стол (RGB + высоты 128x128) и выдаёт x, y, yaw (как гауссиану, через tanh в [-1, 1]),
критик учится предсказывать награду для пары картинка + действие. Две головы у критика, берём минимум - как в
обычном SAC. Эпизод из одного шага, поэтому цель критика - просто награда, без целевых сетей и без гаммы.
Температуру альфа подбираю автоматом, как в статье SAC (Haarnoja et al. 2018, вторая версия).

    bash server/run.sh --timeout 10h tmp_rl/train_sac.py <gpu> --name sac --hours 8 [--num-envs 8]

Падать можно: следующий запуск с тем же --name продолжит с бэкапа (~/grasp_task/ckpt/<name>).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="sac")
parser.add_argument("--hours", type=float, default=8.0, help="сколько всего учиться, с учётом прошлых запусков")
parser.add_argument("--stop-after-min", type=float, default=0, help="для проверок: выйти через столько минут этого запуска")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-min", type=float, default=30)
parser.add_argument("--ckpt-min", type=float, default=10)
parser.add_argument("--warmup", type=int, default=1000, help="столько первых попыток - случайно по объекту, как у Q-карты")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

app = make_app()

import torch
import torch.nn as nn
import torch.nn.functional as F

import viz
from fly_env import WS_X, WS_Y, FlyEnv
from train_common import Buffer, Run, random_on_object, to_kij

# --- сети. Свёртки ужимают 128 -> 8, дальше обычный MLP


def encoder():
    return nn.Sequential(
        nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.ReLU(),  # 64
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 32
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
    )


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = encoder()
        self.mu = nn.Linear(256, 3)
        self.log_std = nn.Linear(256, 3)

    def forward(self, img):
        h = self.enc(img.float() / 255.0)
        mu, log_std = self.mu(h), self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # логарифм вероятности с поправкой на tanh, формула из статьи SAC
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.9189385).sum(1) - torch.log(1 - a**2 + 1e-6).sum(1)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = encoder()
        self.q1 = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, img, a):
        h = torch.cat([self.enc(img.float() / 255.0), a], 1)
        return self.q1(h)[:, 0], self.q2(h)[:, 0]


# действие [-1,1]^3 <-> x, y, yaw
def to_xyyaw(a):
    x = (WS_X[0] + WS_X[1]) / 2 + a[:, 0] * (WS_X[1] - WS_X[0]) / 2
    y = (WS_Y[0] + WS_Y[1]) / 2 + a[:, 1] * (WS_Y[1] - WS_Y[0]) / 2
    yaw = (a[:, 2] + 1) / 2 * torch.pi
    return x, y, yaw


def to_a(x, y, yaw):
    ax = (x - (WS_X[0] + WS_X[1]) / 2) / ((WS_X[1] - WS_X[0]) / 2)
    ay = (y - (WS_Y[0] + WS_Y[1]) / 2) / ((WS_Y[1] - WS_Y[0]) / 2)
    ayaw = (yaw % torch.pi) / torch.pi * 2 - 1
    return torch.stack([ax, ay, ayaw], 1).clamp(-0.999, 0.999)


def main():
    torch.manual_seed(args.seed)
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=True)
    dev = env.dev
    actor, critic = Actor().to(dev), Critic().to(dev)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -3.0
    buf = Buffer()
    run = Run(args.name)
    attempt = 0
    st = run.load(buf)
    if st is not None:
        actor.load_state_dict(st["actor"])
        critic.load_state_dict(st["critic"])
        opt_a.load_state_dict(st["opt_a"])
        opt_c.load_state_dict(st["opt_c"])
        opt_al.load_state_dict(st["opt_al"])
        with torch.no_grad():
            log_alpha.copy_(st["log_alpha"])
        attempt = st["attempt"]
    t_this = time.time()
    first_eval = True

    def state():
        return {
            "actor": actor.state_dict(), "critic": critic.state_dict(), "opt_a": opt_a.state_dict(),
            "opt_c": opt_c.state_dict(), "opt_al": opt_al.state_dict(), "log_alpha": log_alpha.detach().cpu(), "attempt": attempt,
        }

    def evaluate():
        """2 раунда жадно (середина гауссианы). Первый раунд с видео первого стола и картинкой, куда целится"""
        stats = {}
        for rnd in range(2):
            obs = env.reset()
            with torch.no_grad():
                _, _, a = actor(obs["img"].to(dev))
            x, y, yaw = to_xyyaw(a)
            if rnd == 0:
                tag = f"{attempt:06d}"
                titles = [f"{obs['obj'][k]}" for k in range(4)]
                viz.sees_png(obs["img"][:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(run.viz, f"sac_{tag}.png"), titles)
                env.rec = []
            r, _ = env.grasp(x, y, yaw)
            if env.rec:
                viz.save_gif(env.rec, os.path.join(run.viz, f"video_{tag}_{obs['obj'][0]}_{'ok' if r[0] > 0 else 'fail'}.gif"))
            env.rec = None
            for k in range(env.N):
                s = stats.setdefault(obs["obj"][k], [0, 0])
                s[0] += int(r[k].item())
                s[1] += 1
        run.log_eval(attempt, stats)
        print(f"[eval] попытка {attempt}: " + ", ".join(f"{o} {a}/{b}" for o, (a, b) in stats.items()), flush=True)
        if os.path.exists(run.train_csv):
            viz.curves_png(run.train_csv, run.eval_csv, os.path.join(run.viz, "curves.png"), "SAC")

    while run.hours() < args.hours:
        if args.stop_after_min and time.time() - t_this > args.stop_after_min * 60:
            print("stop-after-min: выхожу (проверка resume)", flush=True)
            break
        if first_eval and st is None or time.time() - run.last_eval > args.eval_min * 60:
            evaluate()
            run.last_eval = time.time()
        first_eval = False

        # --- раунд попыток
        obs = env.reset()
        img = obs["img"].to(dev)
        if attempt < args.warmup:
            x, y, yaw = random_on_object(img)
        else:
            with torch.no_grad():
                a, _, _ = actor(img)
            x, y, yaw = to_xyyaw(a)
        r, info = env.grasp(x, y, yaw)
        act = torch.stack([x, y, yaw], dim=1)
        buf.add(img, act, to_kij(x, y, yaw), r)
        run.log_train(attempt, obs["obj"], act.cpu(), r.cpu(), 1.0 if attempt < args.warmup else 0.0)
        attempt += env.N

        # --- учим, по батчу на попытку
        lc, la = [], []
        for _ in range(env.N):
            b_img, b_act, _, b_r = buf.sample(64, dev)
            q1, q2 = critic(b_img, to_a(b_act[:, 0], b_act[:, 1], b_act[:, 2]))
            loss_c = F.mse_loss(q1, b_r) + F.mse_loss(q2, b_r)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            a, logp, _ = actor(b_img)
            q1, q2 = critic(b_img, a)
            alpha = log_alpha.exp().detach()
            loss_a = (alpha * logp - torch.min(q1, q2)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

            loss_al = -(log_alpha * (logp.detach() + target_entropy)).mean()
            opt_al.zero_grad()
            loss_al.backward()
            opt_al.step()
            lc.append(loss_c.item())
            la.append(loss_a.item())
        print(
            f"попытка {attempt:6d}  {run.hours():.2f} ч  {'warmup ' if attempt <= args.warmup else ''}взял {int(r.sum())}/{env.N}  "
            f"недостижимо {int((~info['reach']).sum())}  loss_c {sum(lc) / len(lc):.3f}  loss_a {sum(la) / len(la):.3f}  "
            f"alpha {log_alpha.exp().item():.3f}",
            flush=True,
        )

        if time.time() - run.last_ckpt > args.ckpt_min * 60:
            run.save(state(), buf)
            run.last_ckpt = time.time()

    run.save(state(), buf)
    if run.hours() >= args.hours:
        evaluate()
        open(os.path.join(run.dir, "DONE"), "w").close()
        print("DONE: время вышло, обучение закончено", flush=True)


if __name__ == "__main__":
    main()
    app.close()
