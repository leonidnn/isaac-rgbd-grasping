"""Одна обучалка на все варианты дообучения на чипсах, отличаются флагами. Основа - 6_rl/2_depth/train_qmap.py
(Q-карта, гамма 0, BCE в одной клетке, 4xN батчей по 8, lr 1e-4, буфер 40 тыс.).

Мерка для всех (plan.md, 12.10): ровно --budget попыток (6400, ~4 ч вместе с проверками), проверка каждые 800 попыток,
главный порог 80 % на чипсах. eps 0.5 -> 0.1 за первые 4000 попыток, дальше держу (по попыткам, а не по часам,
иначе за 4 ч он бы не успел опуститься). На старых объектах действует та же текущая сеть.

    1 роутер:        --name chips_router --init none --old-share 0   --eval-objs chips
    3 naive FT:      --name chips_ft     --init old  --old-share 0
    3.2 FT+replay:   --name chips_rp20   --init old  --old-share 0.2
    2 task emb:      --name chips_emb20  --init old  --old-share 0.2 --film
    5 joint:         --name chips_joint  --init none --old-share 0.667   (все три объекта поровну, с нуля)
    4б адаптеры:     --name chips_adapt  --init old  --old-share 0   --adapter   (старое заморожено, без переключателя)

old-share - доля столов со старым объектом (тетрапак или банка поровну), считается в общем бюджете попыток.

    bash server/run.sh --timeout 6h results/7_chips/common/train.py <gpu> --name chips_ft --init old

Падать можно: следующий запуск с тем же --name продолжит со своего бэкапа (~/grasp_task/ckpt/<name>).
В конце DONE в папке бэкапа - по нему очередь (wait_run.sh --until-done) понимает, что перезапускать не надо.
"""

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "grasp")):
    ROOT = os.path.dirname(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--init", choices=["none", "old"], required=True, help="none - случайные веса, old - agent/qmap_depth.pt")
parser.add_argument("--old-share", type=float, default=0.0, help="доля столов с тетрапаком или банкой, остальные - чипсы")
parser.add_argument("--film", action="store_true", help="вариант 2: код объекта через FiLM")
parser.add_argument("--adapter", action="store_true", help="вариант 4б: старая сеть заморожена, учатся только адаптеры")
parser.add_argument("--eval-objs", nargs="*", default=["chips", "tetrapak", "can"])
parser.add_argument("--budget", type=int, default=6400, help="ровно столько попыток учу")
parser.add_argument("--eval-every", type=int, default=800)
parser.add_argument("--eval-n", type=int, default=64, help="попыток на чипсы в проверке (на старые - вполовину)")
parser.add_argument("--eps-start", type=float, default=0.5)
parser.add_argument("--eps-end", type=float, default=0.1)
parser.add_argument("--eps-attempts", type=int, default=4000)
parser.add_argument("--stop-after-min", type=float, default=0, help="для проверок: выйти через столько минут этого запуска")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video-every", type=int, default=2, help="видео на каждой N-й проверке (видео дорогое)")
parser.add_argument("--ckpt-min", type=float, default=10)
parser.add_argument("--warmup", type=int, default=10, help="рендеров перед снимком, меньше 8 - кадр от прошлой сцены")
args = parser.parse_args()
assert not (args.adapter and (args.film or args.init != "old")), "адаптеры только поверх старых весов и без film"
assert args.eval_every % args.num_envs == 0 and args.budget % args.eval_every == 0, "eval-every кратно столам, budget кратно eval-every"

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

app = make_app()

import torch
import torch.nn.functional as F

import viz
from fly_env import FlyEnv
from qmap import QMap, QMapAdapter, QMapFilm, codes, greedy, load_old
from train_common import Buffer, Run, evaluated, random_on_object, to_kij

OLD = ["tetrapak", "can"]


def eps_at(attempt):
    a = min(1.0, attempt / args.eps_attempts)
    return args.eps_start + (args.eps_end - args.eps_start) * a


def eval_tables():
    """раскладка объектов по столам для проверки: на чипсы eval-n попыток, на каждый старый вполовину.
    Объекты по столам перемешаны, а не 'раунд чипсов, раунд банки' - так раундов меньше"""
    objs = []
    for o in args.eval_objs:
        objs += [o] * (args.eval_n if o == "chips" else args.eval_n // 2)
    objs += ["chips"] * (-len(objs) % args.num_envs)  # добить до полного раунда
    return [objs[i : i + args.num_envs] for i in range(0, len(objs), args.num_envs)]


def main():
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=True, warmup=args.warmup)
    dev = env.dev
    net = (QMapFilm() if args.film else QMapAdapter() if args.adapter else QMap()).to(dev)
    if args.init == "old":
        load_old(net, os.path.join(ROOT, "agent", "qmap_depth.pt"))
    # свежий Adam и при init old (решение 12.10). У адаптеров в Adam только они сами, остальное заморожено
    opt = torch.optim.Adam(net.freeze_base() if args.adapter else net.parameters(), lr=1e-4)
    buf = Buffer()
    run = Run(args.name)
    attempt = 0
    st = run.load(buf)
    if st is not None:
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        attempt = st["attempt"]
    else:
        print(f"[init] веса {'старые' if args.init == 'old' else 'с нуля'}, film {args.film}, adapter {args.adapter}, "
              f"учится {sum(p.numel() for p in net.parameters() if p.requires_grad)} весов, буфер пустой", flush=True)

    def pick(img, objs):
        if args.film:
            net.set_code(codes(objs, dev))
        return greedy(net, img)

    n_eval = [0]

    def evaluate():
        """жадно, без случайности. Картинка карт в первом раунде, видео первого стола - на каждой video-every-й"""
        stats = {}
        video = n_eval[0] % args.video_every == 0
        n_eval[0] += 1
        for rnd, objs in enumerate(eval_tables()):
            obs = env.reset(objs)
            x, y, yaw, prob = pick(obs["img"].to(dev), objs)
            if rnd == 0:
                tag = f"{attempt:06d}"
                viz.qmap_png(obs["img"][:4], prob[:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(run.viz, f"qmap_{tag}.png"), objs[:4])
                env.rec = [] if video else None
            r, _ = env.grasp(x, y, yaw)
            if env.rec:
                viz.save_gif(env.rec, os.path.join(run.viz, f"video_{tag}_{objs[0]}_{'ok' if r[0] > 0 else 'fail'}.gif"))
            env.rec = None
            for k in range(env.N):
                s = stats.setdefault(objs[k], [0, 0])
                s[0] += int(r[k].item())
                s[1] += 1
        run.log_eval(attempt, stats)
        print(f"[eval] попытка {attempt}, {run.hours():.2f} ч: " + ", ".join(f"{o} {a}/{b}" for o, (a, b) in stats.items()), flush=True)
        if os.path.exists(run.train_csv):
            viz.curves_png(run.train_csv, run.eval_csv, os.path.join(run.viz, "curves.png"), args.name)

    t_this = time.time()
    while True:
        # проверка: на попытке 0 (до обучения), каждые eval-every и на бюджете
        if attempt % args.eval_every == 0 and not evaluated(run.eval_csv, attempt):
            evaluate()
        if attempt >= args.budget:
            break
        if args.stop_after_min and time.time() - t_this > args.stop_after_min * 60:
            print("stop-after-min: выхожу (проверка resume)", flush=True)
            break

        # --- раунд попыток: на столе старый объект с вероятностью old-share, иначе чипсы
        objs = [rng.choice(OLD) if rng.random() < args.old_share else "chips" for _ in range(env.N)]
        obs = env.reset(objs)
        img = obs["img"].to(dev)
        x, y, yaw, _ = pick(img, objs)
        eps = eps_at(attempt)
        rnd = torch.rand(env.N, device=dev) < eps
        rx, ry, ryaw = random_on_object(img)
        x, y, yaw = torch.where(rnd, rx, x), torch.where(rnd, ry, y), torch.where(rnd, ryaw, yaw)
        true = torch.stack(env.oracle(), dim=1)  # только в лог
        r, info = env.grasp(x, y, yaw)
        act = torch.stack([x, y, yaw], dim=1)
        buf.add(img, act, to_kij(x, y, yaw), r, codes(objs, "cpu"))
        run.log_train(attempt, objs, act.cpu(), r.cpu(), eps, true.cpu())
        attempt += env.N

        # --- учим: на каждую новую попытку 4 батча по 8, батч из общего буфера равномерно
        losses = []
        for _ in range(4 * env.N):
            b_img, _, b_kij, b_r, b_code = buf.sample(8, dev)
            if args.film:
                net.set_code(b_code)
            logits = net(b_img)
            pred = logits[torch.arange(len(b_r), device=dev), b_kij[:, 0], b_kij[:, 1], b_kij[:, 2]]
            loss = F.binary_cross_entropy_with_logits(pred, b_r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(
            f"попытка {attempt:6d}  {run.hours():.2f} ч  eps {eps:.2f}  взял {int(r.sum())}/{env.N}  "
            f"старых {sum(o != 'chips' for o in objs)}  недостижимо {int((~info['reach']).sum())}  loss {sum(losses) / len(losses):.3f}",
            flush=True,
        )

        if time.time() - run.last_ckpt > args.ckpt_min * 60:
            run.save({"net": net.state_dict(), "opt": opt.state_dict(), "attempt": attempt}, buf)
            run.last_ckpt = time.time()

    run.save({"net": net.state_dict(), "opt": opt.state_dict(), "attempt": attempt}, buf)
    if attempt >= args.budget:
        open(os.path.join(run.dir, "DONE"), "w").close()
        print(f"DONE: {attempt} попыток, {run.hours():.2f} ч", flush=True)


if __name__ == "__main__":
    main()
    app.close()
