"""Итоговая оценка любого варианта в быстрой среде после обучения (common/train.py): последние веса = веса на бюджете.
Жадно, по 64 попытки на каждый объект (24 раунда на 8 столах, объекты по столам со сдвигом на раунд), SR с Уилсоном.

    --router  вариант 1: чипсы берёт новая сеть, тетрапак и банку - старая agent/qmap_depth.pt (по известному объекту)
    --film    вариант 2: сеть с кодом объекта
    --adapter вариант 4б: старая сеть + адаптеры

    bash server/run.sh results/7_chips/common/eval_final.py <gpu> --name chips_ft [--router] [--film] [--video]

Кладёт в папку запуска: sr.csv, qmap_round0..2.png (все три объекта), gif первого стола первых трёх раундов.
"""

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "grasp")):
    ROOT = os.path.dirname(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="имя бэкапа обучения в ~/grasp_task/ckpt")
parser.add_argument("--router", action="store_true")
parser.add_argument("--film", action="store_true")
parser.add_argument("--adapter", action="store_true")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--rounds", type=int, default=24, help="24 раунда на 8 столах = по 64 попытки на объект")
parser.add_argument("--video", action="store_true")
parser.add_argument("--seed", type=int, default=1000, help="не 0, как при обучении, чтоб сцены были другие")
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

app = make_app()

import torch

from common import OUT
from fly_env import FlyEnv
from qmap import QMap, QMapAdapter, QMapFilm, codes, greedy, load_old
from train_common import CKPT_ROOT
from viz import qmap_png, save_gif

sys.path.insert(0, os.path.join(ROOT, "agent"))
from agent import wilson

OBJS = ["chips", "tetrapak", "can"]


def trained():
    path = os.path.join(CKPT_ROOT, args.name, "last.pt")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=os.environ["GT_HF_REPO"], filename=f"rl/{args.name}/last.pt")
        print("на диске бэкапа нет, взял с HF", flush=True)
    st = torch.load(path, map_location="cpu")
    print(f"сеть {args.name}: попыток {st['attempt']}, училась {st['hours']:.2f} ч", flush=True)
    return st


def main():
    env = FlyEnv(num_envs=args.num_envs, seed=args.seed, video=args.video)
    dev = env.dev
    st = trained()
    net = (QMapFilm() if args.film else QMapAdapter() if args.adapter else QMap()).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    old = load_old(QMap(), os.path.join(ROOT, "agent", "qmap_depth.pt")).to(dev).eval() if args.router else None
    stats = {o: [0, 0] for o in OBJS}
    for rnd in range(args.rounds):
        t = time.time()
        objs = [OBJS[(k + rnd) % len(OBJS)] for k in range(env.N)]
        obs = env.reset(objs)
        img = obs["img"].to(dev)
        if args.film:
            net.set_code(codes(objs, dev))
        x, y, yaw, prob = greedy(net, img)
        if old is not None:
            # роутер: старая сеть считает на всех столах, беру её ответ там, где не чипсы
            xo, yo, yawo, po = greedy(old, img)
            mine = torch.tensor([o == "chips" for o in objs], device=dev)
            x, y, yaw = torch.where(mine, x, xo), torch.where(mine, y, yo), torch.where(mine, yaw, yawo)
            prob = torch.where(mine[:, None, None, None], prob, po)
        if rnd < 3:
            qmap_png(obs["img"][:4], prob[:4], [(x[k], y[k], yaw[k]) for k in range(4)], os.path.join(OUT, f"qmap_round{rnd}.png"), objs[:4])
        env.rec = [] if (args.video and rnd < 3) else None
        r, _ = env.grasp(x, y, yaw)
        if env.rec:
            save_gif(env.rec, os.path.join(OUT, f"final_round{rnd}_{objs[0]}_{'ok' if r[0] > 0 else 'fail'}.gif"))
        env.rec = None
        for k in range(env.N):
            stats[objs[k]][0] += int(r[k].item())
            stats[objs[k]][1] += 1
        print(f"round {rnd}: {int(r.sum())}/{env.N} взял, {time.time() - t:.1f} с  " + " ".join(f"{objs[k]}:{'ok' if r[k] > 0 else '-'}" for k in range(env.N)), flush=True)

    print(f"SR {args.name}:", flush=True)
    with open(os.path.join(OUT, "sr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "obj", "ok", "n", "sr", "ci_lo", "ci_hi", "attempts", "hours"])
        for o, (a, b) in stats.items():
            lo, hi = wilson(a, b)
            w.writerow([args.name, o, a, b, f"{a / b:.3f}", f"{lo:.3f}", f"{hi:.3f}", st["attempt"], f"{st['hours']:.2f}"])
            print(f"    {o:8s} {a}/{b} = {a / b:.0%}  95% CI [{lo:.0%}, {hi:.0%}]", flush=True)


if __name__ == "__main__":
    main()
    app.close()
