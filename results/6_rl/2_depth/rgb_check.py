"""Почему RGB у агента от прошлой сцены и бледный (ночь 05.10). Раскладываю объекты на 4 стола несколько раз подряд
и после каждой раскладки снимаю после 2, 4, 8, 16, 32 рендеров. Снизу карта высот - она правильная, с ней и сравниваю.
Смотрю глазами, после скольких рендеров цвет совпадает с высотами и перестаёт быть бледным.

    bash server/run.sh tmp_rl2/rgb_check.py <gpu> [--rounds 3]

Кладёт rgb_check_round<N>.png и печатает яркость кадра (если всё белое - среднее под 230+).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=3)
args = parser.parse_args()

sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, os.path.join(ROOT, "grasp"))
sys.path.insert(0, HERE)
from isaac_app import make_app

t0 = time.time()
app = make_app()
print(f"app started in {time.time() - t0:.1f}s", flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT
from fly_env import FlyEnv

STEPS = [2, 4, 8, 16, 32]


def main():
    env = FlyEnv(num_envs=4, seed=1, warmup=0)
    for rnd in range(args.rounds):
        env.reset()  # warmup=0: объекты разложены, но не рендерили
        shots, done = [], 0
        for n in STEPS:
            for _ in range(n - done):
                env.sim.render()
            done = n
            img = env.snap()["img"]
            shots.append(img)
            rgb = img[:, :3].float()
            print(f"round {rnd} после {n:2d} рендеров: яркость RGB средняя {rgb.mean():.0f}, разброс {rgb.std():.1f}", flush=True)

        fig, axs = plt.subplots(len(STEPS) + 1, 4, figsize=(12, 3 * (len(STEPS) + 1)), squeeze=False)
        for k in range(4):
            for r, n in enumerate(STEPS):
                axs[r, k].imshow(shots[r][k, :3].permute(1, 2, 0).cpu().numpy())
                axs[r, k].set_title(f"{env.obj[k]}, {n} рендеров", fontsize=9)
            axs[-1, k].imshow(shots[-1][k, 3].cpu().numpy(), cmap="gray")
            axs[-1, k].set_title("высоты (правильно)", fontsize=9)
            for a in axs[:, k]:
                a.axis("off")
        fig.tight_layout()
        path = os.path.join(OUT, f"rgb_check_round{rnd}.png")
        fig.savefig(path, dpi=70)
        plt.close(fig)
        print(f"картинка {path}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
