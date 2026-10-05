"""Все картинки в одном месте: gif, что видит агент, тепловая карта Q, графики обучения.
Рисую matplotlib-ом без окна (Agg), сохраняю png.
"""

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OBJ_COLOR = {"tetrapak": "tab:blue", "can": "tab:orange", "chips": "tab:green"}


def save_gif(frames, path, ms=100):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    print(f"gif {path}, {len(imgs)} кадров", flush=True)


def _rgb(img):
    return img[:3].permute(1, 2, 0).cpu().numpy()


def _height(img):
    return img[3].cpu().numpy() / 255 * 0.25 * 100  # см


def _mark(ax, x, y, yaw, color="red"):
    # точка в пикселях агента + палка вдоль сжатия пальцев
    from fly_env import FlyEnv

    import torch

    i, j = FlyEnv.xy_to_pix(torch.as_tensor(float(x)), torch.as_tensor(float(y)))
    i, j = int(i), int(j)
    ax.plot(j, i, "+", color=color, ms=12, mew=2)
    d = 12
    yaw = float(yaw)
    ax.plot([j - d * math.cos(yaw), j + d * math.cos(yaw)], [i + d * math.sin(yaw), i - d * math.sin(yaw)], color=color, lw=2)


def sees_png(imgs, marks, path, titles=None):
    """imgs [K,4,H,W] uint8: верхний ряд RGB, нижний высоты. marks - (x, y, yaw) или None"""
    K = len(imgs)
    fig, axs = plt.subplots(2, K, figsize=(3 * K, 6), squeeze=False)
    for k in range(K):
        axs[0, k].imshow(_rgb(imgs[k]))
        im = axs[1, k].imshow(_height(imgs[k]), cmap="viridis", vmin=0, vmax=20)
        if marks and marks[k] is not None:
            _mark(axs[0, k], *marks[k])
            _mark(axs[1, k], *marks[k])
        if titles:
            axs[0, k].set_title(titles[k], fontsize=9)
        for a in axs[:, k]:
            a.axis("off")
    fig.colorbar(im, ax=axs[1, :].tolist(), shrink=0.6, label="высота, см")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def qmap_png(imgs, probs, picks, path, titles=None):
    """тепловая карта Q-карты: imgs [K,4,H,W], probs [K,8,H,W] - вероятности успеха по yaw.
    Сверху картинка с выбранной точкой, снизу максимум по yaw (где вообще можно брать)"""
    K = len(imgs)
    fig, axs = plt.subplots(2, K, figsize=(3 * K, 6), squeeze=False)
    for k in range(K):
        axs[0, k].imshow(_rgb(imgs[k]))
        if picks[k] is not None:
            _mark(axs[0, k], *picks[k])
        p = probs[k].max(dim=0).values.cpu().numpy()
        axs[1, k].imshow(_rgb(imgs[k]))
        im = axs[1, k].imshow(p, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        if titles:
            axs[0, k].set_title(titles[k], fontsize=9)
        for a in axs[:, k]:
            a.axis("off")
    fig.colorbar(im, ax=axs[1, :].tolist(), shrink=0.6, label="P(успех), макс по yaw")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def curves_png(log_path, eval_path, path, title, window=300):
    """графики обучения: скользящий SR по объектам на обучающих попытках + SR жадной политики на проверках"""
    rows = read_log(log_path)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for obj, c in OBJ_COLOR.items():
        rr = [(int(r["attempt"]), float(r["reward"])) for r in rows if r["obj"] == obj]
        if len(rr) < 5:
            continue
        a = np.array(rr)
        w = min(window // 3, len(a))
        sm = np.convolve(a[:, 1], np.ones(w) / w, mode="valid")
        axs[0].plot(a[w - 1:, 0], sm, color=c, label=obj)
    axs[0].set_title(f"{title}: SR на обучении (скользящее по ~{window // 3} попыткам объекта)")
    axs[0].set_xlabel("попыток всего")
    axs[0].set_ylim(-0.02, 1.02)
    axs[0].grid(alpha=0.3)
    axs[0].legend()
    try:
        ev = read_log(eval_path)
    except FileNotFoundError:
        ev = []
    for obj, c in OBJ_COLOR.items():
        e = [r for r in ev if r["obj"] == obj]
        if not e:
            continue
        xs = [int(r["attempt"]) for r in e]
        k = [int(r["ok"]) for r in e]
        n = [int(r["n"]) for r in e]
        sr = [a / b for a, b in zip(k, n)]
        lo = [a - wilson(x, y)[0] for a, x, y in zip(sr, k, n)]
        hi = [wilson(x, y)[1] - a for a, x, y in zip(sr, k, n)]
        axs[1].errorbar(xs, sr, yerr=[lo, hi], color=c, marker="o", capsize=3, label=obj)
    axs[1].set_title(f"{title}: SR жадной политики на проверках (95% Уилсон)")
    axs[1].set_xlabel("попыток всего")
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
