"""Картинки для отчёта по уже собранным логам, считаю у себя на компе (py -3 analysis.py), сервер не нужен.

1. tolerance_map.png - насколько точно надо хватать: успех в зависимости от того, куда ткнули относительно центра
   объекта (в осях объекта) и насколько ошиблись с углом. Данные - попытка 2 (только там пишется настоящая поза)
2. summary_table.png + summary_table.md - оракул / baseline / RL, SR по объектам с 95% интервалом Уилсона
3. rgb_bug.png - попытка 1 (битый RGB-D) против попытки 2 (только глубина): SR на обучении по часам

Цифры оракула и baseline переписал руками из run.log (пути в таблице), попытки RL читаю из csv.
"""

import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
OBJS = ["tetrapak", "can", "chips"]
RU = {"tetrapak": "тетрапак", "can": "банка", "chips": "чипсы"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------- 1. карта допуска


def tolerance_map():
    rows = [r for r in read(os.path.join(HERE, "data", "qmap_d_train.csv")) if r["obj"] in ("tetrapak", "can")]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.35})
    lim, step = 0.06, 0.01
    edges = np.arange(-lim, lim + 1e-9, step)
    for a, obj in zip(axs[:2], ("can", "tetrapak")):
        rr = [r for r in rows if r["obj"] == obj]
        dx = np.array([float(r["x"]) - float(r["obj_x"]) for r in rr])
        dy = np.array([float(r["y"]) - float(r["obj_y"]) for r in rr])
        oy = np.array([float(r["obj_yaw"]) for r in rr])
        ok = np.array([int(r["reward"]) for r in rr])
        # в оси объекта: u - вдоль его оси x, v - поперёк
        u = dx * np.cos(oy) + dy * np.sin(oy)
        v = -dx * np.sin(oy) + dy * np.cos(oy)
        n, _, _ = np.histogram2d(u, v, bins=[edges, edges])
        s, _, _ = np.histogram2d(u, v, bins=[edges, edges], weights=ok)
        sr = np.where(n >= 5, s / np.maximum(n, 1), np.nan)  # меньше 5 попыток в клетке - не рисую
        im = a.imshow(sr.T, origin="lower", extent=[-lim * 100, lim * 100, -lim * 100, lim * 100], cmap="RdYlGn", vmin=0, vmax=1)
        a.set_title(f"{RU[obj]}: успех по месту схвата\n{len(rr)} попыток, клетка 1 см, ≥5 попыток", fontsize=10)
        a.set_xlabel("смещение вдоль оси x объекта, см")
        a.set_ylabel("поперёк, см")
        a.plot(0, 0, "k+", ms=12)
    fig.colorbar(im, ax=axs[:2].tolist(), shrink=0.8, label="доля успехов")

    # угол для тетрапака: на сколько повёрнуты пальцы относительно оси x объекта (0 и 180 - одно и то же)
    a = axs[2]
    rr = [r for r in rows if r["obj"] == "tetrapak"]
    dist = np.array([math.hypot(float(r["x"]) - float(r["obj_x"]), float(r["y"]) - float(r["obj_y"])) for r in rr])
    d = np.array([(float(r["yaw"]) - float(r["obj_yaw"])) for r in rr])
    d = np.degrees(d) % 180
    d = np.where(d > 90, 180 - d, d)
    ok = np.array([int(r["reward"]) for r in rr])
    for mask, lab, c in ((dist < 0.02, "точка ближе 2 см к центру", "tab:green"), (dist >= 0.02, "дальше 2 см", "tab:gray")):
        xs, ys, lo, hi = [], [], [], []
        for b in range(0, 90, 15):
            m = mask & (d >= b) & (d < b + 15)
            k, n = int(ok[m].sum()), int(m.sum())
            if n < 5:
                continue
            p = k / n
            w = wilson(k, n)
            xs.append(b + 7.5), ys.append(p), lo.append(p - w[0]), hi.append(w[1] - p)
        a.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color=c, label=lab)
    a.set_title("тетрапак: успех от угла пальцев\n(0° - пальцы сходятся вдоль оси x объекта)")
    a.set_xlabel("ошибка угла, градусы")
    a.set_ylabel("доля успехов (95% Уилсон)")
    a.set_ylim(-0.02, 1.02)
    a.grid(alpha=0.3)
    a.legend()
    fig.savefig(os.path.join(HERE, "tolerance_map.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    # и цифры по расстоянию - в лог
    lines = []
    for obj in ("can", "tetrapak"):
        rr = [r for r in rows if r["obj"] == obj]
        dd = np.array([math.hypot(float(r["x"]) - float(r["obj_x"]), float(r["y"]) - float(r["obj_y"])) for r in rr]) * 100
        ok = np.array([int(r["reward"]) for r in rr])
        for b in range(0, 6):
            m = (dd >= b) & (dd < b + 1)
            if m.sum():
                lo, hi = wilson(int(ok[m].sum()), int(m.sum()))
                lines.append(f"{RU[obj]:9s} {b}-{b + 1} см: {int(ok[m].sum())}/{int(m.sum())} = {ok[m].mean():.0%} [{lo:.0%}, {hi:.0%}]")
    return lines


# ---------------- 2. общая таблица


def evals_from_csv(path):
    k, n = {o: 0 for o in OBJS}, {o: 0 for o in OBJS}
    for r in read(path):
        k[r["obj"]] += int(r["ok"])
        n[r["obj"]] += int(r["n"])
    return {o: (k[o], n[o]) for o in OBJS}


def evals_from_log(path, last=None):
    """строки [eval] из лога; last=N - только последние N проверок"""
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.startswith("[eval]")]
    if last:
        lines = lines[-last:]
    k, n = {o: 0 for o in OBJS}, {o: 0 for o in OBJS}
    for line in lines:
        for o, a, b in re.findall(r"(tetrapak|can|chips) (\d+)/(\d+)", line):
            k[o] += int(a)
            n[o] += int(b)
    return {o: (k[o], n[o]) for o in OBJS}, len(lines)


def train_from_csv(path, min_hours):
    """успехи на обучении начиная с min_hours часов"""
    k, n = {o: 0 for o in OBJS}, {o: 0 for o in OBJS}
    for r in read(path):
        if float(r["hours"]) >= min_hours:
            k[r["obj"]] += int(r["reward"])
            n[r["obj"]] += 1
    return {o: (k[o], n[o]) for o in OBJS}


def arm_from_json(path):
    import json

    with open(path, encoding="utf-8") as f:
        eps = json.load(f)
    k, n = {o: 0 for o in OBJS}, {o: 0 for o in OBJS}
    for e in eps:
        k[e["obj"]] += int(bool(e["success"]))
        n[e["obj"]] += 1
    return {o: ((k[o], n[o]) if n[o] else None) for o in OBJS}


def summary_table():
    ev = os.path.join(HERE, "data", "qmap_d_evals.txt")
    q2_last, n_last = evals_from_log(ev, last=6)
    rows = [
        ("Оракул, рука + IK (4_env/oracle_yaw0)", {"tetrapak": (10, 10), "can": (10, 10), "chips": (0, 10)}),
        ("Оракул, быстрая среда (fly_check 4+8 столов)", {"tetrapak": (9, 10), "can": (8, 8), "chips": (0, 10)}),
        ("Baseline: глубина + планировщик (5_baseline/plan_11x2)", {"tetrapak": (11, 11), "can": (11, 11), "chips": None}),
        ("Q-карта, попытка 1, RGB-D (жадные проверки за ночь)", evals_from_csv(os.path.join(RL, "1_rgbd", "night_qmap", "eval.csv"))),
        ("SAC, попытка 1, RGB-D (жадные проверки за ночь)", evals_from_csv(os.path.join(RL, "1_rgbd", "night_sac", "eval.csv"))),
        ("Q-карта, попытка 2, обучение: последние 4 ч (eps 0.1)", train_from_csv(os.path.join(HERE, "data", "qmap_d_train.csv"), 8.0)),
        (f"Q-карта, попытка 2, инференс: жадно, последние {n_last} проверок (после 7 ч)", q2_last),
        ("Q-карта, попытка 2, инференс на настоящей руке (3_arm/eval_20x2)", arm_from_json(os.path.join(RL, "3_arm", "eval_20x2", "episodes.json"))),
    ]
    md = ["| метод | тетрапак | банка | чипсы |", "|---|---|---|---|"]
    cells = []
    for name, d in rows:
        line, cl = [name], []
        for o in OBJS:
            if d.get(o) is None or d[o][1] == 0:
                line.append("—")
                cl.append("—")
                continue
            k, n = d[o]
            lo, hi = wilson(k, n)
            s = f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]"
            line.append(s)
            cl.append(s)
        md.append("| " + " | ".join(line) + " |")
        cells.append(cl)
    with open(os.path.join(HERE, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("SR по объектам, в скобках 95% интервал Уилсона\n\n" + "\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(15, 0.6 * len(rows) + 1))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=[RU[o] for o in OBJS], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    ax.set_title("SR по объектам, в скобках 95% интервал Уилсона", pad=10)
    fig.savefig(os.path.join(HERE, "summary_table.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return md


# ---------------- 3. баг с цветом: попытка 1 против попытки 2


def hourly(path):
    out = {}
    for r in read(path):
        if r["obj"] == "chips":
            continue
        h = int(float(r["hours"]))
        k, n = out.get(h, (0, 0))
        out[h] = (k + int(r["reward"]), n + 1)
    return out


def rgb_bug():
    a1 = hourly(os.path.join(RL, "1_rgbd", "night_qmap", "train.csv"))
    a2 = hourly(os.path.join(HERE, "data", "qmap_d_train.csv"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d, lab, c in ((a1, "попытка 1: RGB-D (RGB от прошлой сцены), с нуля", "tab:red"), (a2, "попытка 2: только глубина, точка только на объекте, с весов попытки 1, eps 0.5->0.1 за 8 ч", "tab:green")):
        hs = [h for h in sorted(d) if d[h][1] >= 50]  # хвостовой час с парой попыток не рисую
        p = [d[h][0] / d[h][1] for h in hs]
        lo = [pp - wilson(*d[h])[0] for pp, h in zip(p, hs)]
        hi = [wilson(*d[h])[1] - pp for pp, h in zip(p, hs)]
        ax.errorbar([h + 0.5 for h in hs], p, yerr=[lo, hi], marker="o", capsize=3, color=c, label=lab)
    ax.set_xlabel("часы обучения")
    ax.set_ylabel("доля успехов на обучении\n(тетрапак + банка, с исследованием)")
    ax.set_title("Q-карта: доля успехов на обучении по часам, попытка 1 и попытка 2")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(HERE, "rgb_bug.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)
    return a1, a2


if __name__ == "__main__":
    for l in tolerance_map():
        print(l)
    for l in summary_table():
        print(l)
    a1, a2 = rgb_bug()
    print("попытка 1 по часам:", {h: f"{k}/{n}" for h, (k, n) in sorted(a1.items())})
    print("попытка 2 по часам:", {h: f"{k}/{n}" for h, (k, n) in sorted(a2.items())})
