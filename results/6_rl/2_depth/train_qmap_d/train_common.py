"""Общее для обеих обучалок: буфер опыта, бэкапы, логи, выгрузка на HF.

Бэкапы лежат не в папке запуска (run.sh каждый раз делает новую), а в ~/grasp_task/ckpt/<имя>/,
чтоб после падения следующий запуск с тем же именем нашёл их и продолжил. Туда же логи и картинки.
Раз в 10 мин сохраняю всё вместе: веса + буфер на диск, веса и логи ещё и на HF (resume_test.py так уже делал).
Буфер на HF не гоню, он большой - если диск потеряется, продолжим с весами, но с пустым буфером.
"""

import csv
import os
import time

import torch

from fly_env import IMG, N_YAW, FlyEnv

CKPT_ROOT = os.path.expanduser("~/grasp_task/ckpt")


class Buffer:
    """весь опыт: картинка, что сделали, что получили. Лежит в обычной памяти, на видюху только батч"""

    def __init__(self, cap=40000):
        self.obs = torch.zeros(cap, 4, IMG, IMG, dtype=torch.uint8)
        self.act = torch.zeros(cap, 3)  # x, y, yaw
        self.kij = torch.zeros(cap, 3, dtype=torch.long)  # то же в клетках: yaw-номер, строка, столбец
        self.rew = torch.zeros(cap)
        self.n = 0
        self.cap = cap

    def add(self, obs, act, kij, rew):
        for b in range(len(rew)):
            k = self.n % self.cap
            self.obs[k] = obs[b].cpu()
            self.act[k] = act[b].cpu()
            self.kij[k] = kij[b].cpu()
            self.rew[k] = rew[b].cpu()
            self.n += 1

    def sample(self, bs, dev):
        idx = torch.randint(0, min(self.n, self.cap), (bs,))
        return (self.obs[idx].to(dev), self.act[idx].to(dev), self.kij[idx].to(dev), self.rew[idx].to(dev))

    def state(self):
        m = min(self.n, self.cap)
        return {"obs": self.obs[:m].clone(), "act": self.act[:m].clone(), "kij": self.kij[:m].clone(), "rew": self.rew[:m].clone(), "n": self.n}

    def load(self, s):
        m = len(s["rew"])
        self.obs[:m], self.act[:m], self.kij[:m], self.rew[:m] = s["obs"], s["act"], s["kij"], s["rew"]
        self.n = s["n"]


def to_kij(x, y, yaw):
    i, j = FlyEnv.xy_to_pix(x, y)
    k = torch.round((yaw % torch.pi) / (torch.pi / N_YAW)).long() % N_YAW
    return torch.stack([k, i, j], dim=1)


def object_pixels(img):
    """клетки, где что-то выше стола на 1+ см - туда тыкаемся при случайных попытках, а не в пустой стол"""
    m = img[:, 3] > int(0.01 / 0.25 * 255)
    # полосу у робота (y < 0.29, как в vision.py) выкидываю: там на снимке висит кисть, она тоже "выше стола",
    # и в первую ночь больше половины случайных тычков ушло в неё, а не в объект
    rows = torch.arange(IMG, device=img.device)
    y = FlyEnv.pix_to_xy(rows, torch.zeros_like(rows))[1]
    m[:, y < 0.29, :] = False
    return m


def random_on_object(img):
    """случайная точка на объекте и случайный yaw, для исследования. Если объекта не видно - куда угодно"""
    N = img.shape[0]
    m = object_pixels(img).flatten(1).float()
    m[m.sum(dim=1) == 0] = 1.0
    flat = torch.multinomial(m, 1)[:, 0]
    i, j = flat // IMG, flat % IMG
    x, y = FlyEnv.pix_to_xy(i, j)
    yaw = torch.rand(N, device=img.device) * torch.pi
    return x, y, yaw


class Run:
    """папка бэкапа, логи, таймеры. Всё, что переживает падение"""

    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(CKPT_ROOT, name)
        self.viz = os.path.join(self.dir, "viz")
        os.makedirs(self.viz, exist_ok=True)
        self.train_csv = os.path.join(self.dir, "train.csv")
        self.eval_csv = os.path.join(self.dir, "eval.csv")
        self.t_start = time.time()
        self.t_before = 0.0  # сколько уже училось до этого запуска
        self.last_ckpt = self.last_eval = time.time()

    def hours(self):
        return (self.t_before + time.time() - self.t_start) / 3600

    def log_train(self, attempt, objs, act, rew, eps, true):
        # true - настоящий центр и yaw объекта (из симулятора), только для разбора, агенту не даётся
        new = not os.path.exists(self.train_csv)
        with open(self.train_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["attempt", "hours", "obj", "x", "y", "yaw", "reward", "eps", "obj_x", "obj_y", "obj_yaw"])
            for b in range(len(rew)):
                w.writerow([attempt + b, f"{self.hours():.4f}", objs[b], f"{act[b, 0]:.4f}", f"{act[b, 1]:.4f}", f"{act[b, 2]:.3f}", int(rew[b]), f"{eps:.3f}",
                            f"{true[b, 0]:.4f}", f"{true[b, 1]:.4f}", f"{true[b, 2]:.3f}"])

    def log_eval(self, attempt, stats):
        new = not os.path.exists(self.eval_csv)
        with open(self.eval_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["attempt", "hours", "obj", "ok", "n"])
            for obj, (ok, n) in stats.items():
                w.writerow([attempt, f"{self.hours():.4f}", obj, ok, n])

    def _cut_csv(self, path, attempt):
        # после падения выкидываю строки, которых нет в бэкапе (их пройдём заново)
        if not os.path.exists(path):
            return
        with open(path) as f:
            rows = list(csv.reader(f))
        keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) < attempt]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(keep)

    def save(self, state, buf=None):
        tmp = os.path.join(self.dir, "last.pt.tmp")
        state = {**state, "hours": self.hours()}
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(self.dir, "last.pt"))  # подмена целиком, чтоб не оставить битый файл
        if buf is not None:
            tmp = os.path.join(self.dir, "buffer.pt.tmp")
            torch.save(buf.state(), tmp)
            os.replace(tmp, os.path.join(self.dir, "buffer.pt"))
        print(f"[ckpt] попытка {state['attempt']}, {self.hours():.2f} ч, буфер {'да' if buf is not None else 'нет'}", flush=True)
        self.upload()

    def load(self, buf):
        """вернёт сохранённое состояние или None. Сначала с диска, если там нет - веса и логи с HF (буфер тогда пустой)"""
        path = os.path.join(self.dir, "last.pt")
        if not os.path.exists(path):
            try:
                from huggingface_hub import hf_hub_download

                repo = os.environ["GT_HF_REPO"]
                for f in ("last.pt", "train.csv", "eval.csv"):
                    p = hf_hub_download(repo_id=repo, filename=f"rl/{self.name}/{f}")
                    with open(p, "rb") as src, open(os.path.join(self.dir, f), "wb") as dst:
                        dst.write(src.read())
                print("[ckpt] на диске пусто, взял веса и логи с HF", flush=True)
            except Exception as e:
                print(f"[ckpt] бэкапа нет ни на диске, ни на HF ({type(e).__name__}), начинаю с нуля", flush=True)
                # упали до первого бэкапа - старые строки логов выкидываю, а то попытки задвоятся
                for p in (self.train_csv, self.eval_csv):
                    if os.path.exists(p):
                        os.remove(p)
                return None
        state = torch.load(path, map_location="cpu")
        bpath = os.path.join(self.dir, "buffer.pt")
        if os.path.exists(bpath):
            buf.load(torch.load(bpath, map_location="cpu"))
        attempt = state["attempt"]
        self.t_before = state["hours"] * 3600
        self._cut_csv(self.train_csv, attempt)
        self._cut_csv(self.eval_csv, attempt + 1)
        print(f"[ckpt] продолжаю с попытки {attempt}, уже училось {state['hours']:.2f} ч, в буфере {buf.n}", flush=True)
        return state

    def upload(self):
        try:
            from huggingface_hub import HfApi

            HfApi().upload_folder(
                folder_path=self.dir,
                path_in_repo=f"rl/{self.name}",
                repo_id=os.environ["GT_HF_REPO"],
                allow_patterns=["last.pt", "train.csv", "eval.csv", "viz/*.png"],
                commit_message=f"{self.name} {self.hours():.2f}h",
            )
        except Exception as e:
            # не загрузилось - не страшно, на диске всё есть. Просто пишу в лог
            print(f"[hf] не загрузилось: {type(e).__name__}: {str(e)[:200]}", flush=True)
