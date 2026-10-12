"""Сеть Q-карты и жадный выбор - вынес из 6_rl/2_depth/train_qmap.py как есть, чтоб все варианты брали одно и то же.
Импортить после make_app() (тянет fly_env).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fly_env import IMG, N_YAW, FlyEnv
from train_common import object_pixels

# --- маленький U-Net. Вход 4 канала (RGB + высота, RGB зануляю), выход 8 каналов (по одному на yaw) того же размера


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class QMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = block(4, 32)  # 128
        self.d2 = block(32, 64)  # 64
        self.d3 = block(64, 128)  # 32
        self.mid = block(128, 128)  # 16
        self.u3 = block(128 + 128, 64)
        self.u2 = block(64 + 64, 32)
        self.u1 = block(32 + 32, 32)
        self.out = nn.Conv2d(32, N_YAW, 1)

    def forward(self, img):
        x = img.float() / 255.0
        # только глубина, как в попытке 2: RGB на нули, каналов 4, чтоб подходили старые веса
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return self.out(u)  # [B,8,128,128] логиты


def greedy(net, img):
    """лучшая клетка и yaw по сети, только среди клеток на объекте (если объекта не видно - где угодно)"""
    with torch.no_grad():
        logits = net(img)
    on_obj = object_pixels(img)
    on_obj[on_obj.flatten(1).sum(1) == 0] = True
    masked = logits.masked_fill(~on_obj.unsqueeze(1), float("-inf"))
    flat = masked.flatten(1).argmax(dim=1)
    k, rest = flat // (IMG * IMG), flat % (IMG * IMG)
    i, j = rest // IMG, rest % IMG
    x, y = FlyEnv.pix_to_xy(i, j)
    return x, y, k.float() * torch.pi / N_YAW, torch.sigmoid(logits)


# --- вариант 2: та же сеть + код объекта через FiLM. Код 0 - "старые" (тетрапак, банка), 1 - чипсы.
# После каждого из 7 блоков x * (1 + g) + b, где g и b берутся из эмбеддинга кода. На старте всё нули,
# так что сеть ровно старая - первая проверка обязана повторить old_agent (42 / 86 / 92 %)

CODE = {"tetrapak": 0, "can": 0, "chips": 1}


class QMapFilm(QMap):
    def __init__(self):
        super().__init__()
        self.chans = {"d1": 32, "d2": 64, "d3": 128, "mid": 128, "u3": 64, "u2": 32, "u1": 32}
        self.film = nn.ModuleDict({k: nn.Embedding(2, 2 * c) for k, c in self.chans.items()})
        for e in self.film.values():
            nn.init.zeros_(e.weight)
        self.code = None  # [B] long, ставлю перед forward через set_code

    def set_code(self, code):
        self.code = code

    def _f(self, name, x):
        g, b = self.film[name](self.code).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, img):
        x = img.float() / 255.0
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)
        a = self._f("d1", self.d1(x))
        b = self._f("d2", self.d2(F.max_pool2d(a, 2)))
        c = self._f("d3", self.d3(F.max_pool2d(b, 2)))
        m = self._f("mid", self.mid(F.max_pool2d(c, 2)))
        u = self._f("u3", self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1)))
        u = self._f("u2", self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1)))
        u = self._f("u1", self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1)))
        return self.out(u)


def codes(objs, dev):
    return torch.tensor([CODE[o] for o in objs], device=dev)


def load_old(net, path):
    """старые веса agent/qmap_depth.pt в QMap, QMapFilm или QMapAdapter. FiLM и адаптеров в файле нет - они остаются как при создании"""
    missing, unexpected = net.load_state_dict(torch.load(path, map_location="cpu")["net"], strict=False)
    assert not unexpected and all(k.startswith(("film.", "ad.")) for k in missing), (missing, unexpected)
    return net


# --- вариант 4б: старая сеть заморожена, после каждого из 7 блоков адаптер x + up(relu(down(x))), сжатие в 4 раза.
# Без переключателя (решение 12.10): адаптеры работают на всех объектах, тип объекта сети не нужен.
# up на старте нули, так что сеть ровно старая. Учатся только адаптеры (~22 тыс. весов)


class Adapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.down = nn.Conv2d(c, c // 4, 1)
        self.up = nn.Conv2d(c // 4, c, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


class QMapAdapter(QMap):
    def __init__(self):
        super().__init__()
        chans = {"d1": 32, "d2": 64, "d3": 128, "mid": 128, "u3": 64, "u2": 32, "u1": 32}
        self.ad = nn.ModuleDict({k: Adapter(c) for k, c in chans.items()})

    def freeze_base(self):
        """всё, кроме адаптеров, не учится. Вернёт то, что отдать в Adam"""
        for n, p in self.named_parameters():
            p.requires_grad = n.startswith("ad.")
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, img):
        x = img.float() / 255.0
        x = torch.cat([torch.zeros_like(x[:, :3]), x[:, 3:]], 1)
        a = self.ad["d1"](self.d1(x))
        b = self.ad["d2"](self.d2(F.max_pool2d(a, 2)))
        c = self.ad["d3"](self.d3(F.max_pool2d(b, 2)))
        m = self.ad["mid"](self.mid(F.max_pool2d(c, 2)))
        u = self.ad["u3"](self.u3(torch.cat([F.interpolate(m, scale_factor=2), c], 1)))
        u = self.ad["u2"](self.u2(torch.cat([F.interpolate(u, scale_factor=2), b], 1)))
        u = self.ad["u1"](self.u1(torch.cat([F.interpolate(u, scale_factor=2), a], 1)))
        return self.out(u)
