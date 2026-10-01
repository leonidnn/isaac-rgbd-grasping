# Цель скрипта - проверить отказоустройчивость, потому что на сервере процесс может оборваться в любой момент

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


RUN_DIR = Path(os.environ["GT_RUN_DIR"]) # Это временная директория, которую создает run
HF_REPO = os.environ["GT_HF_REPO"]

CKPT_DIR = RUN_DIR / "ckpt"
LOCAL_CKPT_PATH = CKPT_DIR / "last.pt"
HF_CKPT_PATH = "resume_test/last.pt"  # Для HF оставляем обычной строкой
UPLOAD_FLAG_PATH = RUN_DIR / "UPLOAD_CONFIRMED"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

device = torch.device("cuda") # CPU строго нельзя по правилам сервера
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

start_step = 0
run_id = None

try:
    print(f"Проверяем чекпоинт {HF_REPO}...")
    downloaded_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT_PATH)
    ckpt = torch.load(downloaded_path, map_location=device)
    
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    run_id = ckpt["run_id"]
    print(f"Продолжили с шага {start_step}, run_id: {run_id}")
except (EntryNotFoundError, Exception) as e:
    run_id = wandb.util.generate_id()
    print(f"Чекпоинт не найден ({e}). Начинаем с шага 0 новый run_id: {run_id}")

# Инициализация WnB
wandb.init(
    project="grasp-rgbd",
    id=run_id,
    resume="allow"
)

def save_and_upload(current_step):
    ckpt_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": current_step,
        "run_id": run_id
    }
    torch.save(ckpt_data, LOCAL_CKPT_PATH)
    api.upload_file(
        path_or_fileobj=LOCAL_CKPT_PATH,
        path_in_repo=HF_CKPT_PATH,
        repo_id=HF_REPO
    )
    print(f"[Step {current_step}] Checkpoint saved & uploaded.")

# Цикл обучения
total_steps = 200
save_interval = 20

for step in range(start_step, total_steps):
    x = torch.rand(64, 1, device=device) * 2 * 3.14159 - 3.14159
    y_true = torch.sin(x)

    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    wandb.log({"loss": loss.item(), "step": step}, step=step)
    print(f"Шаг {step:03d} | Loss: {loss.item():.4f}")

    time.sleep(0.5)

    if (step + 1) % save_interval == 0:
        save_and_upload(step)


save_and_upload(total_steps - 1)

files = api.list_repo_files(repo_id=HF_REPO)
if HF_CKPT_PATH in files:
    print(f"Подтверждено: {HF_CKPT_PATH} в репозитории.")
    with open(UPLOAD_FLAG_PATH, "w") as f:
        pass
    print("UPLOAD_CONFIRMED создан.")
else:
    print("Error: Файл не найден в HF репозитории")

wandb.finish()