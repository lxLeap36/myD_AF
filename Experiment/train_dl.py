import json
import os
import sys
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# 允许从 repo 根目录运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.losses import SpectralL1Loss
from Tools.set_seed import set_seed


def move_batch_to_device(batch, device):
    input_feat, target_mag, meta = batch
    input_feat = input_feat.to(device)
    target_mag = target_mag.to(device)
    return input_feat, target_mag, meta


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_feat, target_mag, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        pred_mag = model(input_feat)                # [B, T, F]
        loss = criterion(pred_mag, target_mag)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = input_feat.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    mean_loss = total_loss / max(total_count, 1)
    return mean_loss


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    cfg = get_config()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # 数据集
    train_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["train_num_samples"],
        split="train",
    )
    val_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["val_num_samples"],
        split="val",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # 用一条样本自动确定 F
    sample_input, sample_target, _ = train_set[0] # 还有meta_dict，额外信息（可选调试）
    _, t0, f0 = sample_input.shape

    model = CNNLSTMSTFT(
        num_freq_bins=f0,
        lstm_hidden=cfg["model"]["lstm_hidden"],
    ).to(device)

    criterion = SpectralL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    last_path = os.path.join(ckpt_dir, "last_model.pt")

    epochs = cfg["train"]["epochs"]
    print(f"Starting training for {epochs} epochs on device: {device}.")
    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(f"[Epoch {epoch:03d}/{epochs:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  -> saved best model to: {best_path}")

        if cfg["train"]["save_every_epoch"]:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "epoch": epoch,
                },
                epoch_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "num_freq_bins": f0,
            "epoch": epochs,
        },
        last_path,
    )

    save_json(history, os.path.join(cfg["output_dir"], "train_history.json"))
    save_json(
        {
            "best_val_loss": float(best_val),
            "best_model_path": best_path,
            "last_model_path": last_path,
            "device_used": device.type,
            "num_freq_bins": int(f0),
            "num_time_frames_example": int(t0),
        },
        os.path.join(cfg["output_dir"], "train_summary.json"),
    )

    print("Training finished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()