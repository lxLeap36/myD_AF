# -*- coding: utf-8 -*-
"""
正常训练版：mask formulation
-----------------------------------------
建议保存为：
    Experiment/train_dl_mask.py

核心改动：
- 模型输出不再解释为 log1p(|S_hat|)
- 而是解释为非负 mask M(t,f)
- 由 |S_hat| = M * |D|
- 再转成 log1p(|S_hat|) 去与 target 的 log1p(|S|) 比较

运行：
    python Experiment/train_dl_mask.py
"""

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
from Training.losses import DTMaskedWeightedSpectralL1Loss
from Tools.set_seed import set_seed


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch, device):
    input_feat, target_logmag, dt_mask, meta = batch
    input_feat = input_feat.to(device)
    target_logmag = target_logmag.to(device)
    dt_mask = dt_mask.to(device)
    return input_feat, target_logmag, dt_mask, meta


def reconstruct_logmag_from_mask(
    pred_mask: torch.Tensor,
    input_feat: torch.Tensor,
    eps: float = 1e-8,
):
    """
    pred_mask: [B, T, F], 非负
    input_feat: [B, 2, T, F], channel 0 = log1p(|D|)

    return:
        recon_logmag: [B, T, F] = log1p(pred_mask * |D|)
        d_mag_lin:    [B, T, F]
    """
    d_logmag = input_feat[:, 0, :, :]                     # [B,T,F]
    d_mag_lin = torch.expm1(torch.clamp(d_logmag, min=0.0))
    pred_mag_lin = pred_mask * d_mag_lin
    recon_logmag = torch.log1p(torch.clamp(pred_mag_lin, min=0.0) + eps)
    return recon_logmag, d_mag_lin


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    max_mask_value: float = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_feat, target_logmag, dt_mask, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        # 仍复用当前模型，但把输出解释成 mask
        pred_mask = model(input_feat)   # [B, T, F], Softplus -> non-negative

        if max_mask_value is not None:
            pred_mask = torch.clamp(pred_mask, max=max_mask_value)

        recon_logmag, _ = reconstruct_logmag_from_mask(pred_mask, input_feat)
        loss = criterion(recon_logmag, target_logmag, dt_mask)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = input_feat.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    mean_loss = total_loss / max(total_count, 1)
    return mean_loss


def main():
    cfg = get_config()

    # -----------------------------
    # 不改原 config 文件，只在这里覆盖
    # -----------------------------
    cfg["output_dir"] = os.path.join(cfg["root_dir"], "Results", "results_dl_mask")

    # ===== 可调参数（先建议保持这组）=====
    max_mask_value = 5.0
    alpha = 4.0
    dt_weight = 4.0
    non_dt_weight = 0.25
    # ====================================

    # 配置与目录准备
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 随机种子与设备
    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
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
    sample_input, sample_target, sample_dt_mask, _ = train_set[0]
    _, t0, f0 = sample_input.shape

    # 模型、损失、优化器
    model = CNNLSTMSTFT(
        num_freq_bins=f0,
        lstm_hidden=cfg["model"]["lstm_hidden"],
    ).to(device)

    criterion = DTMaskedWeightedSpectralL1Loss(
        alpha=alpha,
        dt_weight=dt_weight,
        non_dt_weight=non_dt_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # early stopping 配置
    patience = cfg["train"].get("early_stopping_patience", 5)
    min_delta = cfg["train"].get("early_stopping_min_delta", 1e-4)
    bad_epochs = 0
    last_epoch_trained = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model_mask.pt")
    last_path = os.path.join(ckpt_dir, "last_model_mask.pt")

    epochs = cfg["train"]["epochs"]

    print(f"Starting MASK training for {epochs} epochs on device: {device}.")
    print(f"Output dir: {cfg['output_dir']}")
    print(f"max_mask_value={max_mask_value}")
    print(f"num_freq_bins={f0}, example_frames={t0}")

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_mask_value=max_mask_value,
        )

        val_loss = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_mask_value=max_mask_value,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        last_epoch_trained = epoch
        improved = val_loss < (best_val - min_delta)

        if improved:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "formulation": "mask_times_D_mag",
                    "max_mask_value": max_mask_value,
                },
                best_path,
            )
            print(f"  -> saved best model to: {best_path}")
        else:
            bad_epochs += 1
            print(f"  -> no improvement for {bad_epochs}/{patience} epoch(s)")

        if cfg["train"]["save_every_epoch"]:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "epoch": epoch,
                    "formulation": "mask_times_D_mag",
                    "max_mask_value": max_mask_value,
                },
                epoch_path,
            )

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "num_freq_bins": f0,
            "epoch": last_epoch_trained,
            "formulation": "mask_times_D_mag",
            "max_mask_value": max_mask_value,
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
            "actual_epochs_trained": int(last_epoch_trained),
            "early_stopping_patience": int(patience),
            "early_stopping_min_delta": float(min_delta),
            "formulation": "mask_times_D_mag",
            "max_mask_value": float(max_mask_value) if max_mask_value is not None else None,
            "train_num_samples": int(cfg["train_num_samples"]),
            "val_num_samples": int(cfg["val_num_samples"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "lstm_hidden": int(cfg["model"]["lstm_hidden"]),
        },
        os.path.join(cfg["output_dir"], "train_summary.json"),
    )

    print("MASK training finished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()