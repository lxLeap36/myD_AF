# -*- coding: utf-8 -*-
"""
正常训练版：CRM（Complex Ratio Mask）formulation
------------------------------------------------
建议保存为：
    Experiment/train_dl_crm.py

核心思路：
- 输入: [D_r, D_i, X_r, X_i] -> [B,4,T,F]
- 模型输出: [B,T,2F] -> reshape 成 [B,2,T,F]，对应复数 mask 的实部和虚部
- 通过复乘:
      S_hat = M_c * D
- 目标:
      S 的实部 / 虚部
"""

import json
import os
import sys
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.audio_features import apply_complex_mask_ri
from Training.losses import DTMaskedComplexRIL1Loss
from Tools.set_seed import set_seed


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch, device):
    input_feat, target_ri, dt_mask, meta = batch
    input_feat = input_feat.to(device)
    target_ri = target_ri.to(device)
    dt_mask = dt_mask.to(device)
    return input_feat, target_ri, dt_mask, meta


def unpack_model_output_to_mask_ri(pred: torch.Tensor, num_freq_bins: int) -> torch.Tensor:
    """
    pred: [B, T, 2F]
    return:
        mask_ri: [B, 2, T, F]
    """
    b, t, two_f = pred.shape
    if two_f != 2 * num_freq_bins:
        raise ValueError(f"Expected last dim = {2 * num_freq_bins}, got {two_f}")

    pred = pred.view(b, t, 2, num_freq_bins)             # [B, T, 2, F]
    pred = pred.permute(0, 2, 1, 3).contiguous()         # [B, 2, T, F]
    return pred


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_freq_bins: int,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_feat, target_ri, dt_mask, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        pred_flat = model(input_feat)                               # [B, T, 2F]
        pred_mask_ri = unpack_model_output_to_mask_ri(pred_flat, num_freq_bins)   # [B,2,T,F]

        d_ri = input_feat[:, 0:2, :, :]                             # [B,2,T,F]
        pred_s_ri = apply_complex_mask_ri(pred_mask_ri, d_ri)       # [B,2,T,F]

        loss = criterion(pred_s_ri, target_ri, dt_mask)

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

    # CRM 单独输出目录
    cfg["output_dir"] = os.path.join(cfg["root_dir"], "Results", "results_dl_crm")

    # 可调参数
    alpha = 4.0
    dt_weight = 4.0
    non_dt_weight = 0.25
    feature_mode = "crm_ri"
    in_channels = 4
    head_hidden = 256
    output_activation = "identity"

    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    train_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["train_num_samples"],
        split="train",
        feature_mode=feature_mode,
    )
    val_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["val_num_samples"],
        split="val",
        feature_mode=feature_mode,
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

    sample_input, sample_target, sample_dt_mask, _ = train_set[0]
    _, t0, f0 = sample_input.shape

    model = CNNLSTMSTFT(
        num_freq_bins=f0,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        in_channels=in_channels,
        out_dim=2 * f0,
        head_hidden=head_hidden,
        output_activation=output_activation,
    ).to(device)

    criterion = DTMaskedComplexRIL1Loss(
        alpha=alpha,
        dt_weight=dt_weight,
        non_dt_weight=non_dt_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    patience = cfg["train"].get("early_stopping_patience", 5)
    min_delta = cfg["train"].get("early_stopping_min_delta", 1e-4)
    bad_epochs = 0
    last_epoch_trained = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model_crm.pt")
    last_path = os.path.join(ckpt_dir, "last_model_crm.pt")

    epochs = cfg["train"]["epochs"]

    print(f"Starting CRM training for {epochs} epochs on device: {device}.")
    print(f"Output dir: {cfg['output_dir']}")
    print(f"feature_mode={feature_mode}")
    print(f"in_channels={in_channels}, num_freq_bins={f0}, example_frames={t0}")
    print(f"lstm_hidden={cfg['model']['lstm_hidden']}, head_hidden={head_hidden}")

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_freq_bins=f0,
            optimizer=optimizer,
        )

        val_loss = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_freq_bins=f0,
            optimizer=None,
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
                    "formulation": "crm_ri_complex_mask",
                    "feature_mode": feature_mode,
                    "in_channels": in_channels,
                    "out_dim": 2 * f0,
                    "head_hidden": head_hidden,
                    "output_activation": output_activation,
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
                    "formulation": "crm_ri_complex_mask",
                    "feature_mode": feature_mode,
                    "in_channels": in_channels,
                    "out_dim": 2 * f0,
                    "head_hidden": head_hidden,
                    "output_activation": output_activation,
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
            "formulation": "crm_ri_complex_mask",
            "feature_mode": feature_mode,
            "in_channels": in_channels,
            "out_dim": 2 * f0,
            "head_hidden": head_hidden,
            "output_activation": output_activation,
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
            "formulation": "crm_ri_complex_mask",
            "feature_mode": feature_mode,
            "train_num_samples": int(cfg["train_num_samples"]),
            "val_num_samples": int(cfg["val_num_samples"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "lstm_hidden": int(cfg["model"]["lstm_hidden"]),
            "head_hidden": int(head_hidden),
            "in_channels": int(in_channels),
            "out_dim": int(2 * f0),
            "output_activation": output_activation,
        },
        os.path.join(cfg["output_dir"], "train_summary.json"),
    )

    print("CRM training finished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()