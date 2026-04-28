# -*- coding: utf-8 -*-
"""
Hybrid training:
幅度主分支 + 复数修正分支
----------------------------------------
建议保存为：
    Experiment/train_dl_hybrid.py

结构：
- 输入:
    mag_feat: [B, 2, T, F] = [log1p(|D|), log1p(|X|)]
    ri_feat : [B, 4, T, F] = [D_r, D_i, X_r, X_i]

- 模型输出:
    mag_mask   : [B, T, F]
    res_ri_flat: [B, T, 2F]

- 构造:
    S_base = mag_mask * D
    DeltaS = beta * tanh(res_raw) * |D|
    S_hat  = S_base + DeltaS

- 损失:
    L = lambda_mag * L_mag
      + lambda_ri  * L_ri
      + lambda_wav * L_wav
      + lambda_res * L_res
"""

import json
import os
import sys
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import HybridMagComplexNet
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.audio_features import (
    apply_real_mask_to_ri,
    ri_channels_to_complex,
    istft_complex,
)
from Training.losses import (
    DTMaskedWeightedSpectralL1Loss,
    DTMaskedComplexRIL1Loss,
)
from Tools.set_seed import set_seed


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch, device):
    (mag_feat, ri_feat), (target_logmag, target_ri), dt_mask, meta = batch
    mag_feat = mag_feat.to(device)
    ri_feat = ri_feat.to(device)
    target_logmag = target_logmag.to(device)
    target_ri = target_ri.to(device)
    dt_mask = dt_mask.to(device)
    return mag_feat, ri_feat, target_logmag, target_ri, dt_mask, meta


def unpack_residual_ri(pred: torch.Tensor, num_freq_bins: int) -> torch.Tensor:
    """
    pred: [B, T, 2F]
    return:
        res_ri: [B, 2, T, F]
    """
    b, t, two_f = pred.shape
    if two_f != 2 * num_freq_bins:
        raise ValueError(f"Expected last dim = {2 * num_freq_bins}, got {two_f}")

    pred = pred.view(b, t, 2, num_freq_bins)         # [B, T, 2, F]
    pred = pred.permute(0, 2, 1, 3).contiguous()     # [B, 2, T, F]
    return pred


def compute_time_loss(
    pred_s_ri: torch.Tensor,
    target_ri: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    signal_length: int,
    time_loss_type: str = "wave_l1",
) -> torch.Tensor:
    pred_spec = ri_channels_to_complex(pred_s_ri)      # [B, F, T]
    target_spec = ri_channels_to_complex(target_ri)    # [B, F, T]

    pred_wav = istft_complex(
        pred_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_length,
    )  # [B, N]

    target_wav = istft_complex(
        target_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_length,
    )  # [B, N]

    if time_loss_type == "wave_l1":
        return torch.mean(torch.abs(pred_wav - target_wav))

    raise ValueError(f"Unsupported time_loss_type: {time_loss_type}")


def build_outputs(
    mag_mask: torch.Tensor,
    res_ri_flat: torch.Tensor,
    ri_feat: torch.Tensor,
    *,
    beta_residual: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        mag_mask   : [B, T, F]
        res_ri_flat: [B, T, 2F]
        ri_feat    : [B, 4, T, F] = [D_r, D_i, X_r, X_i]

    Returns:
        base_s_ri  : [B, 2, T, F]
        delta_ri   : [B, 2, T, F]
        pred_s_ri  : [B, 2, T, F]
        d_mag      : [B, T, F]
    """
    d_ri = ri_feat[:, 0:2, :, :]                       # [B, 2, T, F]
    d_mag = torch.sqrt(d_ri[:, 0] ** 2 + d_ri[:, 1] ** 2 + eps)   # [B, T, F]

    base_s_ri = apply_real_mask_to_ri(mag_mask, d_ri)              # [B, 2, T, F]

    res_raw_ri = unpack_residual_ri(res_ri_flat, mag_mask.shape[-1])   # [B, 2, T, F]
    delta_ri = beta_residual * torch.tanh(res_raw_ri) * d_mag.unsqueeze(1)

    pred_s_ri = base_s_ri + delta_ri
    return base_s_ri, delta_ri, pred_s_ri, d_mag


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    mag_criterion: torch.nn.Module,
    ri_criterion: torch.nn.Module,
    device: torch.device,
    num_freq_bins: int,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    signal_length: int,
    time_loss_type: str,
    beta_residual: float,
    lambda_mag: float,
    lambda_ri: float,
    lambda_wav: float,
    lambda_res: float,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total = {
        "loss": 0.0,
        "mag_loss": 0.0,
        "ri_loss": 0.0,
        "wav_loss": 0.0,
        "res_loss": 0.0,
        "count": 0,
    }

    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    with grad_context:
        for batch in loader:
            mag_feat, ri_feat, target_logmag, target_ri, dt_mask, _ = move_batch_to_device(batch, device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            mag_mask, res_ri_flat = model(mag_feat, ri_feat)

            base_s_ri, delta_ri, pred_s_ri, d_mag = build_outputs(
                mag_mask,
                res_ri_flat,
                ri_feat,
                beta_residual=beta_residual,
            )

            # 主分支幅度监督
            base_mag = torch.sqrt(base_s_ri[:, 0] ** 2 + base_s_ri[:, 1] ** 2 + 1e-8)   # [B, T, F]
            base_logmag = torch.log1p(torch.clamp(base_mag, min=0.0))
            mag_loss = mag_criterion(base_logmag, target_logmag, dt_mask)

            # 最终复谱监督
            ri_loss = ri_criterion(pred_s_ri, target_ri, dt_mask)

            # 时域辅助监督
            wav_loss = compute_time_loss(
                pred_s_ri,
                target_ri,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                signal_length=signal_length,
                time_loss_type=time_loss_type,
            )

            # 修正分支正则
            res_loss = torch.mean(torch.abs(delta_ri) / (d_mag.unsqueeze(1) + 1e-8))

            loss = (
                lambda_mag * mag_loss
                + lambda_ri * ri_loss
                + lambda_wav * wav_loss
                + lambda_res * res_loss
            )

            if is_train:
                loss.backward()
                optimizer.step()

            bsz = mag_feat.size(0)
            total["loss"] += loss.item() * bsz
            total["mag_loss"] += mag_loss.item() * bsz
            total["ri_loss"] += ri_loss.item() * bsz
            total["wav_loss"] += wav_loss.item() * bsz
            total["res_loss"] += res_loss.item() * bsz
            total["count"] += bsz

            del mag_feat, ri_feat, target_logmag, target_ri, dt_mask
            del mag_mask, res_ri_flat, base_s_ri, delta_ri, pred_s_ri, d_mag
            del base_mag, base_logmag
            del mag_loss, ri_loss, wav_loss, res_loss, loss

    count = max(total["count"], 1)
    return {
        "loss": total["loss"] / count,
        "mag_loss": total["mag_loss"] / count,
        "ri_loss": total["ri_loss"] / count,
        "wav_loss": total["wav_loss"] / count,
        "res_loss": total["res_loss"] / count,
    }


def main():
    cfg = get_config()

    # 单独输出目录
    cfg["output_dir"] = os.path.join(cfg["root_dir"], "Results", "results_dl_hybrid")

    # -----------------------------
    # 可调参数
    # -----------------------------
    feature_mode = "hybrid_mag_ri"

    stem_channels = 16
    trunk_channels = 32
    head_hidden = 256

    mag_output_activation = "softplus"
    res_output_activation = "identity"

    # 复数修正分支的尺度限制
    beta_residual = 0.30

    # loss 系数
    lambda_mag = 1.0
    lambda_ri = 0.5
    lambda_wav = 0.05
    lambda_res = 0.01

    # 损失内部权重
    alpha_mag = 4.0
    alpha_ri = 4.0
    dt_weight = 4.0
    non_dt_weight = 0.25

    time_loss_type = "wave_l1"

    # -----------------------------
    # 目录 / 随机种子 / 设备
    # -----------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # -----------------------------
    # STFT / waveform loss 配置
    # -----------------------------
    stft_cfg = cfg["stft"]
    n_fft = int(stft_cfg["n_fft"])
    hop_length = int(stft_cfg["hop_length"])
    win_length = int(stft_cfg["win_length"])
    signal_length = int(round(cfg["sample_rate"] * cfg["duration_sec"]))
    window = torch.hann_window(win_length, device=device)

    # -----------------------------
    # 数据集
    # -----------------------------
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

    # 取一个样本推断 F / T
    (mag_feat0, ri_feat0), (target_logmag0, target_ri0), dt_mask0, _ = train_set[0]
    _, t0, f0 = mag_feat0.shape

    # -----------------------------
    # 模型 / 损失 / 优化器
    # -----------------------------
    model = HybridMagComplexNet(
        num_freq_bins=f0,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        stem_channels=stem_channels,
        trunk_channels=trunk_channels,
        head_hidden=head_hidden,
        mag_output_activation=mag_output_activation,
        res_output_activation=res_output_activation,
    ).to(device)

    mag_criterion = DTMaskedWeightedSpectralL1Loss(
        alpha=alpha_mag,
        dt_weight=dt_weight,
        non_dt_weight=non_dt_weight,
    )

    ri_criterion = DTMaskedComplexRIL1Loss(
        alpha=alpha_ri,
        dt_weight=dt_weight,
        non_dt_weight=non_dt_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # -----------------------------
    # Early stopping
    # -----------------------------
    patience = cfg["train"].get("early_stopping_patience", 5)
    min_delta = cfg["train"].get("early_stopping_min_delta", 1e-4)
    bad_epochs = 0
    last_epoch_trained = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mag_loss": [],
        "val_mag_loss": [],
        "train_ri_loss": [],
        "val_ri_loss": [],
        "train_wav_loss": [],
        "val_wav_loss": [],
        "train_res_loss": [],
        "val_res_loss": [],
    }

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model_hybrid.pt")
    last_path = os.path.join(ckpt_dir, "last_model_hybrid.pt")

    epochs = cfg["train"]["epochs"]

    print(f"Starting Hybrid training for {epochs} epochs on device: {device}.")
    print(f"Output dir: {cfg['output_dir']}")
    print(f"feature_mode={feature_mode}")
    print(f"num_freq_bins={f0}, example_frames={t0}")
    print(f"lstm_hidden={cfg['model']['lstm_hidden']}, stem_channels={stem_channels}, trunk_channels={trunk_channels}, head_hidden={head_hidden}")
    print(f"beta_residual={beta_residual}")
    print(f"loss_weights: mag={lambda_mag}, ri={lambda_ri}, wav={lambda_wav}, res={lambda_res}")

    for epoch in range(1, epochs + 1):
        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            mag_criterion=mag_criterion,
            ri_criterion=ri_criterion,
            device=device,
            num_freq_bins=f0,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            signal_length=signal_length,
            time_loss_type=time_loss_type,
            beta_residual=beta_residual,
            lambda_mag=lambda_mag,
            lambda_ri=lambda_ri,
            lambda_wav=lambda_wav,
            lambda_res=lambda_res,
            optimizer=optimizer,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            mag_criterion=mag_criterion,
            ri_criterion=ri_criterion,
            device=device,
            num_freq_bins=f0,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            signal_length=signal_length,
            time_loss_type=time_loss_type,
            beta_residual=beta_residual,
            lambda_mag=lambda_mag,
            lambda_ri=lambda_ri,
            lambda_wav=lambda_wav,
            lambda_res=lambda_res,
            optimizer=None,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        history["train_loss"].append(float(train_stats["loss"]))
        history["val_loss"].append(float(val_stats["loss"]))
        history["train_mag_loss"].append(float(train_stats["mag_loss"]))
        history["val_mag_loss"].append(float(val_stats["mag_loss"]))
        history["train_ri_loss"].append(float(train_stats["ri_loss"]))
        history["val_ri_loss"].append(float(val_stats["ri_loss"]))
        history["train_wav_loss"].append(float(train_stats["wav_loss"]))
        history["val_wav_loss"].append(float(val_stats["wav_loss"]))
        history["train_res_loss"].append(float(train_stats["res_loss"]))
        history["val_res_loss"].append(float(val_stats["res_loss"]))

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"(mag={train_stats['mag_loss']:.6f}, ri={train_stats['ri_loss']:.6f}, "
            f"wav={train_stats['wav_loss']:.6f}, res={train_stats['res_loss']:.6f}) "
            f"val_loss={val_stats['loss']:.6f} "
            f"(mag={val_stats['mag_loss']:.6f}, ri={val_stats['ri_loss']:.6f}, "
            f"wav={val_stats['wav_loss']:.6f}, res={val_stats['res_loss']:.6f})"
        )

        last_epoch_trained = epoch
        improved = val_stats["loss"] < (best_val - min_delta)

        if improved:
            best_val = val_stats["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "formulation": "hybrid_mag_complex_residual",
                    "feature_mode": feature_mode,
                    "stem_channels": stem_channels,
                    "trunk_channels": trunk_channels,
                    "head_hidden": head_hidden,
                    "mag_output_activation": mag_output_activation,
                    "res_output_activation": res_output_activation,
                    "beta_residual": beta_residual,
                    "lambda_mag": lambda_mag,
                    "lambda_ri": lambda_ri,
                    "lambda_wav": lambda_wav,
                    "lambda_res": lambda_res,
                    "time_loss_type": time_loss_type,
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
                    "formulation": "hybrid_mag_complex_residual",
                    "feature_mode": feature_mode,
                    "stem_channels": stem_channels,
                    "trunk_channels": trunk_channels,
                    "head_hidden": head_hidden,
                    "mag_output_activation": mag_output_activation,
                    "res_output_activation": res_output_activation,
                    "beta_residual": beta_residual,
                    "lambda_mag": lambda_mag,
                    "lambda_ri": lambda_ri,
                    "lambda_wav": lambda_wav,
                    "lambda_res": lambda_res,
                    "time_loss_type": time_loss_type,
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
            "formulation": "hybrid_mag_complex_residual",
            "feature_mode": feature_mode,
            "stem_channels": stem_channels,
            "trunk_channels": trunk_channels,
            "head_hidden": head_hidden,
            "mag_output_activation": mag_output_activation,
            "res_output_activation": res_output_activation,
            "beta_residual": beta_residual,
            "lambda_mag": lambda_mag,
            "lambda_ri": lambda_ri,
            "lambda_wav": lambda_wav,
            "lambda_res": lambda_res,
            "time_loss_type": time_loss_type,
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
            "formulation": "hybrid_mag_complex_residual",
            "feature_mode": feature_mode,
            "train_num_samples": int(cfg["train_num_samples"]),
            "val_num_samples": int(cfg["val_num_samples"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "lstm_hidden": int(cfg["model"]["lstm_hidden"]),
            "stem_channels": int(stem_channels),
            "trunk_channels": int(trunk_channels),
            "head_hidden": int(head_hidden),
            "mag_output_activation": mag_output_activation,
            "res_output_activation": res_output_activation,
            "beta_residual": float(beta_residual),
            "lambda_mag": float(lambda_mag),
            "lambda_ri": float(lambda_ri),
            "lambda_wav": float(lambda_wav),
            "lambda_res": float(lambda_res),
            "time_loss_type": time_loss_type,
            "signal_length": int(signal_length),
        },
        os.path.join(cfg["output_dir"], "train_summary.json"),
    )

    print("Hybrid training finished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()