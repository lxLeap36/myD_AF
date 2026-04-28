# -*- coding: utf-8 -*-
"""
Hybrid inference:
幅度主分支 + 复数修正分支
----------------------------------------
建议保存为：
    Experiment/infer_dl_hybrid.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import HybridMagComplexNet
from Training.audio_features import (
    apply_real_mask_to_ri,
    ri_channels_to_complex,
    istft_complex,
)
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Metrics.pesq_metric import compute_pesq
from Metrics.si_sdr_metric import compute_si_sdr
from Tools.set_seed import set_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def build_outputs(
    mag_mask: torch.Tensor,
    res_ri_flat: torch.Tensor,
    ri_feat: torch.Tensor,
    *,
    beta_residual: float,
    eps: float = 1e-8,
):
    """
    Args:
        mag_mask   : [B, T, F]
        res_ri_flat: [B, T, 2F]
        ri_feat    : [B, 4, T, F]

    Returns:
        base_s_ri  : [B, 2, T, F]
        delta_ri   : [B, 2, T, F]
        pred_s_ri  : [B, 2, T, F]
        d_mag      : [B, T, F]
    """
    d_ri = ri_feat[:, 0:2, :, :]                                     # [B, 2, T, F]
    d_mag = torch.sqrt(d_ri[:, 0] ** 2 + d_ri[:, 1] ** 2 + eps)      # [B, T, F]

    base_s_ri = apply_real_mask_to_ri(mag_mask, d_ri)                # [B, 2, T, F]

    res_raw_ri = unpack_residual_ri(res_ri_flat, mag_mask.shape[-1]) # [B, 2, T, F]
    delta_ri = beta_residual * torch.tanh(res_raw_ri) * d_mag.unsqueeze(1)

    pred_s_ri = base_s_ri + delta_ri
    return base_s_ri, delta_ri, pred_s_ri, d_mag


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    hybrid_output_dir = os.path.join(cfg["root_dir"], "Results", "results_dl_hybrid")
    ckpt_path = os.path.join(hybrid_output_dir, "checkpoints", "best_model_hybrid.pt")

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = os.path.join(hybrid_output_dir, "inference_hybrid")
    ensure_dir(out_dir)

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    num_freq_bins = checkpoint["num_freq_bins"]
    stem_channels = checkpoint.get("stem_channels", 16)
    trunk_channels = checkpoint.get("trunk_channels", 32)
    head_hidden = checkpoint.get("head_hidden", 256)
    mag_output_activation = checkpoint.get("mag_output_activation", "softplus")
    res_output_activation = checkpoint.get("res_output_activation", "identity")
    beta_residual = checkpoint.get("beta_residual", 0.30)

    print(f"Loaded checkpoint from {ckpt_path}, num_freq_bins={num_freq_bins}")

    model = HybridMagComplexNet(
        num_freq_bins=num_freq_bins,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        stem_channels=stem_channels,
        trunk_channels=trunk_channels,
        head_hidden=head_hidden,
        mag_output_activation=mag_output_activation,
        res_output_activation=res_output_activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    sample_index = cfg["inference"]["sample_index"]
    print(f"Building Hybrid dataset and loading sample index {sample_index}...")

    dataset = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=max(1, sample_index + 1),
        split="val",
        feature_mode="hybrid_mag_ri",
    )

    raw_sample = dataset.get_raw_sample(sample_index)
    (mag_feat, ri_feat), (target_logmag, target_ri), dt_mask_frame, meta = dataset.sample_to_example(raw_sample)

    x = torch.tensor(raw_sample["x"], dtype=torch.float32)
    d = torch.tensor(raw_sample["d"], dtype=torch.float32)
    s = torch.tensor(raw_sample["s"], dtype=torch.float32)

    mag_feat_b = mag_feat.unsqueeze(0).to(device)   # [1, 2, T, F]
    ri_feat_b = ri_feat.unsqueeze(0).to(device)     # [1, 4, T, F]

    with torch.no_grad():
        mag_mask_b, res_ri_flat_b = model(mag_feat_b, ri_feat_b)

        base_s_ri_b, delta_ri_b, pred_s_ri_b, d_mag_b = build_outputs(
            mag_mask_b,
            res_ri_flat_b,
            ri_feat_b,
            beta_residual=beta_residual,
        )

    mag_mask = mag_mask_b[0].cpu()        # [T, F]
    base_s_ri = base_s_ri_b[0].cpu()      # [2, T, F]
    delta_ri = delta_ri_b[0].cpu()        # [2, T, F]
    pred_s_ri = pred_s_ri_b[0].cpu()      # [2, T, F]
    d_mag = d_mag_b[0].cpu()              # [T, F]

    d_ri = ri_feat[0:2]                   # [2, T, F]

    pred_spec = ri_channels_to_complex(pred_s_ri)     # [F, T], complex
    base_spec = ri_channels_to_complex(base_s_ri)     # [F, T], complex
    delta_spec = ri_channels_to_complex(delta_ri)     # [F, T], complex
    target_spec = ri_channels_to_complex(target_ri)   # [F, T], complex
    d_spec = ri_channels_to_complex(d_ri)             # [F, T], complex

    stft_cfg = cfg["stft"]
    n_fft = stft_cfg["n_fft"]
    hop_length = stft_cfg["hop_length"]
    win_length = stft_cfg["win_length"]
    window = torch.hann_window(win_length)

    s_hat_t = istft_complex(
        pred_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu()

    s_base_t = istft_complex(
        base_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu()

    delta_t = istft_complex(
        delta_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu()

    s_hat = s_hat_t.numpy()
    s_base = s_base_t.numpy()
    delta_wav = delta_t.numpy()

    s_np = s.cpu().numpy()
    d_np = d.cpu().numpy()
    x_np = x.cpu().numpy()

    # 指标
    pesq_val = compute_pesq(s_np, s_hat, fs=cfg["sample_rate"])
    si_sdr_val = compute_si_sdr(s_np, s_hat)

    residual = d_np - s_hat
    ref_echo = d_np - s_np
    erle_ref = 10.0 * np.log10(
        (np.sum(ref_echo ** 2) + 1e-12) / (np.sum(residual ** 2) + 1e-12)
    )

    print("开始保存音频...")
    sf.write(os.path.join(out_dir, "far_end_x.wav"), x_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "mic_d.wav"), d_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "clean_near_s.wav"), s_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "pred_near_s_hat.wav"), s_hat, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "pred_base_s.wav"), s_base, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "pred_delta.wav"), delta_wav, cfg["sample_rate"])

    pred_logmag = torch.log1p(torch.abs(pred_spec)).transpose(0, 1).contiguous()     # [T, F]
    base_logmag = torch.log1p(torch.abs(base_spec)).transpose(0, 1).contiguous()
    delta_logmag = torch.log1p(torch.abs(delta_spec)).transpose(0, 1).contiguous()
    target_logmag_plot = torch.log1p(torch.abs(target_spec)).transpose(0, 1).contiguous()
    d_logmag = torch.log1p(torch.abs(d_spec)).transpose(0, 1).contiguous()

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        x=x_np,
        d=d_np,
        s=s_np,
        s_hat=s_hat,
        s_base=s_base,
        delta_wav=delta_wav,
        mag_mask=mag_mask.numpy(),
        base_s_ri=base_s_ri.numpy(),
        delta_ri=delta_ri.numpy(),
        pred_s_ri=pred_s_ri.numpy(),
        target_ri=target_ri.numpy(),
        target_logmag=target_logmag.numpy(),
        pred_logmag=pred_logmag.numpy(),
        base_logmag=base_logmag.numpy(),
        delta_logmag=delta_logmag.numpy(),
        target_logmag_plot=target_logmag_plot.numpy(),
        d_logmag=d_logmag.numpy(),
        d_mag=d_mag.numpy(),
        dt_mask_frame=dt_mask_frame.numpy(),
    )

    # 默认仍只画 3 张，和你之前对比风格保持一致
    fig = plt.figure(figsize=(15, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(pred_logmag.numpy().T, aspect="auto", origin="lower")
    ax1.set_title("log1p(|S_hat|)")
    ax1.set_xlabel("Time frame")
    ax1.set_ylabel("Freq bin")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(target_logmag_plot.numpy().T, aspect="auto", origin="lower")
    ax2.set_title("log1p(|S|)")
    ax2.set_xlabel("Time frame")
    ax2.set_ylabel("Freq bin")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(d_logmag.numpy().T, aspect="auto", origin="lower")
    ax3.set_title("log1p(|D|)")
    ax3.set_xlabel("Time frame")
    ax3.set_ylabel("Freq bin")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectrogram_compare.png"), dpi=150)
    plt.close(fig)

    summary = {
        "checkpoint_path": ckpt_path,
        "formulation": "hybrid_mag_complex_residual",
        "feature_mode": "hybrid_mag_ri",
        "pesq": None if pesq_val is None else float(pesq_val),
        "si_sdr": None if si_sdr_val is None else float(si_sdr_val),
        "erle_reference": float(erle_ref),
        "sample_rate": int(cfg["sample_rate"]),
        "signal_length": int(len(d_np)),
        "beta_residual": float(beta_residual),
        "stem_channels": int(stem_channels),
        "trunk_channels": int(trunk_channels),
        "head_hidden": int(head_hidden),
        "far_path": meta.get("far_path"),
        "near_path": meta.get("near_path"),
        "far_activity_ratio": meta.get("far_activity_ratio"),
        "near_activity_ratio": meta.get("near_activity_ratio"),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    print("Hybrid inference finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()