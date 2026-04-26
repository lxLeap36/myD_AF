# -*- coding: utf-8 -*-
"""
CRM（Complex Ratio Mask）版推理脚本
----------------------------------
建议保存为：
    Experiment/infer_dl_crm.py
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
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.audio_features import (
    apply_complex_mask_ri,
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


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    crm_output_dir = os.path.join(cfg["root_dir"], "Results", "results_dl_crm_wave_l1")
    ckpt_path = os.path.join(crm_output_dir, "checkpoints", "best_model_crm.pt")

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = os.path.join(cfg["output_dir"], "inference_crm_wave_l1")
    ensure_dir(out_dir)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    num_freq_bins = checkpoint["num_freq_bins"]
    in_channels = checkpoint.get("in_channels", 4)
    head_hidden = checkpoint.get("head_hidden", 256)
    output_activation = checkpoint.get("output_activation", "identity")

    print(f"Loaded checkpoint from {ckpt_path}, num_freq_bins={num_freq_bins}")

    model = CNNLSTMSTFT(
        num_freq_bins=num_freq_bins,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        in_channels=in_channels,
        out_dim=2 * num_freq_bins,
        head_hidden=head_hidden,
        output_activation=output_activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    sample_index = cfg["inference"]["sample_index"]
    print(f"Building CRM dataset and loading sample index {sample_index}...")

    dataset = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=max(1, sample_index + 1),
        split="val",
        feature_mode="crm_ri",
    )

    raw_sample = dataset.get_raw_sample(sample_index)
    input_feat, target_ri, dt_mask_frame, meta = dataset.sample_to_example(raw_sample)

    x = torch.tensor(raw_sample["x"], dtype=torch.float32)
    d = torch.tensor(raw_sample["d"], dtype=torch.float32)
    s = torch.tensor(raw_sample["s"], dtype=torch.float32)

    input_feat_b = input_feat.unsqueeze(0).to(device)    # [1,4,T,F]

    with torch.no_grad():
        pred_flat = model(input_feat_b)                              # [1,T,2F]
        pred_mask_ri = unpack_model_output_to_mask_ri(pred_flat, num_freq_bins)[0].cpu()  # [2,T,F]

    d_ri = input_feat[0:2]                                           # [2,T,F]
    pred_s_ri = apply_complex_mask_ri(pred_mask_ri, d_ri)            # [2,T,F]

    pred_spec = ri_channels_to_complex(pred_s_ri)                    # [F,T], complex
    target_spec = ri_channels_to_complex(target_ri)                  # [F,T], complex
    d_spec = ri_channels_to_complex(d_ri)                            # [F,T], complex

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

    s_hat = s_hat_t.numpy()
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

    pred_logmag = torch.log1p(torch.abs(pred_spec)).transpose(0, 1).contiguous()   # [T,F]
    target_logmag = torch.log1p(torch.abs(target_spec)).transpose(0, 1).contiguous()
    d_logmag = torch.log1p(torch.abs(d_spec)).transpose(0, 1).contiguous()

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        x=x_np,
        d=d_np,
        s=s_np,
        s_hat=s_hat,
        pred_mask_ri=pred_mask_ri.numpy(),
        pred_s_ri=pred_s_ri.numpy(),
        target_ri=target_ri.numpy(),
        pred_logmag=pred_logmag.numpy(),
        target_logmag=target_logmag.numpy(),
        d_logmag=d_logmag.numpy(),
        dt_mask_frame=dt_mask_frame.numpy(),
    )

    fig = plt.figure(figsize=(15, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(pred_logmag.numpy().T, aspect="auto", origin="lower")
    ax1.set_title("log1p(|S_hat|)")
    ax1.set_xlabel("Time frame")
    ax1.set_ylabel("Freq bin")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(target_logmag.numpy().T, aspect="auto", origin="lower")
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
        "formulation": "crm_ri_complex_mask",
        "feature_mode": "crm_ri",
        "pesq": None if pesq_val is None else float(pesq_val),
        "si_sdr": None if si_sdr_val is None else float(si_sdr_val),
        "erle_reference": float(erle_ref),
        "sample_rate": int(cfg["sample_rate"]),
        "signal_length": int(len(d_np)),
        "far_path": meta.get("far_path"),
        "near_path": meta.get("near_path"),
        "far_activity_ratio": meta.get("far_activity_ratio"),
        "near_activity_ratio": meta.get("near_activity_ratio"),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    print("CRM inference finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()