# -*- coding: utf-8 -*-
"""
长时长推理实验：
1) 3s full
2) 6s full
3) 9s full
4) 9s chunked (3s + 3s + 3s)

面向当前 hybrid 模型：
    best_model_hybrid.pt

输出目录：
    Results/results_dl_hybrid/longform_eval/

建议运行：
    python -u eval_longform_hybrid.py
"""

import csv
import json
import math
import os
import sys
import time
from typing import Dict, Tuple, List

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
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.audio_features import (
    stft_complex,
    complex_to_ri_channels,
    ri_channels_to_complex,
    istft_complex,
    apply_real_mask_to_ri,
)
from Metrics.pesq_metric import compute_pesq
from Metrics.si_sdr_metric import compute_si_sdr
from Tools.set_seed import set_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def unpack_residual_ri(pred: torch.Tensor, num_freq_bins: int) -> torch.Tensor:
    """
    pred: [B, T, 2F]
    return: [B, 2, T, F]
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
        ri_feat    : [B, 4, T, F] = [D_r, D_i, X_r, X_i]

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


def build_hybrid_features_from_waveforms(
    x_np: np.ndarray,
    d_np: np.ndarray,
    s_np: np.ndarray,
    cfg: Dict,
):
    """
    从原始时域波形直接构造 hybrid 所需特征：
        mag_feat: [2, T, F]
        ri_feat : [4, T, F]
        target_ri: [2, T, F]
    """
    stft_cfg = cfg["stft"]

    x_t = torch.tensor(x_np, dtype=torch.float32)
    d_t = torch.tensor(d_np, dtype=torch.float32)
    s_t = torch.tensor(s_np, dtype=torch.float32)

    window = torch.hann_window(stft_cfg["win_length"])

    X = stft_complex(
        x_t,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
    )  # [F, T], complex

    D = stft_complex(
        d_t,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
    )  # [F, T], complex

    S = stft_complex(
        s_t,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
    )  # [F, T], complex

    X_logmag = torch.log1p(torch.abs(X)).transpose(0, 1).contiguous()   # [T, F]
    D_logmag = torch.log1p(torch.abs(D)).transpose(0, 1).contiguous()   # [T, F]

    X_ri = complex_to_ri_channels(X)   # [2, T, F]
    D_ri = complex_to_ri_channels(D)   # [2, T, F]
    S_ri = complex_to_ri_channels(S)   # [2, T, F]

    mag_feat = torch.stack([D_logmag, X_logmag], dim=0)   # [2, T, F]
    ri_feat = torch.cat([D_ri, X_ri], dim=0)              # [4, T, F]

    return mag_feat, ri_feat, S_ri, X, D, S


def run_hybrid_infer_on_waveforms(
    model: torch.nn.Module,
    x_np: np.ndarray,
    d_np: np.ndarray,
    s_np: np.ndarray,
    cfg: Dict,
    device: torch.device,
    *,
    beta_residual: float,
    tag: str,
):
    """
    对一段任意长度语音做一次 hybrid 推理。
    """
    mag_feat, ri_feat, target_ri, X, D, S = build_hybrid_features_from_waveforms(
        x_np, d_np, s_np, cfg
    )

    mag_feat_b = mag_feat.unsqueeze(0).to(device)   # [1, 2, T, F]
    ri_feat_b = ri_feat.unsqueeze(0).to(device)     # [1, 4, T, F]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        mag_mask_b, res_ri_flat_b = model(mag_feat_b, ri_feat_b)
        base_s_ri_b, delta_ri_b, pred_s_ri_b, d_mag_b = build_outputs(
            mag_mask_b,
            res_ri_flat_b,
            ri_feat_b,
            beta_residual=beta_residual,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    infer_time_sec = time.perf_counter() - t0

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    pred_s_ri = pred_s_ri_b[0].cpu()       # [2, T, F]
    base_s_ri = base_s_ri_b[0].cpu()
    delta_ri = delta_ri_b[0].cpu()
    d_mag = d_mag_b[0].cpu()

    pred_spec = ri_channels_to_complex(pred_s_ri)   # [F, T], complex
    base_spec = ri_channels_to_complex(base_s_ri)
    delta_spec = ri_channels_to_complex(delta_ri)

    stft_cfg = cfg["stft"]
    window = torch.hann_window(stft_cfg["win_length"])

    s_hat_t = istft_complex(
        pred_spec,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
        length=len(d_np),
    ).cpu()

    s_base_t = istft_complex(
        base_spec,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
        length=len(d_np),
    ).cpu()

    delta_t = istft_complex(
        delta_spec,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
        length=len(d_np),
    ).cpu()

    s_hat = s_hat_t.numpy()
    s_base = s_base_t.numpy()
    delta_wav = delta_t.numpy()

    pesq_val = compute_pesq(s_np, s_hat, fs=cfg["sample_rate"])
    si_sdr_val = compute_si_sdr(s_np, s_hat)

    residual = d_np - s_hat
    ref_echo = d_np - s_np
    erle_ref = 10.0 * np.log10(
        (np.sum(ref_echo ** 2) + 1e-12) / (np.sum(residual ** 2) + 1e-12)
    )

    pred_logmag = torch.log1p(torch.abs(pred_spec)).transpose(0, 1).contiguous().numpy()  # [T, F]
    target_logmag = torch.log1p(torch.abs(S)).transpose(0, 1).contiguous().numpy()
    d_logmag = torch.log1p(torch.abs(D)).transpose(0, 1).contiguous().numpy()

    return {
        "tag": tag,
        "x": x_np,
        "d": d_np,
        "s": s_np,
        "s_hat": s_hat,
        "s_base": s_base,
        "delta_wav": delta_wav,
        "pred_logmag": pred_logmag,
        "target_logmag": target_logmag,
        "d_logmag": d_logmag,
        "pesq": None if pesq_val is None else float(pesq_val),
        "si_sdr": None if si_sdr_val is None else float(si_sdr_val),
        "erle_reference": float(erle_ref),
        "infer_time_sec": float(infer_time_sec),
        "peak_mem_mb": None if peak_mem_mb is None else float(peak_mem_mb),
    }


def save_case_outputs(case_dir: str, result: Dict, sample_rate: int):
    ensure_dir(case_dir)

    sf.write(os.path.join(case_dir, "far_end_x.wav"), result["x"], sample_rate)
    sf.write(os.path.join(case_dir, "mic_d.wav"), result["d"], sample_rate)
    sf.write(os.path.join(case_dir, "clean_near_s.wav"), result["s"], sample_rate)
    sf.write(os.path.join(case_dir, "pred_near_s_hat.wav"), result["s_hat"], sample_rate)
    sf.write(os.path.join(case_dir, "pred_base_s.wav"), result["s_base"], sample_rate)
    sf.write(os.path.join(case_dir, "pred_delta.wav"), result["delta_wav"], sample_rate)

    np.savez(
        os.path.join(case_dir, "arrays.npz"),
        x=result["x"],
        d=result["d"],
        s=result["s"],
        s_hat=result["s_hat"],
        s_base=result["s_base"],
        delta_wav=result["delta_wav"],
        pred_logmag=result["pred_logmag"],
        target_logmag=result["target_logmag"],
        d_logmag=result["d_logmag"],
    )

    fig = plt.figure(figsize=(15, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(result["pred_logmag"].T, aspect="auto", origin="lower")
    ax1.set_title(f"{result['tag']}: log1p(|S_hat|)")
    ax1.set_xlabel("Time frame")
    ax1.set_ylabel("Freq bin")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(result["target_logmag"].T, aspect="auto", origin="lower")
    ax2.set_title("log1p(|S|)")
    ax2.set_xlabel("Time frame")
    ax2.set_ylabel("Freq bin")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(result["d_logmag"].T, aspect="auto", origin="lower")
    ax3.set_title("log1p(|D|)")
    ax3.set_xlabel("Time frame")
    ax3.set_ylabel("Freq bin")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, "spectrogram_compare.png"), dpi=150)
    plt.close(fig)

    summary = {
        "tag": result["tag"],
        "pesq": result["pesq"],
        "si_sdr": result["si_sdr"],
        "erle_reference": result["erle_reference"],
        "infer_time_sec": result["infer_time_sec"],
        "peak_mem_mb": result["peak_mem_mb"],
        "signal_length": int(len(result["d"])),
        "duration_sec": float(len(result["d"]) / sample_rate),
    }
    save_json(summary, os.path.join(case_dir, "summary.json"))


def save_chunked_9s_plot(case_dir: str, result: Dict, sample_rate: int, boundary_sec_list: List[float]):
    """
    对 chunked 9s 多画一个带边界的图，方便看 3s / 6s 处是否有拼接痕迹。
    """
    ensure_dir(case_dir)

    stft_cfg = get_config()["stft"]
    hop_length = stft_cfg["hop_length"]

    fig = plt.figure(figsize=(15, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(result["pred_logmag"].T, aspect="auto", origin="lower")
    ax1.set_title(f"{result['tag']}: log1p(|S_hat|)")
    ax1.set_xlabel("Time frame")
    ax1.set_ylabel("Freq bin")

    for bsec in boundary_sec_list:
        x_frame = int(round((bsec * sample_rate) / hop_length))
        ax1.axvline(x_frame, color="w", linestyle="--", linewidth=1)

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(result["target_logmag"].T, aspect="auto", origin="lower")
    ax2.set_title("log1p(|S|)")
    ax2.set_xlabel("Time frame")
    ax2.set_ylabel("Freq bin")
    for bsec in boundary_sec_list:
        x_frame = int(round((bsec * sample_rate) / hop_length))
        ax2.axvline(x_frame, color="w", linestyle="--", linewidth=1)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(result["d_logmag"].T, aspect="auto", origin="lower")
    ax3.set_title("log1p(|D|)")
    ax3.set_xlabel("Time frame")
    ax3.set_ylabel("Freq bin")
    for bsec in boundary_sec_list:
        x_frame = int(round((bsec * sample_rate) / hop_length))
        ax3.axvline(x_frame, color="w", linestyle="--", linewidth=1)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, "spectrogram_compare_with_chunk_boundaries.png"), dpi=150)
    plt.close(fig)


def slice_prefix(arr: np.ndarray, fs: int, duration_sec: float) -> np.ndarray:
    length = int(round(fs * duration_sec))
    return arr[:length].copy()


def main():
    import copy

    cfg = get_config()
    set_seed(cfg["seed"])

    hybrid_output_dir = os.path.join(cfg["root_dir"], "Results", "results_dl_hybrid")
    ckpt_path = os.path.join(hybrid_output_dir, "checkpoints", "best_model_hybrid.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 本地配置为主，避免把服务器路径带进来
    cfg_used = copy.deepcopy(cfg)

    # 但以下这些数值型配置，最好和训练时保持一致
    train_cfg = checkpoint.get("config", {})
    if "sample_rate" in train_cfg:
        cfg_used["sample_rate"] = train_cfg["sample_rate"]
    if "stft" in train_cfg:
        cfg_used["stft"] = train_cfg["stft"]
    if "model" in train_cfg:
        cfg_used["model"] = train_cfg["model"]

    device_name = cfg_used["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    num_freq_bins = checkpoint["num_freq_bins"]
    stem_channels = checkpoint.get("stem_channels", 16)
    trunk_channels = checkpoint.get("trunk_channels", 32)
    head_hidden = checkpoint.get("head_hidden", 256)
    mag_output_activation = checkpoint.get("mag_output_activation", "softplus")
    res_output_activation = checkpoint.get("res_output_activation", "identity")
    beta_residual = checkpoint.get("beta_residual", 0.30)

    model = HybridMagComplexNet(
        num_freq_bins=num_freq_bins,
        lstm_hidden=cfg_used["model"]["lstm_hidden"],
        stem_channels=stem_channels,
        trunk_channels=trunk_channels,
        head_hidden=head_hidden,
        mag_output_activation=mag_output_activation,
        res_output_activation=res_output_activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # =============================
    # 先固定生成 1 条 9s 样本
    # =============================
    fs = cfg_used["sample_rate"]
    sample_index = cfg_used["inference"]["sample_index"]

    cfg_9 = json.loads(json.dumps(cfg_used))
    cfg_9["duration_sec"] = 9.0

    dataset_9 = DoubleTalkSTFTDataset(
        base_cfg=cfg_9,
        num_samples=max(1, sample_index + 1),
        split="val",
        feature_mode="hybrid_mag_ri",
    )
    raw9 = dataset_9.get_raw_sample(sample_index)

    x9 = np.asarray(raw9["x"], dtype=np.float32)
    d9 = np.asarray(raw9["d"], dtype=np.float32)
    s9 = np.asarray(raw9["s"], dtype=np.float32)

    out_root = os.path.join(hybrid_output_dir, "longform_eval")
    ensure_dir(out_root)

    # =============================
    # 实验 1：3s full
    # =============================
    x3 = slice_prefix(x9, fs, 3.0)
    d3 = slice_prefix(d9, fs, 3.0)
    s3 = slice_prefix(s9, fs, 3.0)

    res_3_full = run_hybrid_infer_on_waveforms(
        model, x3, d3, s3, cfg_used, device,
        beta_residual=beta_residual,
        tag="3s_full",
    )
    save_case_outputs(os.path.join(out_root, "exp1_3s_full"), res_3_full, fs)

    # =============================
    # 实验 2：6s full
    # =============================
    x6 = slice_prefix(x9, fs, 6.0)
    d6 = slice_prefix(d9, fs, 6.0)
    s6 = slice_prefix(s9, fs, 6.0)

    res_6_full = run_hybrid_infer_on_waveforms(
        model, x6, d6, s6, cfg_used, device,
        beta_residual=beta_residual,
        tag="6s_full",
    )
    save_case_outputs(os.path.join(out_root, "exp2_6s_full"), res_6_full, fs)

    # =============================
    # 实验 3：9s full
    # =============================
    res_9_full = run_hybrid_infer_on_waveforms(
        model, x9, d9, s9, cfg_used, device,
        beta_residual=beta_residual,
        tag="9s_full",
    )
    save_case_outputs(os.path.join(out_root, "exp3_9s_full"), res_9_full, fs)

    # =============================
    # 实验 4：9s chunked by 3s
    # =============================
    chunk_sec = 3.0
    chunk_len = int(round(chunk_sec * fs))
    num_chunks = int(math.ceil(len(d9) / chunk_len))

    chunk_results = []
    total_chunk_time = 0.0
    peak_mem_chunk_mb = None

    s_hat_chunks = []
    s_base_chunks = []
    delta_chunks = []

    for i in range(num_chunks):
        st = i * chunk_len
        ed = min((i + 1) * chunk_len, len(d9))

        xi = x9[st:ed].copy()
        di = d9[st:ed].copy()
        si = s9[st:ed].copy()

        res_i = run_hybrid_infer_on_waveforms(
            model, xi, di, si, cfg_used, device,
            beta_residual=beta_residual,
            tag=f"9s_chunked_chunk{i+1}",
        )
        chunk_results.append(res_i)
        total_chunk_time += res_i["infer_time_sec"]

        if res_i["peak_mem_mb"] is not None:
            if peak_mem_chunk_mb is None:
                peak_mem_chunk_mb = res_i["peak_mem_mb"]
            else:
                peak_mem_chunk_mb = max(peak_mem_chunk_mb, res_i["peak_mem_mb"])

        s_hat_chunks.append(res_i["s_hat"])
        s_base_chunks.append(res_i["s_base"])
        delta_chunks.append(res_i["delta_wav"])

    s_hat_chunked = np.concatenate(s_hat_chunks, axis=0)
    s_base_chunked = np.concatenate(s_base_chunks, axis=0)
    delta_chunked = np.concatenate(delta_chunks, axis=0)

    pesq_chunked = compute_pesq(s9, s_hat_chunked, fs=fs)
    si_sdr_chunked = compute_si_sdr(s9, s_hat_chunked)
    residual_chunked = d9 - s_hat_chunked
    ref_echo_chunked = d9 - s9
    erle_chunked = 10.0 * np.log10(
        (np.sum(ref_echo_chunked ** 2) + 1e-12) /
        (np.sum(residual_chunked ** 2) + 1e-12)
    )

    # 为了统一画图，再从完整波形重算一遍 STFT 图
    _, _, _, _, D_chunked_spec, S_chunked_spec = build_hybrid_features_from_waveforms(
        x9, d9, s9, cfg_used
    )
    window_chunked = torch.hann_window(cfg_used["stft"]["win_length"])

    pred_chunked_spec = stft_complex(
        torch.tensor(s_hat_chunked, dtype=torch.float32),
        n_fft=cfg_used["stft"]["n_fft"],
        hop_length=cfg_used["stft"]["hop_length"],
        win_length=cfg_used["stft"]["win_length"],
        window=window_chunked,
    )

    res_9_chunked = {
        "tag": "9s_chunked_3s",
        "x": x9,
        "d": d9,
        "s": s9,
        "s_hat": s_hat_chunked,
        "s_base": s_base_chunked,
        "delta_wav": delta_chunked,
        "pred_logmag": torch.log1p(torch.abs(pred_chunked_spec)).transpose(0, 1).contiguous().numpy(),
        "target_logmag": torch.log1p(torch.abs(S_chunked_spec)).transpose(0, 1).contiguous().numpy(),
        "d_logmag": torch.log1p(torch.abs(D_chunked_spec)).transpose(0, 1).contiguous().numpy(),
        "pesq": None if pesq_chunked is None else float(pesq_chunked),
        "si_sdr": None if si_sdr_chunked is None else float(si_sdr_chunked),
        "erle_reference": float(erle_chunked),
        "infer_time_sec": float(total_chunk_time),
        "peak_mem_mb": None if peak_mem_chunk_mb is None else float(peak_mem_chunk_mb),
        "num_chunks": int(num_chunks),
        "chunk_sec": float(chunk_sec),
    }
    save_case_outputs(os.path.join(out_root, "exp4_9s_chunked_3s"), res_9_chunked, fs)
    save_chunked_9s_plot(
        os.path.join(out_root, "exp4_9s_chunked_3s"),
        res_9_chunked,
        fs,
        boundary_sec_list=[3.0, 6.0],
    )

    # =============================
    # 汇总
    # =============================
    all_summary = [
        {
            "experiment": "exp1_3s_full",
            "pesq": res_3_full["pesq"],
            "si_sdr": res_3_full["si_sdr"],
            "erle_reference": res_3_full["erle_reference"],
            "infer_time_sec": res_3_full["infer_time_sec"],
            "peak_mem_mb": res_3_full["peak_mem_mb"],
        },
        {
            "experiment": "exp2_6s_full",
            "pesq": res_6_full["pesq"],
            "si_sdr": res_6_full["si_sdr"],
            "erle_reference": res_6_full["erle_reference"],
            "infer_time_sec": res_6_full["infer_time_sec"],
            "peak_mem_mb": res_6_full["peak_mem_mb"],
        },
        {
            "experiment": "exp3_9s_full",
            "pesq": res_9_full["pesq"],
            "si_sdr": res_9_full["si_sdr"],
            "erle_reference": res_9_full["erle_reference"],
            "infer_time_sec": res_9_full["infer_time_sec"],
            "peak_mem_mb": res_9_full["peak_mem_mb"],
        },
        {
            "experiment": "exp4_9s_chunked_3s",
            "pesq": res_9_chunked["pesq"],
            "si_sdr": res_9_chunked["si_sdr"],
            "erle_reference": res_9_chunked["erle_reference"],
            "infer_time_sec": res_9_chunked["infer_time_sec"],
            "peak_mem_mb": res_9_chunked["peak_mem_mb"],
            "num_chunks": res_9_chunked["num_chunks"],
            "chunk_sec": res_9_chunked["chunk_sec"],
        },
    ]

    save_json(
        {
            "checkpoint_path": ckpt_path,
            "sample_index": int(sample_index),
            "far_path": raw9.get("far_path", None),
            "near_path": raw9.get("near_path", None),
            "results": all_summary,
        },
        os.path.join(out_root, "summary_all.json"),
    )

    csv_path = os.path.join(out_root, "summary_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "pesq",
                "si_sdr",
                "erle_reference",
                "infer_time_sec",
                "peak_mem_mb",
                "num_chunks",
                "chunk_sec",
            ],
        )
        writer.writeheader()
        for row in all_summary:
            writer.writerow(row)

    print("\n===== Longform evaluation summary =====")
    for row in all_summary:
        print(row)

    print(f"\nSaved to: {out_root}")


if __name__ == "__main__":
    main()