"""
double_talk 场景下的 STFT-domain CNN-LSTM 近端恢复模型（MASK 版推理）
V2-1: 模型输出解释为 mask M，预测幅度由 |S_hat| = M * |D| 得到
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
    stft_complex,
    istft_complex,
    mag_phase_to_complex,
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


def find_checkpoint(cfg):
    ckpt = cfg["inference"]["checkpoint_path"]
    if ckpt is not None:
        return ckpt
    return os.path.join(cfg["output_dir"], "checkpoints", "best_model_mask.pt")


def compute_logmag_tf_from_waveform(
    wav_tensor: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
):
    """
    输入:
        wav_tensor: [N]
    输出:
        logmag_tf: [T, F]
        complex_spec: [F, T]
    """
    spec = stft_complex(wav_tensor, n_fft, hop_length, win_length, window)   # [F,T]
    logmag_tf = torch.log1p(torch.abs(spec)).transpose(0, 1).contiguous()     # [T,F]
    return logmag_tf, spec


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    # mask 版训练输出目录
    out_dir = os.path.join(cfg["output_dir"], "inference_mask")
    ensure_dir(out_dir)

    ckpt_path = find_checkpoint(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    num_freq_bins = checkpoint["num_freq_bins"]
    max_mask_value = checkpoint.get("max_mask_value", 5.0)

    print(f"Loaded checkpoint from {ckpt_path}, num_freq_bins={num_freq_bins}, max_mask_value={max_mask_value}")

    model = CNNLSTMSTFT(
        num_freq_bins=num_freq_bins,
        lstm_hidden=cfg["model"]["lstm_hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    sample_index = cfg["inference"]["sample_index"]
    print(f"Building dataset and loading sample index {sample_index}...")

    dataset = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=max(1, sample_index + 1),
        split="val",
    )

    # 同一条 raw sample 生成特征和评估目标，避免不一致
    raw_sample = dataset.get_raw_sample(sample_index)
    input_feat, target_logmag, dt_mask_frame, meta = dataset.sample_to_example(raw_sample)

    x = torch.tensor(raw_sample["x"], dtype=torch.float32)
    d = torch.tensor(raw_sample["d"], dtype=torch.float32)
    s = torch.tensor(raw_sample["s"], dtype=torch.float32)

    input_feat_b = input_feat.unsqueeze(0).to(device)   # [1,2,T,F]

    with torch.no_grad():
        # 模型输出解释为 mask M，形状 [T,F]
        pred_mask = model(input_feat_b)[0].cpu()

    # 与训练保持一致
    pred_mask = torch.clamp(pred_mask, max=max_mask_value)

    # STFT 参数
    stft_cfg = cfg["stft"]
    n_fft = stft_cfg["n_fft"]
    hop_length = stft_cfg["hop_length"]
    win_length = stft_cfg["win_length"]
    window = torch.hann_window(win_length)

    # 原始 D / S 的复谱
    D = stft_complex(d, n_fft, hop_length, win_length, window)   # [F,T]
    S = stft_complex(s, n_fft, hop_length, win_length, window)   # [F,T]

    D_phase = torch.angle(D)                                     # [F,T]
    S_phase = torch.angle(S)                                     # [F,T]

    # pred_mask: [T,F] -> [F,T]
    pred_mask_ft = pred_mask.transpose(0, 1).contiguous()        # [F,T]

    # 线性幅度域：|S_hat| = M * |D|
    D_mag = torch.abs(D)                                         # [F,T]
    pred_mag_lin = torch.clamp(pred_mask_ft * D_mag, min=0.0)    # [F,T]

    # 供可视化使用：log1p(|S_hat|)
    pred_logmag_ft = torch.log1p(pred_mag_lin)                   # [F,T]
    pred_logmag_tf = pred_logmag_ft.transpose(0, 1).contiguous() # [T,F]

    # 用 D 相位重建
    S_hat_Dphase_complex = mag_phase_to_complex(pred_mag_lin, D_phase)
    s_hat_Dphase_t = istft_complex(
        S_hat_Dphase_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu()

    # 用 S 相位重建（oracle phase）
    S_hat_Sphase_complex = mag_phase_to_complex(pred_mag_lin, S_phase)
    s_hat_Sphase_t = istft_complex(
        S_hat_Sphase_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu()

    # 默认主输出仍按 D 相位版本
    s_hat_t = s_hat_Dphase_t

    # 转 numpy
    s_np = s.cpu().numpy()
    d_np = d.cpu().numpy()
    x_np = x.cpu().numpy()
    s_hat_Dphase = s_hat_Dphase_t.numpy()
    s_hat_Sphase = s_hat_Sphase_t.numpy()
    s_hat = s_hat_t.numpy()

    # 指标
    pesq_val_Dphase = compute_pesq(s_np, s_hat_Dphase, fs=cfg["sample_rate"])
    si_sdr_val_Dphase = compute_si_sdr(s_np, s_hat_Dphase)

    pesq_val_Sphase = compute_pesq(s_np, s_hat_Sphase, fs=cfg["sample_rate"])
    si_sdr_val_Sphase = compute_si_sdr(s_np, s_hat_Sphase)

    # ERLE 仅作为参考（按 D 相位主输出）
    residual = d_np - s_hat
    ref_echo = d_np - s_np
    erle_ref = 10.0 * np.log10(
        (np.sum(ref_echo ** 2) + 1e-12) / (np.sum(residual ** 2) + 1e-12)
    )

    print("开始保存音频...")
    sf.write(os.path.join(out_dir, "far_end_x.wav"), x_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "mic_d.wav"), d_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "clean_near_s.wav"), s_np, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "pred_near_s_hat_Dphase.wav"), s_hat_Dphase, cfg["sample_rate"])
    sf.write(os.path.join(out_dir, "pred_near_s_hat_Sphase.wav"), s_hat_Sphase, cfg["sample_rate"])

    # 为 6 个子图准备时频图
    # 1) log1p(|S_hat|) —— 直接由 mask 重建的预测幅度
    logmag_Shat_tf = pred_logmag_tf                                     # [T,F]

    # 2) log1p(|S|) —— dataset 直接返回的 target
    logmag_S_tf = target_logmag                                         # [T,F]

    # 3) log1p(|D|)
    logmag_D_tf = input_feat[0]                                         # [T,F]

    # 4) 用 D 相位重建后的 S_hat_Dphase 的时频图
    logmag_Shat_Dphase_tf, _ = compute_logmag_tf_from_waveform(
        s_hat_Dphase_t, n_fft, hop_length, win_length, window
    )

    # 5) 用 S 相位重建后的 S_hat_Sphase 的时频图
    logmag_Shat_Sphase_tf, _ = compute_logmag_tf_from_waveform(
        s_hat_Sphase_t, n_fft, hop_length, win_length, window
    )

    # 6) 真正的 S 的时频图
    logmag_trueS_tf, _ = compute_logmag_tf_from_waveform(
        s, n_fft, hop_length, win_length, window
    )

    # 保存数组
    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        x=x_np,
        d=d_np,
        s=s_np,
        s_hat_Dphase=s_hat_Dphase,
        s_hat_Sphase=s_hat_Sphase,
        pred_mask=pred_mask.numpy(),                    # [T,F]
        pred_logmag=logmag_Shat_tf.numpy(),             # [T,F]
        target_logmag=logmag_S_tf.numpy(),              # [T,F]
        d_logmag=logmag_D_tf.numpy(),                   # [T,F]
        logmag_Shat_Dphase=logmag_Shat_Dphase_tf.numpy(),
        logmag_Shat_Sphase=logmag_Shat_Sphase_tf.numpy(),
        logmag_trueS=logmag_trueS_tf.numpy(),
        dt_mask_frame=dt_mask_frame.numpy(),
    )

    # 6 个时频图
    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(logmag_Shat_tf.numpy().T, aspect="auto", origin="lower")
    ax1.set_title("log1p(|S_hat|)")
    ax1.set_ylabel("Freq bin")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(logmag_S_tf.numpy().T, aspect="auto", origin="lower")
    ax2.set_title("log1p(|S|)")
    ax2.set_ylabel("Freq bin")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(logmag_D_tf.numpy().T, aspect="auto", origin="lower")
    ax3.set_title("log1p(|D|)")
    ax3.set_ylabel("Freq bin")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(logmag_Shat_Dphase_tf.numpy().T, aspect="auto", origin="lower")
    ax4.set_title("S_hat_Dphase")
    ax4.set_xlabel("Time frame")
    ax4.set_ylabel("Freq bin")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(logmag_Shat_Sphase_tf.numpy().T, aspect="auto", origin="lower")
    ax5.set_title("S_hat_Sphase")
    ax5.set_xlabel("Time frame")
    ax5.set_ylabel("Freq bin")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(logmag_trueS_tf.numpy().T, aspect="auto", origin="lower")
    ax6.set_title("True S")
    ax6.set_xlabel("Time frame")
    ax6.set_ylabel("Freq bin")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_6subplots.png"), dpi=150)
    plt.close(fig)

    summary = {
        "checkpoint_path": ckpt_path,
        "formulation": "mask_times_D_mag",
        "max_mask_value": float(max_mask_value),
        "pesq_Dphase": None if pesq_val_Dphase is None else float(pesq_val_Dphase),
        "si_sdr_Dphase": None if si_sdr_val_Dphase is None else float(si_sdr_val_Dphase),
        "pesq_Sphase": None if pesq_val_Sphase is None else float(pesq_val_Sphase),
        "si_sdr_Sphase": None if si_sdr_val_Sphase is None else float(si_sdr_val_Sphase),
        "erle_reference": float(erle_ref),
        "sample_rate": int(cfg["sample_rate"]),
        "signal_length": int(len(d_np)),
        "far_path": meta.get("far_path"),
        "near_path": meta.get("near_path"),
        "far_activity_ratio": meta.get("far_activity_ratio"),
        "near_activity_ratio": meta.get("near_activity_ratio"),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    print("Inference finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()