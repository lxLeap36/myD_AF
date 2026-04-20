"""double_talk 场景下的 STFT-domain CNN-LSTM 近端恢复模型
V2-1 先做简单的，只估计幅度，忽略相位
而且，先不对应传统算法的接口
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


def find_checkpoint(cfg):
    ckpt = cfg["inference"]["checkpoint_path"]
    if ckpt is not None:
        return ckpt
    return os.path.join(cfg["output_dir"], "checkpoints", "best_model.pt")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    out_dir = os.path.join(cfg["output_dir"], "inference")
    ensure_dir(out_dir)

    ckpt_path = find_checkpoint(cfg) # 找到要加载的best_model.pt路径，或者是cfg里指定的路径
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu") # 指定 map_location="cpu" 会强制将所有张量加载到 CPU 内存中
    num_freq_bins = checkpoint["num_freq_bins"]
    # checkpoint可以理解为 模型快照 或 存档点，从best_model.pt中加载的 checkpoint 包含了模型在训练过程中最好的状态信息，
    #                                                                                       通常是在验证集上表现最好的模型参数。
    # 它包含了模型的状态字典（model_state_dict）以及其他相关信息（如 "model_state_dict", "config", "best_val_loss", "epoch" 等）。
    # 通过加载 checkpoint，我们可以恢复模型的训练状态或者进行推理。

    print(f"Loaded checkpoint from {ckpt_path}, num_freq_bins={num_freq_bins}")
    model = CNNLSTMSTFT(
        num_freq_bins=num_freq_bins,
        lstm_hidden=cfg["model"]["lstm_hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    # 从 checkpoint 中提取模型的状态字典（model_state_dict），并将其加载到 model 实例中。这一步会将 checkpoint 中保存的
    #                               模型参数（权重和偏置）赋值给 model 实例，使其恢复到 checkpoint 时的状态，模型具备了推理能力。
    model.to(device)
    model.eval()

    print(f"Building dataset and loading sample index {cfg['inference']['sample_index']}...")
    dataset = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=max(1, cfg["inference"]["sample_index"] + 1),
        split="val",
    )

    # 使用同一条 raw sample 来生成特征和评估目标，避免不一致
    raw_sample = dataset.get_raw_sample(cfg["inference"]["sample_index"])
    input_feat, target_mag, meta = dataset.sample_to_example(raw_sample)

    x = torch.tensor(raw_sample["x"], dtype=torch.float32)
    d = torch.tensor(raw_sample["d"], dtype=torch.float32)
    s = torch.tensor(raw_sample["s"], dtype=torch.float32)

    input_feat_b = input_feat.unsqueeze(0).to(device)   # [1,2,T,F]

    with torch.no_grad():
        pred_logmag = model(input_feat_b)[0].cpu()      # [T,F]

    # 还原 STFT 所需参数
    stft_cfg = cfg["stft"]
    n_fft = stft_cfg["n_fft"]
    hop_length = stft_cfg["hop_length"]
    win_length = stft_cfg["win_length"]
    window = torch.hann_window(win_length)

    # 原始 D 的复谱，用其相位恢复
    D = stft_complex(d, n_fft, hop_length, win_length, window)   # [F,T], complex
    D_phase = torch.angle(D)                                     # [F,T]

    # pred_logmag: [T,F] -> [F,T]
    pred_logmag_ft = pred_logmag.transpose(0, 1).contiguous()

    # log1p(|S_hat|) -> |S_hat|
    pred_mag = torch.expm1(pred_logmag_ft)
    pred_mag = torch.clamp(pred_mag, min=0.0)

    # 与 D 相位组合
    S_hat_complex = mag_phase_to_complex(pred_mag, D_phase)

    # iSTFT -> s_hat
    s_hat = istft_complex(
        S_hat_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(d),
    ).cpu().numpy()

    s_np = s.cpu().numpy()
    d_np = d.cpu().numpy()
    x_np = x.cpu().numpy()

    # 评估：PESQ / SI-SDR 为主
    pesq_val = compute_pesq(s_np, s_hat, fs=cfg["sample_rate"])
    si_sdr_val = compute_si_sdr(s_np, s_hat)

    # ERLE 仅作为参考：
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

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        x=x_np,
        d=d_np,
        s=s_np,
        s_hat=s_hat,
        pred_logmag=pred_logmag.numpy(),
        target_logmag=target_mag.numpy(),
    )

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(target_mag.numpy().T, aspect="auto", origin="lower")
    ax1.set_title("Target log1p(|S|)")
    ax1.set_ylabel("Freq bin")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(pred_logmag.numpy().T, aspect="auto", origin="lower")
    ax2.set_title("Predicted log1p(|S_hat|)")
    ax2.set_ylabel("Freq bin")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(input_feat[0].numpy().T, aspect="auto", origin="lower")
    ax3.set_title("Input channel 0: log1p(|D|)")
    ax3.set_xlabel("Time frame")
    ax3.set_ylabel("Freq bin")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectrogram_compare.png"), dpi=150)
    plt.close(fig)

    summary = {
        "checkpoint_path": ckpt_path,
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

    print("Inference finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()