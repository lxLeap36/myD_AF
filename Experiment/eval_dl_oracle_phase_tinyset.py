# -*- coding: utf-8 -*-
"""
实验 2：oracle phase 测试
---------------------------------
建议保存为：
    Experiment/eval_dl_oracle_phase_tinyset.py

用途：
1. 重建和实验1相同的固定 tiny set
2. 加载实验1训练好的 overfit 模型
3. 比较三种重建：
   (a) pred_mag + angle(D)        -> 当前推理方式
   (b) pred_mag + angle(S)        -> oracle phase
   (c) true|S|  + angle(D)        -> 诊断“相位本身是不是瓶颈”

4. 输出：
   - 每条样本的 SI-SDR / PESQ
   - 汇总平均值
   - 每条样本的谱图对比图
   - 可选保存 wav

运行示例：
    python Experiment/eval_dl_oracle_phase_tinyset.py
"""

import json
import math
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# ===== repo import path =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Tools.set_seed import set_seed

# 可选：保存 wav
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False

# 可选：PESQ
try:
    from pesq import pesq as pesq_api
    HAS_PESQ = True
except Exception:
    HAS_PESQ = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stft_complex_local(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """
    输入:
        x: [N]
    输出:
        X: [F, T] complex
    """
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def istft_complex_local(
    X: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """
    输入:
        X: [F, T] complex
    输出:
        x: [N]
    """
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )


def si_sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """
    zero-mean SI-SDR, 单位 dB
    ref, est: [N]
    """
    ref = ref.float()
    est = est.float()

    ref = ref - ref.mean()
    est = est - est.mean()

    ref_energy = torch.sum(ref * ref) + eps
    proj = torch.sum(est * ref) * ref / ref_energy
    noise = est - proj

    ratio = (torch.sum(proj * proj) + eps) / (torch.sum(noise * noise) + eps)
    return float(10.0 * torch.log10(ratio + eps).item())


def safe_pesq(fs: int, ref: np.ndarray, deg: np.ndarray) -> Optional[float]:
    """
    容错 PESQ。若包不存在或计算失败，返回 None。
    """
    if not HAS_PESQ:
        return None

    try:
        # wideband 模式适合 16k
        score = pesq_api(fs, ref.astype(np.float32), deg.astype(np.float32), "wb")
        return float(score)
    except Exception:
        return None


def to_numpy_audio(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def mag_phase_to_complex(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    mag:   [F, T], nonnegative
    phase: [F, T], rad
    return complex [F, T]
    """
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.complex(real, imag)


def tensor_logmag_from_complex(X: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.abs(X))


class FixedTinyRawDataset(Dataset):
    """
    和实验1一致：一次性固定 tiny set，不再在线重采样。
    但这里缓存的是 raw sample，便于拿到 x/d/s 做 oracle-phase 分析。
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        num_samples: int,
        split: str = "train",
    ):
        super().__init__()
        self.base_cfg = deepcopy(base_cfg)
        self.num_samples = int(num_samples)
        self.split = split

        self.source_dataset = DoubleTalkSTFTDataset(
            base_cfg=self.base_cfg,
            num_samples=self.num_samples,
            split=split,
        )

        self.cached_raw_samples: List[Dict[str, Any]] = []
        self._build_and_cache()

    def _build_and_cache(self):
        print(f"Building fixed raw tiny dataset with {self.num_samples} samples ...")
        for idx in range(self.num_samples):
            sample = self.source_dataset.build_valid_sample(idx)
            self.cached_raw_samples.append(sample)

            extra_meta = sample.get("meta", {}).get("extra", {})
            far_meta = extra_meta.get("far_meta", {}) or {}
            near_meta = extra_meta.get("near_meta", {}) or {}

            print(
                f"[cache {idx:02d}] "
                f"far={far_meta.get('file_path')} | "
                f"near={near_meta.get('file_path')}"
            )
        print("Fixed raw tiny dataset ready.\n")

    def __len__(self):
        return len(self.cached_raw_samples)

    def __getitem__(self, idx: int):
        return deepcopy(self.cached_raw_samples[idx])


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> CNNLSTMSTFT:
    ckpt = torch.load(ckpt_path, map_location=device)
    num_freq_bins = int(ckpt["num_freq_bins"])

    cfg = ckpt.get("config", get_config())
    lstm_hidden = int(cfg["model"]["lstm_hidden"])

    model = CNNLSTMSTFT(
        num_freq_bins=num_freq_bins,
        lstm_hidden=lstm_hidden,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_one_sample(
    model: torch.nn.Module,
    raw_sample: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    fs = int(cfg["fs"])
    stft_cfg = cfg["stft"]
    n_fft = int(stft_cfg["n_fft"])
    hop_length = int(stft_cfg["hop_length"])
    win_length = int(stft_cfg["win_length"])

    window = torch.hann_window(win_length)

    x = torch.tensor(raw_sample["x"], dtype=torch.float32)
    d = torch.tensor(raw_sample["d"], dtype=torch.float32)
    s = torch.tensor(raw_sample["s"], dtype=torch.float32)

    # STFT
    X = stft_complex_local(x, n_fft, hop_length, win_length, window)  # [F,T]
    D = stft_complex_local(d, n_fft, hop_length, win_length, window)
    S = stft_complex_local(s, n_fft, hop_length, win_length, window)

    # 当前模型输入
    D_logmag_TF = torch.log1p(torch.abs(D)).transpose(0, 1).contiguous()   # [T,F]
    X_logmag_TF = torch.log1p(torch.abs(X)).transpose(0, 1).contiguous()   # [T,F]
    input_feat = torch.stack([D_logmag_TF, X_logmag_TF], dim=0).unsqueeze(0)  # [1,2,T,F]

    pred_logmag_TF = model(input_feat.to(device))[0].cpu()   # [T,F]
    pred_mag = torch.expm1(torch.clamp(pred_logmag_TF, min=0.0)).transpose(0, 1).contiguous()  # [F,T]

    true_mag = torch.abs(S)   # [F,T]

    phase_D = torch.angle(D)
    phase_S = torch.angle(S)

    # 三种重建
    pred_with_D_phase = mag_phase_to_complex(pred_mag, phase_D)
    pred_with_S_phase = mag_phase_to_complex(pred_mag, phase_S)
    true_mag_with_D_phase = mag_phase_to_complex(true_mag, phase_D)

    # iSTFT
    s_hat_D = istft_complex_local(
        pred_with_D_phase, n_fft, hop_length, win_length, window, length=len(s)
    )
    s_hat_Soracle = istft_complex_local(
        pred_with_S_phase, n_fft, hop_length, win_length, window, length=len(s)
    )
    s_hat_trueMag_Dphase = istft_complex_local(
        true_mag_with_D_phase, n_fft, hop_length, win_length, window, length=len(s)
    )

    # 指标
    si_sdr_pred_D = si_sdr(s, s_hat_D)
    si_sdr_pred_Soracle = si_sdr(s, s_hat_Soracle)
    si_sdr_trueMag_Dphase = si_sdr(s, s_hat_trueMag_Dphase)

    s_np = to_numpy_audio(s)
    s_hat_D_np = to_numpy_audio(s_hat_D)
    s_hat_Soracle_np = to_numpy_audio(s_hat_Soracle)
    s_hat_trueMag_Dphase_np = to_numpy_audio(s_hat_trueMag_Dphase)

    pesq_pred_D = safe_pesq(fs, s_np, s_hat_D_np)
    pesq_pred_Soracle = safe_pesq(fs, s_np, s_hat_Soracle_np)
    pesq_trueMag_Dphase = safe_pesq(fs, s_np, s_hat_trueMag_Dphase_np)

    # 元信息
    extra_meta = raw_sample.get("meta", {}).get("extra", {})
    far_meta = extra_meta.get("far_meta", {}) or {}
    near_meta = extra_meta.get("near_meta", {}) or {}

    meta = {
        "length": int(len(s)),
        "far_path": far_meta.get("file_path"),
        "near_path": near_meta.get("file_path"),
        "far_activity_ratio": far_meta.get("activity_ratio"),
        "near_activity_ratio": near_meta.get("activity_ratio"),
    }

    return {
        "meta": meta,
        "fs": fs,
        "x": x,
        "d": d,
        "s": s,
        "X": X,
        "D": D,
        "S": S,
        "pred_logmag_TF": pred_logmag_TF,  # [T,F]
        "true_logmag_TF": torch.log1p(torch.abs(S)).transpose(0, 1).contiguous(),
        "recon_logmag_Dphase_TF": tensor_logmag_from_complex(pred_with_D_phase).transpose(0, 1).contiguous(),
        "recon_logmag_Soracle_TF": tensor_logmag_from_complex(pred_with_S_phase).transpose(0, 1).contiguous(),
        "recon_logmag_trueMag_Dphase_TF": tensor_logmag_from_complex(true_mag_with_D_phase).transpose(0, 1).contiguous(),
        "s_hat_D": s_hat_D,
        "s_hat_Soracle": s_hat_Soracle,
        "s_hat_trueMag_Dphase": s_hat_trueMag_Dphase,
        "metrics": {
            "si_sdr_pred_Dphase": si_sdr_pred_D,
            "si_sdr_pred_Soracle": si_sdr_pred_Soracle,
            "si_sdr_trueMag_Dphase": si_sdr_trueMag_Dphase,
            "pesq_pred_Dphase": pesq_pred_D,
            "pesq_pred_Soracle": pesq_pred_Soracle,
            "pesq_trueMag_Dphase": pesq_trueMag_Dphase,
        },
    }


def save_audio_if_possible(
    result: Dict[str, Any],
    sample_idx: int,
    save_dir: str,
):
    if not HAS_SF:
        return

    fs = int(result["fs"])
    ensure_dir(save_dir)

    sf.write(os.path.join(save_dir, f"sample_{sample_idx:02d}_clean_s.wav"), to_numpy_audio(result["s"]), fs)
    sf.write(os.path.join(save_dir, f"sample_{sample_idx:02d}_pred_Dphase.wav"), to_numpy_audio(result["s_hat_D"]), fs)
    sf.write(os.path.join(save_dir, f"sample_{sample_idx:02d}_pred_Soracle.wav"), to_numpy_audio(result["s_hat_Soracle"]), fs)
    sf.write(os.path.join(save_dir, f"sample_{sample_idx:02d}_trueMag_Dphase.wav"), to_numpy_audio(result["s_hat_trueMag_Dphase"]), fs)


def _plot_tf(ax, mat_TF: torch.Tensor, title: str):
    im = ax.imshow(
        mat_TF.T.numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Freq bin")
    return im


def save_result_figure(
    result: Dict[str, Any],
    sample_idx: int,
    save_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    true_logmag_TF = result["true_logmag_TF"]
    pred_logmag_TF = result["pred_logmag_TF"]
    recon_logmag_Dphase_TF = result["recon_logmag_Dphase_TF"]
    recon_logmag_Soracle_TF = result["recon_logmag_Soracle_TF"]
    recon_logmag_trueMag_Dphase_TF = result["recon_logmag_trueMag_Dphase_TF"]

    err_pred = torch.abs(pred_logmag_TF - true_logmag_TF)
    err_recon_D = torch.abs(recon_logmag_Dphase_TF - true_logmag_TF)
    err_recon_Soracle = torch.abs(recon_logmag_Soracle_TF - true_logmag_TF)

    im0 = _plot_tf(axes[0, 0], true_logmag_TF, "Target: log1p(|S|)")
    im1 = _plot_tf(axes[0, 1], pred_logmag_TF, "Pred mag: log1p(|S_hat|)")
    im2 = _plot_tf(axes[0, 2], err_pred, "|PredMag - Target|")

    im3 = _plot_tf(axes[1, 0], recon_logmag_Dphase_TF, "Recon: pred_mag + phase(D)")
    im4 = _plot_tf(axes[1, 1], recon_logmag_Soracle_TF, "Recon: pred_mag + phase(S) [oracle]")
    im5 = _plot_tf(axes[1, 2], recon_logmag_trueMag_Dphase_TF, "Recon: true|S| + phase(D)")

    for im, ax in zip([im0, im1, im2, im3, im4, im5], axes.flatten()):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    metrics = result["metrics"]
    meta = result["meta"]
    far_name = os.path.basename(meta.get("far_path") or "unknown_far")
    near_name = os.path.basename(meta.get("near_path") or "unknown_near")

    fig.suptitle(
        f"sample_{sample_idx:02d}\n"
        f"far={far_name} | near={near_name}\n"
        f"SI-SDR: pred+D={metrics['si_sdr_pred_Dphase']:.2f} dB | "
        f"pred+Soracle={metrics['si_sdr_pred_Soracle']:.2f} dB | "
        f"trueMag+D={metrics['si_sdr_trueMag_Dphase']:.2f} dB",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def summarize_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean_of(key: str) -> Optional[float]:
        vals = [r["metrics"][key] for r in all_results if r["metrics"][key] is not None]
        if len(vals) == 0:
            return None
        return float(np.mean(vals))

    summary = {
        "mean_si_sdr_pred_Dphase": mean_of("si_sdr_pred_Dphase"),
        "mean_si_sdr_pred_Soracle": mean_of("si_sdr_pred_Soracle"),
        "mean_si_sdr_trueMag_Dphase": mean_of("si_sdr_trueMag_Dphase"),
        "mean_pesq_pred_Dphase": mean_of("pesq_pred_Dphase"),
        "mean_pesq_pred_Soracle": mean_of("pesq_pred_Soracle"),
        "mean_pesq_trueMag_Dphase": mean_of("pesq_trueMag_Dphase"),
    }

    # 额外给提升量
    if summary["mean_si_sdr_pred_Dphase"] is not None and summary["mean_si_sdr_pred_Soracle"] is not None:
        summary["mean_delta_si_sdr_oraclePhase_minus_Dphase"] = (
            summary["mean_si_sdr_pred_Soracle"] - summary["mean_si_sdr_pred_Dphase"]
        )
    else:
        summary["mean_delta_si_sdr_oraclePhase_minus_Dphase"] = None

    if summary["mean_pesq_pred_Dphase"] is not None and summary["mean_pesq_pred_Soracle"] is not None:
        summary["mean_delta_pesq_oraclePhase_minus_Dphase"] = (
            summary["mean_pesq_pred_Soracle"] - summary["mean_pesq_pred_Dphase"]
        )
    else:
        summary["mean_delta_pesq_oraclePhase_minus_Dphase"] = None

    return summary


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    # ========= 你主要改这里 =========
    tiny_num_samples = 12
    split = "train"

    ckpt_path = os.path.join(
        cfg["root_dir"],
        "Results",
        f"results_dl_overfit_tiny_{tiny_num_samples}",
        "checkpoints",
        "best_overfit_model.pt",
    )

    save_audio = True
    # =================================

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = os.path.join(
        cfg["root_dir"],
        "Results",
        f"results_dl_oracle_phase_tiny_{tiny_num_samples}",
    )
    fig_dir = os.path.join(output_dir, "figures")
    wav_dir = os.path.join(output_dir, "wav")
    ensure_dir(output_dir)
    ensure_dir(fig_dir)
    if save_audio:
        ensure_dir(wav_dir)

    print(f"Device: {device}")
    print(f"Load checkpoint: {ckpt_path}")

    model = build_model_from_ckpt(ckpt_path, device)

    tiny_raw_set = FixedTinyRawDataset(
        base_cfg=cfg,
        num_samples=tiny_num_samples,
        split=split,
    )

    all_results: List[Dict[str, Any]] = []
    per_sample_metrics: Dict[str, Any] = {}

    for idx in range(len(tiny_raw_set)):
        raw_sample = tiny_raw_set[idx]
        result = run_one_sample(model, raw_sample, cfg, device)
        all_results.append(result)

        if save_audio:
            save_audio_if_possible(result, idx, wav_dir)

        save_result_figure(
            result,
            idx,
            os.path.join(fig_dir, f"sample_{idx:02d}.png"),
        )

        per_sample_metrics[f"sample_{idx:02d}"] = {
            "meta": result["meta"],
            **result["metrics"],
        }

        print(
            f"[sample {idx:02d}] "
            f"SI-SDR pred+D={result['metrics']['si_sdr_pred_Dphase']:.3f} dB | "
            f"pred+Soracle={result['metrics']['si_sdr_pred_Soracle']:.3f} dB | "
            f"trueMag+D={result['metrics']['si_sdr_trueMag_Dphase']:.3f} dB"
        )

    summary = summarize_metrics(all_results)
    summary["tiny_num_samples"] = tiny_num_samples
    summary["checkpoint"] = ckpt_path
    summary["has_pesq"] = HAS_PESQ
    summary["has_soundfile"] = HAS_SF

    save_json(per_sample_metrics, os.path.join(output_dir, "per_sample_metrics.json"))
    save_json(summary, os.path.join(output_dir, "summary.json"))

    print("\n===== Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nFigures saved to: {fig_dir}")
    if save_audio and HAS_SF:
        print(f"Wavs saved to: {wav_dir}")


if __name__ == "__main__":
    main()