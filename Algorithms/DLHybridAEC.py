# -*- coding: utf-8 -*-
"""
DLHybridAEC wrapper for V2-4A.

作用：
    把 V2-3 hybrid 深度模型包装成和 LMS/NLMS/RLS 一样的算法接口。

平台统一调用：
    e = algo.process(x, d)

这里约定：
    e = s_hat

也就是：
    深度模型直接输出近端估计 s_hat，
    在 AEC 平台里把它当作 error / AEC output。

同时派生：
    y_hat = d - s_hat

注意：
    这个 wrapper 不估计回声路径 h_hat，
    所以 weights = None，path comparison 会跳过 DL 方法。
"""

from pathlib import Path
import sys
import numpy as np
import torch

# 保证从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Models.cnn_lstm_stft import HybridMagComplexNet
from Training.audio_features import (
    stft_complex,
    complex_to_ri_channels,
    apply_real_mask_to_ri,
    ri_channels_to_complex,
    istft_complex,
)


def _to_numpy_1d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {x.shape}")
    return x.astype(np.float32)


def _unpack_residual_ri(pred: torch.Tensor, num_freq_bins: int) -> torch.Tensor:
    """
    pred: [B, T, 2F]
    return: [B, 2, T, F]
    """
    b, t, two_f = pred.shape
    expected = 2 * num_freq_bins
    if two_f != expected:
        raise ValueError(f"Expected last dim = {expected}, got {two_f}")

    pred = pred.view(b, t, 2, num_freq_bins)          # [B, T, 2, F]
    pred = pred.permute(0, 2, 1, 3).contiguous()      # [B, 2, T, F]
    return pred


def _build_hybrid_outputs(
    mag_mask: torch.Tensor,
    res_ri_flat: torch.Tensor,
    ri_feat: torch.Tensor,
    *,
    beta_residual: float,
    eps: float = 1e-8,
):
    """
    和 train_dl_hybrid.py / infer_dl_hybrid.py 中的 build_outputs 保持一致。

    Args:
        mag_mask   : [B, T, F]
        res_ri_flat: [B, T, 2F]
        ri_feat    : [B, 4, T, F] = [D_r, D_i, X_r, X_i]

    Returns:
        pred_s_ri  : [B, 2, T, F]
    """
    d_ri = ri_feat[:, 0:2, :, :]  # [B, 2, T, F]
    d_mag = torch.sqrt(d_ri[:, 0] ** 2 + d_ri[:, 1] ** 2 + eps)  # [B, T, F]

    base_s_ri = apply_real_mask_to_ri(mag_mask, d_ri)  # [B, 2, T, F]

    res_raw_ri = _unpack_residual_ri(res_ri_flat, mag_mask.shape[-1])
    delta_ri = beta_residual * torch.tanh(res_raw_ri) * d_mag.unsqueeze(1)

    pred_s_ri = base_s_ri + delta_ri
    return pred_s_ri


class DLHybridAEC:
    """
    把 V2-3 HybridMagComplexNet 封装成传统平台算法接口。

    process(x, d) 返回：
        e = s_hat

    派生属性：
        self.estimated_echo = d - s_hat
        self.last_output = s_hat

    为了兼容 run_basic.py：
        self.weights = None
        self.weight_history = []
    """

    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        stft_cfg=None,
        beta_residual=None,
        strict=True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.strict = strict

        if self.device_name == "cuda" and not torch.cuda.is_available():
            print("[DLHybridAEC] CUDA requested but not available. Falling back to CPU.")
            self.device_name = "cpu"

        self.device = torch.device(self.device_name)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DL checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        self.checkpoint = checkpoint

        ckpt_cfg = checkpoint.get("config", {})
        ckpt_stft = ckpt_cfg.get("stft", {})

        if stft_cfg is None:
            stft_cfg = ckpt_stft

        if not stft_cfg:
            raise ValueError(
                "stft_cfg is empty. Please provide n_fft / hop_length / win_length "
                "in config_basic.py -> alg_params['dl_hybrid']['stft']."
            )

        self.n_fft = int(stft_cfg["n_fft"])
        self.hop_length = int(stft_cfg["hop_length"])
        self.win_length = int(stft_cfg["win_length"])

        self.num_freq_bins = int(checkpoint["num_freq_bins"])

        model_cfg = ckpt_cfg.get("model", {})
        lstm_hidden = int(model_cfg.get("lstm_hidden", 128))

        stem_channels = int(checkpoint.get("stem_channels", 16))
        trunk_channels = int(checkpoint.get("trunk_channels", 32))
        head_hidden = int(checkpoint.get("head_hidden", 256))
        mag_output_activation = checkpoint.get("mag_output_activation", "softplus")
        res_output_activation = checkpoint.get("res_output_activation", "identity")

        self.beta_residual = (
            float(beta_residual)
            if beta_residual is not None
            else float(checkpoint.get("beta_residual", 0.30))
        )

        self.model = HybridMagComplexNet(
            num_freq_bins=self.num_freq_bins,
            lstm_hidden=lstm_hidden,
            stem_channels=stem_channels,
            trunk_channels=trunk_channels,
            head_hidden=head_hidden,
            mag_output_activation=mag_output_activation,
            res_output_activation=res_output_activation,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=self.strict)
        self.model.to(self.device)
        self.model.eval()

        self.window = torch.hann_window(self.win_length, device=self.device)

        # 兼容传统平台字段
        self.weights = None
        self.weight_history = []

        # 推理后记录
        self.last_output = None          # s_hat
        self.estimated_echo = None       # y_hat = d - s_hat

        print("[DLHybridAEC] Loaded hybrid model:")
        print(f"  checkpoint : {self.checkpoint_path}")
        print(f"  device     : {self.device}")
        print(f"  n_fft      : {self.n_fft}")
        print(f"  hop_length : {self.hop_length}")
        print(f"  win_length : {self.win_length}")
        print(f"  F          : {self.num_freq_bins}")
        print(f"  beta       : {self.beta_residual}")

    def reset(self):
        """
        传统 LMS/NLMS/RLS 每次样本前需要 reset。
        深度模型推理没有在线权值状态，这里只清空上一次输出。
        """
        self.last_output = None
        self.estimated_echo = None
        self.weight_history = []

    def process(self, x, d):
        """
        Args:
            x: far-end signal, [N]
            d: microphone signal, [N]

        Returns:
            e: AEC output / error signal.
               对 DL hybrid 来说，e = s_hat。
        """
        x_np = _to_numpy_1d(x)
        d_np = _to_numpy_1d(d)

        n = min(len(x_np), len(d_np))
        if len(x_np) != len(d_np):
            print(
                f"[DLHybridAEC] Warning: len(x)={len(x_np)}, len(d)={len(d_np)}. "
                f"Cropping both to {n}."
            )
            x_np = x_np[:n]
            d_np = d_np[:n]

        x_t = torch.from_numpy(x_np).float().to(self.device)
        d_t = torch.from_numpy(d_np).float().to(self.device)

        with torch.no_grad():
            # STFT: [F, T], complex
            x_spec = stft_complex(
                x_t,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
            )
            d_spec = stft_complex(
                d_t,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
            )

            if x_spec.shape[0] != self.num_freq_bins:
                raise ValueError(
                    f"STFT freq bins mismatch: checkpoint F={self.num_freq_bins}, "
                    f"current F={x_spec.shape[0]}. Please check n_fft."
                )

            # mag_feat: [1, 2, T, F] = [log1p(|D|), log1p(|X|)]
            d_logmag = torch.log1p(torch.abs(d_spec)).transpose(0, 1).contiguous()
            x_logmag = torch.log1p(torch.abs(x_spec)).transpose(0, 1).contiguous()
            mag_feat = torch.stack([d_logmag, x_logmag], dim=0).unsqueeze(0)

            # ri_feat: [1, 4, T, F] = [D_r, D_i, X_r, X_i]
            d_ri = complex_to_ri_channels(d_spec)  # [2, T, F]
            x_ri = complex_to_ri_channels(x_spec)  # [2, T, F]
            ri_feat = torch.cat([d_ri, x_ri], dim=0).unsqueeze(0)

            mag_feat = mag_feat.to(self.device)
            ri_feat = ri_feat.to(self.device)

            mag_mask, res_ri_flat = self.model(mag_feat, ri_feat)

            pred_s_ri = _build_hybrid_outputs(
                mag_mask,
                res_ri_flat,
                ri_feat,
                beta_residual=self.beta_residual,
            )  # [1, 2, T, F]

            pred_spec = ri_channels_to_complex(pred_s_ri[0])  # [F, T], complex

            s_hat_t = istft_complex(
                pred_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=n,
            )

        s_hat = s_hat_t.detach().cpu().numpy().astype(np.float32)

        # 平台统一含义：
        #   e_hat = s_hat
        #   y_hat = d - s_hat
        self.last_output = s_hat
        self.estimated_echo = (d_np - s_hat).astype(np.float32)

        return s_hat