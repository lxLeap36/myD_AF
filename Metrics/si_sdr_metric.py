from __future__ import annotations

from typing import Union

import torch
from torch import Tensor
from torchmetrics.functional.audio import (scale_invariant_signal_distortion_ratio, )

AudioLike = Union[Tensor, list, tuple]

def _to_audio_tensor(x: AudioLike) -> Tensor:
    """Convert input to float tensor."""
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _match_shape(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Crop to the same last-dimension length if needed."""
    min_len = min(preds.shape[-1], target.shape[-1])
    preds = preds[..., :min_len]
    target = target[..., :min_len]
    return preds, target


def compute_si_sdr(
    clean: AudioLike,
    enhanced: AudioLike,
    zero_mean: bool = True,
) -> Tensor:
    """
    计算 SI-SDR（Scale-Invariant Signal-to-Distortion Ratio）分数。
    功能概述:
    - 接受参考干净语音（clean）和增强/降质语音（enhanced），返回每条语音的 SI-SDR 得分。
    - 支持批量输入：输入可为 1D (time,) 或 N-D (..., time)。若为 1D，会在前面新增 batch 维度以便统一处理。
    - 内部使用 torchmetrics.functional.audio.scale_invariant_signal_distortion_ratio 计算实际分数。
    参数:
    clean: 参考干净语音，可以是 Tensor 或 list/tuple，可带 batch 或其它前置维度，最后一维为时间 (..., time)。
    enhanced: 增强/降质语音，同样格式，应与 clean 在 batch/其它前置维度上可广播或对齐（函数会在时间轴上裁剪到相同长度）。
    zero_mean: 是否在计算前将信号去均值（zero-mean）。SI-SDR 通常建议去均值以消除 DC 分量对结果的影响。
    返回:
    一个 torch.Tensor：
    - 若输入为批量（即前面有 batch 维），返回形状为 (...,) 对应每条语音的 SI-SDR 分数。
    - 若输入原本为 1D（单条语音），返回标量张量（squeezed）。
    返回张量的 dtype 与 scale_invariant_signal_distortion_ratio 的输出一致（通常为 float）。

    """
    clean = _to_audio_tensor(clean)
    enhanced = _to_audio_tensor(enhanced)

    if clean.ndim == 1:
        clean = clean.unsqueeze(0)
    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)

    clean, enhanced = _match_shape(enhanced, clean)

    scores = scale_invariant_signal_distortion_ratio(
        preds=enhanced,
        target=clean,
        zero_mean=zero_mean,
    )

    return scores.squeeze()