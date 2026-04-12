from __future__ import annotations

from typing import Union

import torch
from torch import Tensor
from torchmetrics.functional.audio.pesq import (perceptual_evaluation_speech_quality,)

AudioLike = Union[Tensor, list, tuple]

def _to_audio_tensor(x: AudioLike) -> Tensor:
    """Convert input to float tensor."""
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _match_shape(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Crop裁剪 to the same last-dimension length if needed."""
    min_len = min(preds.shape[-1], target.shape[-1])
    preds = preds[..., :min_len]
    target = target[..., :min_len]
    return preds, target


def compute_pesq(
    clean: AudioLike,
    enhanced: AudioLike,
    fs: int,
    mode: str | None = None,
    keep_same_device: bool = True,
    n_processes: int = 1,
) -> Tensor:
    """
    计算 PESQ（Perceptual Evaluation of Speech Quality）分数。

    功能概述:
    - 接受参考干净语音（clean）和增强/降质语音（enhanced），返回每条语音的 PESQ 得分。
    - 支持批量输入：输入可为 1D (time,) 或 N-D (..., time)。若为 1D，会在前面新增 batch 维度以便统一处理。
    - 内部使用 torchmetrics.functional.audio.pesq.perceptual_evaluation_speech_quality 计算实际分数。

    参数:
        clean: 参考干净语音，可以是 Tensor 或 list/tuple，可带 batch 或其它前置维度，最后一维为时间 (..., time)。
        enhanced: 增强/降质语音，同样格式，应与 clean 在 batch/其它前置维度上可广播或对齐（函数会在时间轴上裁剪到相同长度）。
        fs: 采样率，必须是 8000 或 16000。
        mode: 'nb'（窄带）或 'wb'（宽带）。若为 None，则根据 fs 自动推断：8000 -> 'nb'，16000 -> 'wb'。
        keep_same_device: 是否在返回结果时将输出移动回 preds（enhanced）的 device。当使用多进程或计算发生在 CPU/GPU 切换时该选项有用。
        n_processes: 并行进程数，用于批量 PESQ 计算。若为 1，则不并行。

    返回:
        一个 torch.Tensor：
        - 若输入为批量（即前面有 batch 维），返回形状为 (...,) 对应每条语音的 PESQ 分数。
        - 若输入原本为 1D（单条语音），返回标量张量（squeezed）。
        返回张量的 dtype 与 perceptual_evaluation_speech_quality 的输出一致（通常为 float）。

    异常:
        - 若 fs 不在 (8000, 16000) 范围内，会抛出 ValueError。
        - 若 mode 非 'nb' 或 'wb'，会抛出 ValueError。

    使用示例（要点）:
    - 单条语音:
        clean = torch.randn(16000)
        enh = torch.randn(16000)
        score = compute_pesq(clean, enh, fs=16000)  # 返回标量张量
    - 批量语音:
        clean = torch.randn(4, 16000)
        enh = torch.randn(4, 16000)
        scores = compute_pesq(clean, enh, fs=16000)  # 返回形状 (4,)

    """
    clean = _to_audio_tensor(clean)
    enhanced = _to_audio_tensor(enhanced)

    if clean.ndim == 1:
        clean = clean.unsqueeze(0)  # 在前面新增一个维度，使其成为形状 (1, time)，表示单条语音的批量形式。
    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)

    clean, enhanced = _match_shape(clean, enhanced)

    if fs not in (8000, 16000):
        raise ValueError("PESQ only supports fs=8000 or fs=16000.")
    if mode is None:
        mode = "nb" if fs == 8000 else "wb"
    if mode not in ("nb", "wb"):
        raise ValueError("mode must be 'nb' or 'wb'.")

    scores = perceptual_evaluation_speech_quality(
        preds=enhanced,
        target=clean,
        fs=fs,
        mode=mode,
        keep_same_device=keep_same_device,
        n_processes=n_processes,
    )

    return scores.squeeze()  # 删除所有长度为 1 的维度， 例如：形状 (1, 3, 1, 4) → 形状 (3, 4)。