from __future__ import annotations
from typing import Union
import torch
from torch import Tensor

AudioLike = Union[Tensor, list, tuple]

def _to_audio_tensor(x: AudioLike) -> Tensor:
    """Convert input to float tensor."""
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)

def _match_shape(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Crop to the same last-dimension length if needed."""
    min_len = min(x.shape[-1], y.shape[-1])
    x = x[..., :min_len]
    y = y[..., :min_len]
    return x, y

def compute_erle(
    echo: AudioLike,
    residual: AudioLike,
    eps: float = 1e-8,
) -> Tensor:
    """
    计算整体 ERLE（Echo Return Loss Enhancement）。

    定义:
        ERLE = 10 * log10( sum(echo^2) / sum(residual^2) )
        ELRE用于单讲时，麦克风信号d就是回声信号echo，d-y 就是 实际回声信号-估计回声信号，即残余信号residual

    功能说明:
    - 计算参考回声信号与消除后残余信号总体能量比，结果以 dB 表示。
    - 支持单条信号 (time,) 或批量信号 (..., time)。若为 1D 输入，会在前面添加 batch 维以统一计算，返回时会 squeeze 掉多余的维度。

    参数:
        echo (AudioLike): 参考回声分量，形状 (..., time) 或 (time,)。
        residual (AudioLike): 回声消除后的残余信号，形状 (..., time) 或 (time,)。
        eps (float): 为避免除零或对数负无穷而加到能量上的小常数，默认 1e-8。

    返回:
        Tensor: 每条信号的 ERLE 值（以 dB 为单位）。
            - 若输入为批量返回形状 (...,)（与输入的 batch/前置维一致）。
            - 若输入为单条信号返回标量张量。

    注意事项:
    - 若某帧 residual 全为零，加 eps 后仍会得到非常大的 ERLE 值，请根据场景合理设置 eps 或在上游处理静音帧。
    - 结果可能为负值（当 residual 能量大于 echo 能量时）。
    """
    echo = _to_audio_tensor(echo)
    residual = _to_audio_tensor(residual)

    if echo.ndim == 1:
        echo = echo.unsqueeze(0)
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)

    echo, residual = _match_shape(echo, residual)

    echo_power = torch.sum(echo ** 2, dim=-1) + eps
    residual_power = torch.sum(residual ** 2, dim=-1) + eps
    erle = 10.0 * torch.log10(echo_power / residual_power)

    return erle.squeeze()

def compute_erle_curve(
    echo: AudioLike,
    residual: AudioLike,
    frame_size: int = 512,
    hop_size: int | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    逐帧计算 ERLE 曲线（frame-wise ERLE）。

    功能说明:
    - 将信号切分为帧（使用 torch.Tensor.unfold），对每帧分别计算平均能量并求 ERLE。
    - 返回每帧的 ERLE 值序列，可用于查看在时间轴上回声抑制效果的变化。

    参数:
        echo (AudioLike): 参考回声分量，形状 (..., time) 或 (time,)。
        residual (AudioLike): 回声消除后的残余信号，形状 (..., time) 或 (time,)。
        frame_size (int): 帧长（以采样点数为单位），用于计算每帧的平均能量，默认 512。
        hop_size (int | None): 帧移（步长）。如果为 None，则等于 frame_size（无重叠）。若希望重叠帧可设置小于 frame_size 的值。
        eps (float): 为避免除零或对数负无穷而加到每帧能量上的小常数，默认 1e-8。

    返回:
        Tensor: ERLE 曲线，形状为 (..., num_frames)，与输入的前置维一致，最后一维为帧数。
            - 若输入为单条信号，返回形状为 (num_frames,)（函数末尾会 squeeze 掉长度为1的 batch 维）。

    异常:
        - 若输入信号的时间长度小于 frame_size，则抛出 ValueError。

    实现细节:
    - 使用每帧的 mean(e^2) 作为能量估计（等价于帧内功率）。
    - hop_size 缺省时设为 frame_size，表示帧间无重叠。
    - 对每帧能量加 eps 以保障数值稳定性，随后计算 10*log10(echo_power / residual_power)。

    示例:
        echo.shape = (16000,), frame_size=400, hop_size=160 -> num_frames = floor((16000 - 400)/160)+1
    """
    echo = _to_audio_tensor(echo)
    residual = _to_audio_tensor(residual)

    if echo.ndim == 1:
        echo = echo.unsqueeze(0)
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)

    echo, residual = _match_shape(echo, residual)

    if hop_size is None:
        hop_size = frame_size
    if echo.shape[-1] < frame_size:
        raise ValueError("Signal length must be >= frame_size.")

    echo_frames = echo.unfold(-1, frame_size, hop_size)        # (..., num_frames, frame_size)
    residual_frames = residual.unfold(-1, frame_size, hop_size)

    echo_power = torch.mean(echo_frames ** 2, dim=-1) + eps
    residual_power = torch.mean(residual_frames ** 2, dim=-1) + eps

    erle_curve = 10.0 * torch.log10(echo_power / residual_power)
    return erle_curve.squeeze()