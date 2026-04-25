import torch
import torch.nn as nn


class SpectralL1Loss(nn.Module):
    """
    L1 loss on predicted / target log-magnitude spectrograms.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, T, F]
            target: [B, T, F]
        """
        return self.loss_fn(pred, target)

class WeightedSpectralL1Loss(nn.Module):
    """
    Weighted L1 loss on log-magnitude spectrograms.
    让 target 能量较大的时频点权重更高，减少均值解塌缩。
    """

    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, T, F]
        """
        # target 越大，权重越高
        # 这里权重范围大致是 [1, 1+alpha]
        # 将每个样本的 target 除以其自身在全部时间‑频率位置上的最大值。能量越大的时频点，target_norm 越接近 1；能量极小的点则接近 0。
        target_norm = target / (target.amax(dim=(1, 2), keepdim=True) + 1e-8)
        weight = 1.0 + self.alpha * target_norm

        loss = torch.abs(pred - target) * weight
        return loss.mean()

class DTMaskedWeightedSpectralL1Loss(nn.Module):
    """
    训练时利用 double-talk mask 的加权谱损失。

    改进点：
    1) 仍然逐 (B,T,F) 格子计算绝对误差
    2) 仍然保留 target 能量加权 + dt_mask 时间加权
    3) 但最后不用简单 mean，而改成“按总权重归一化”的加权平均
       这样 loss 数值不会随着 dt/non-dt 占比和权重绝对值变化而乱漂
    """

    def __init__(
        self,
        alpha: float = 4.0,
        dt_weight: float = 4.0,
        non_dt_weight: float = 0.25,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.dt_weight = float(dt_weight)
        self.non_dt_weight = float(non_dt_weight)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred   : [B, T, F]
        target : [B, T, F]
        dt_mask: [B, T]
        """
        # 每个样本内部按 target 最大值做归一化
        target_norm = target / (target.amax(dim=(1, 2), keepdim=True) + self.eps)
        spectral_weight = 1.0 + self.alpha * target_norm          # [B,T,F]

        # dt 帧与 non-dt 帧的时间权重
        time_weight = self.non_dt_weight + (self.dt_weight - self.non_dt_weight) * dt_mask
        time_weight = time_weight.unsqueeze(-1)                   # [B,T,1]

        # 总权重
        weight = spectral_weight * time_weight                    # [B,T,F]

        abs_err = torch.abs(pred - target)                       # [B,T,F]
        weighted_loss = abs_err * weight

        # 关键：用权重和归一化，而不是直接 mean
        loss = weighted_loss.sum() / (weight.sum() + self.eps)
        return loss

class DTMaskedComplexRIL1Loss(nn.Module):
    """
    CRM 训练用的复谱 RI 加权 L1 损失。

    pred_ri   : [B, 2, T, F]
    target_ri : [B, 2, T, F]
    dt_mask   : [B, T]

    设计思路：
    1) 在实部/虚部上做 L1
    2) 权重参考 target 的复谱幅度 |S|
    3) 继续使用 dt_mask 的时间加权
    4) 最后按总权重归一化，避免比例漂移
    """

    def __init__(
        self,
        alpha: float = 4.0,
        dt_weight: float = 4.0,
        non_dt_weight: float = 0.25,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.dt_weight = float(dt_weight)
        self.non_dt_weight = float(non_dt_weight)
        self.eps = float(eps)

    def forward(
        self,
        pred_ri: torch.Tensor,
        target_ri: torch.Tensor,
        dt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_ri   : [B, 2, T, F]
        target_ri : [B, 2, T, F]
        dt_mask   : [B, T]
        """
        if pred_ri.shape != target_ri.shape:
            raise ValueError(f"Shape mismatch: {pred_ri.shape} vs {target_ri.shape}")

        if pred_ri.ndim != 4 or pred_ri.shape[1] != 2:
            raise ValueError("pred_ri / target_ri must be [B, 2, T, F]")

        # 目标复谱幅度 |S|
        target_mag = torch.sqrt(target_ri[:, 0] ** 2 + target_ri[:, 1] ** 2 + self.eps)  # [B,T,F]

        # 频谱能量权重
        target_norm = target_mag / (target_mag.amax(dim=(1, 2), keepdim=True) + self.eps)
        spectral_weight = 1.0 + self.alpha * target_norm   # [B,T,F]

        # dt / non-dt 时间权重
        time_weight = self.non_dt_weight + (self.dt_weight - self.non_dt_weight) * dt_mask
        time_weight = time_weight.unsqueeze(-1)            # [B,T,1]

        weight = spectral_weight * time_weight             # [B,T,F]
        weight_ri = weight.unsqueeze(1).expand_as(pred_ri) # [B,2,T,F]

        abs_err = torch.abs(pred_ri - target_ri)
        weighted_loss = abs_err * weight_ri

        loss = weighted_loss.sum() / (weight_ri.sum() + self.eps)
        return loss