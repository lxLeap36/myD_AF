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