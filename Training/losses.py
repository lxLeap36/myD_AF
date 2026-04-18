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