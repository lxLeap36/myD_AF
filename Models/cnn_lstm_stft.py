import torch
import torch.nn as nn


class CNNLSTMSTFT(nn.Module):
    """
    Generic minimal STFT-domain CNN-LSTM model.

    Input:
        x: [B, C, T, F]

    Output:
        y: [B, T, out_dim]
    """

    def __init__(
        self,
        num_freq_bins: int,
        lstm_hidden: int = 128,
        in_channels: int = 2,
        out_dim: int = None,
        head_hidden: int = 256,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.num_freq_bins = int(num_freq_bins)
        self.lstm_hidden = int(lstm_hidden)
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim) if out_dim is not None else int(num_freq_bins)
        self.head_hidden = int(head_hidden)
        self.output_activation = output_activation

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=32 * self.num_freq_bins,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(self.lstm_hidden, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, self.out_dim),
        )

        if output_activation == "softplus":
            self.out_act = nn.Softplus()
        elif output_activation in ("identity", None):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]

        Returns:
            y: [B, T, out_dim]
        """
        feat = self.cnn(x)                               # [B, 32, T, F]
        b, c, t, f = feat.shape

        feat = feat.permute(0, 2, 1, 3).contiguous()    # [B, T, 32, F]
        feat = feat.view(b, t, c * f)                   # [B, T, 32*F]

        lstm_out, _ = self.lstm(feat)                   # [B, T, H]
        y = self.head(lstm_out)                         # [B, T, out_dim]
        y = self.out_act(y)

        return y