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


class HybridMagComplexNet(nn.Module):
    """
    Hybrid model:
        双输入 stem + 共享主干 + 双输出 head

    Inputs:
        mag_x: [B, 2, T, F]
            channel 0 = log1p(|D|)
            channel 1 = log1p(|X|)

        ri_x: [B, 4, T, F]
            channel 0 = D_r
            channel 1 = D_i
            channel 2 = X_r
            channel 3 = X_i

    Outputs:
        mag_mask: [B, T, F]
            幅度主分支输出的实数 mask

        res_ri_flat: [B, T, 2F]
            复数修正分支输出的 raw residual
            后续在 train / infer 脚本里 reshape 成 [B, 2, T, F]
    """

    def __init__(
        self,
        num_freq_bins: int,
        lstm_hidden: int = 128,
        stem_channels: int = 16,
        trunk_channels: int = 32,
        head_hidden: int = 256,
        mag_output_activation: str = "softplus",
        res_output_activation: str = "identity",
    ):
        super().__init__()
        self.num_freq_bins = int(num_freq_bins)
        self.lstm_hidden = int(lstm_hidden)
        self.stem_channels = int(stem_channels)
        self.trunk_channels = int(trunk_channels)
        self.head_hidden = int(head_hidden)
        self.mag_output_activation = mag_output_activation
        self.res_output_activation = res_output_activation

        # -------------------------
        # Dual input stems
        # -------------------------
        self.mag_stem = nn.Sequential(
            nn.Conv2d(2, self.stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.ri_stem = nn.Sequential(
            nn.Conv2d(4, self.stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Shared trunk
        # concat([mag_stem, ri_stem]) -> [B, 2*stem_channels, T, F]
        # -------------------------
        trunk_in_channels = 2 * self.stem_channels

        self.shared_cnn = nn.Sequential(
            nn.Conv2d(trunk_in_channels, self.trunk_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.trunk_channels, self.trunk_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=self.trunk_channels * self.num_freq_bins,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # -------------------------
        # Head A: magnitude mask
        # output: [B, T, F]
        # -------------------------
        self.head_mag = nn.Sequential(
            nn.Linear(self.lstm_hidden, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, self.num_freq_bins),
        )

        # -------------------------
        # Head B: complex residual
        # output: [B, T, 2F]
        # -------------------------
        self.head_res = nn.Sequential(
            nn.Linear(self.lstm_hidden, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, 2 * self.num_freq_bins),
        )

        if self.mag_output_activation == "softplus":
            self.mag_out_act = nn.Softplus()
        elif self.mag_output_activation in ("identity", None):
            self.mag_out_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported mag_output_activation: {self.mag_output_activation}")

        if self.res_output_activation == "softplus":
            self.res_out_act = nn.Softplus()
        elif self.res_output_activation in ("identity", None):
            self.res_out_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported res_output_activation: {self.res_output_activation}")

    def forward(
        self,
        mag_x: torch.Tensor,
        ri_x: torch.Tensor,
    ):
        """
        Args:
            mag_x: [B, 2, T, F]
            ri_x : [B, 4, T, F]

        Returns:
            mag_mask   : [B, T, F]
            res_ri_flat: [B, T, 2F]
        """
        if mag_x.ndim != 4 or mag_x.shape[1] != 2:
            raise ValueError(f"mag_x must be [B, 2, T, F], got {tuple(mag_x.shape)}")

        if ri_x.ndim != 4 or ri_x.shape[1] != 4:
            raise ValueError(f"ri_x must be [B, 4, T, F], got {tuple(ri_x.shape)}")

        if mag_x.shape[0] != ri_x.shape[0] or mag_x.shape[2] != ri_x.shape[2] or mag_x.shape[3] != ri_x.shape[3]:
            raise ValueError(f"mag_x and ri_x shape mismatch: {tuple(mag_x.shape)} vs {tuple(ri_x.shape)}")

        mag_feat = self.mag_stem(mag_x)                  # [B, stem, T, F]
        ri_feat = self.ri_stem(ri_x)                     # [B, stem, T, F]

        feat = torch.cat([mag_feat, ri_feat], dim=1)     # [B, 2*stem, T, F]
        feat = self.shared_cnn(feat)                     # [B, trunk, T, F]

        b, c, t, f = feat.shape
        feat = feat.permute(0, 2, 1, 3).contiguous()     # [B, T, trunk, F]
        feat = feat.view(b, t, c * f)                    # [B, T, trunk*F]

        lstm_out, _ = self.lstm(feat)                    # [B, T, H]

        mag_mask = self.head_mag(lstm_out)               # [B, T, F]
        mag_mask = self.mag_out_act(mag_mask)

        res_ri_flat = self.head_res(lstm_out)            # [B, T, 2F]
        res_ri_flat = self.res_out_act(res_ri_flat)

        return mag_mask, res_ri_flat