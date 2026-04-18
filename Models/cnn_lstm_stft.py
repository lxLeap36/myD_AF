import torch
import torch.nn as nn


class CNNLSTMSTFT(nn.Module):
    """
    Minimal STFT-domain CNN-LSTM model.

    Input:
        x: [B, 2, T, F]
            channel 0 -> log1p(|D|)
            channel 1 -> log1p(|X|)

    Output:
        y: [B, T, F]
            predicted log1p(|S_hat|)
    """

    def __init__(self, num_freq_bins: int, lstm_hidden: int = 128):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.lstm_hidden = lstm_hidden

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=32 * num_freq_bins,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(lstm_hidden, num_freq_bins)

        # 保证输出非负，对应 log1p(|S_hat|)
        self.out_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, T, F]

        Returns:
            y: [B, T, F]
        """
        feat = self.cnn(x)                 # [B, 32, T, F]
        b, c, t, f = feat.shape

        feat = feat.permute(0, 2, 1, 3).contiguous()   # [B, T, 32, F]
        feat = feat.view(b, t, c * f)                  # [B, T, 32*F]

        lstm_out, _ = self.lstm(feat)                  # [B, T, H]
        y = self.fc(lstm_out)                          # [B, T, F]
        y = self.out_act(y)                            # non-negative

        return y