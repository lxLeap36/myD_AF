import copy
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Experiment.utils_basic import build_scenario
from Training.audio_features import stft_complex


class DoubleTalkSTFTDataset(Dataset):
    """
    在线生成 double_talk 场景样本，并转换成最小 STFT-domain 训练样本。

    输出:
        input_feat   : [2, T, F]
            channel 0 = log1p(|D|)
            channel 1 = log1p(|X|)

        target_mag   : [T, F]
            log1p(|S|)

        dt_mask_frame: [T]
            double-talk 帧级 soft mask，范围大致在 [0, 1]

        meta_dict    : 额外信息（可选调试）
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        num_samples: int,
        split: str = "train",
        max_build_retries: int = 20,
    ):
        super().__init__()
        self.base_cfg = copy.deepcopy(base_cfg)
        self.num_samples = int(num_samples)
        self.split = split
        self.max_build_retries = int(max_build_retries)

        stft_cfg = self.base_cfg["stft"]
        self.n_fft = int(stft_cfg["n_fft"])
        self.hop_length = int(stft_cfg["hop_length"])
        self.win_length = int(stft_cfg["win_length"])

        self.window = torch.hann_window(self.win_length)

        # 让 train / val 至少种子区间不同
        self.seed_offset = 0 if split == "train" else 100000

    def __len__(self):
        return self.num_samples

    def _build_one_sample(self, idx: int) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.base_cfg)

        # 用不同 seed 在线采样
        cfg["seed"] = int(cfg.get("seed", 42)) + self.seed_offset + idx

        # 强制使用 double_talk
        cfg["scenario_name"] = "double_talk"

        return build_scenario(cfg)

    def build_valid_sample(self, idx: int) -> Dict[str, Any]:
        """
        允许在构造样本失败时自动换 seed 重试，
        避免单个 seed 让 train / infer 直接崩掉。
        """
        last_error = None

        for retry in range(self.max_build_retries):
            try_idx = idx + retry * 1000003
            try:
                return self._build_one_sample(try_idx)
            except RuntimeError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"Failed to build a valid double_talk sample after {self.max_build_retries} retries. "
            f"Last error: {last_error}"
        )

    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        return self.build_valid_sample(idx)

    def _time_mask_to_frame_mask(self, mask_1d: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        把时域 [N] mask 映射成帧级 [T] soft mask。

        做法：
        - 按 STFT 的 center=True 近似做左右 padding
        - 用 win_length 窗口、hop_length 步长对 mask 做 unfold
        - 每帧取窗口平均值，得到 [0, 1] 的 soft mask
        """
        mask_1d = mask_1d.float()
        if mask_1d.ndim != 1:
            raise ValueError("mask_1d must be 1-D")

        pad = self.n_fft // 2
        x = mask_1d.unsqueeze(0).unsqueeze(0)  # [1,1,N]
        x = F.pad(x, (pad, pad))

        needed_len = (num_frames - 1) * self.hop_length + self.win_length
        cur_len = x.shape[-1]
        if cur_len < needed_len:
            x = F.pad(x, (0, needed_len - cur_len))

        frames = x.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        # [1,1,T,win]
        frame_mask = frames[0, 0, :num_frames].mean(dim=-1)  # [T]
        frame_mask = torch.clamp(frame_mask, 0.0, 1.0)
        return frame_mask

    def sample_to_example(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = torch.tensor(sample["x"], dtype=torch.float32)   # far-end
        d = torch.tensor(sample["d"], dtype=torch.float32)   # mic
        s = torch.tensor(sample["s"], dtype=torch.float32)   # clean near-end

        X = stft_complex(x, self.n_fft, self.hop_length, self.win_length, self.window)
        D = stft_complex(d, self.n_fft, self.hop_length, self.win_length, self.window)
        S = stft_complex(s, self.n_fft, self.hop_length, self.win_length, self.window)

        X_mag = torch.log1p(torch.abs(X))   # [F, T]
        D_mag = torch.log1p(torch.abs(D))   # [F, T]
        S_mag = torch.log1p(torch.abs(S))   # [F, T]

        X_mag = X_mag.transpose(0, 1).contiguous()   # [T, F]
        D_mag = D_mag.transpose(0, 1).contiguous()   # [T, F]
        S_mag = S_mag.transpose(0, 1).contiguous()   # [T, F]

        input_feat = torch.stack([D_mag, X_mag], dim=0)  # [2, T, F]

        # 取 double-talk 时域 mask -> 帧级 soft mask
        masks = sample.get("masks", {})
        dt_mask_time = torch.tensor(
            masks.get("double_talk_mask", torch.zeros(len(d))),
            dtype=torch.float32,
        )
        dt_mask_frame = self._time_mask_to_frame_mask(
            dt_mask_time,
            num_frames=S_mag.shape[0],
        )  # [T]

        extra_meta = sample.get("meta", {}).get("extra", {})
        far_meta = extra_meta.get("far_meta", {}) or {}
        near_meta = extra_meta.get("near_meta", {}) or {}

        meta = {
            "length": int(len(d)),
            "far_path": far_meta.get("file_path"),
            "near_path": near_meta.get("file_path"),
            "far_activity_ratio": far_meta.get("activity_ratio"),
            "near_activity_ratio": near_meta.get("activity_ratio"),
        }

        return input_feat, S_mag, dt_mask_frame, meta

    def __getitem__(self, idx: int):
        sample = self.build_valid_sample(idx)
        return self.sample_to_example(sample)