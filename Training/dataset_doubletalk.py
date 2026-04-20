import copy
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import Dataset

from Experiment.utils_basic import build_scenario
from Training.audio_features import stft_complex


class DoubleTalkSTFTDataset(Dataset):
    """
    在线生成 double_talk 场景样本，并转换成最小 STFT-domain 训练样本。

    输出:
        input_feat : [2, T, F]
            channel 0 = log1p(|D|)
            channel 1 = log1p(|X|)
        target_mag : [T, F]
            log1p(|S|)
        meta_dict   : 额外信息（可选调试）
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
        """
        给训练外流程（例如 infer）直接拿同一条 raw sample。
        """
        return self.build_valid_sample(idx)

    def sample_to_example(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = torch.tensor(sample["x"], dtype=torch.float32)   # far-end
        d = torch.tensor(sample["d"], dtype=torch.float32)   # mic
        s = torch.tensor(sample["s"], dtype=torch.float32)   # clean near-end

        # STFT: [F, T], complex
        X = stft_complex(x, self.n_fft, self.hop_length, self.win_length, self.window)
        D = stft_complex(d, self.n_fft, self.hop_length, self.win_length, self.window)
        S = stft_complex(s, self.n_fft, self.hop_length, self.win_length, self.window)

        # log1p magnitude
        X_mag = torch.log1p(torch.abs(X))   # [F, T]
        D_mag = torch.log1p(torch.abs(D))   # [F, T]
        S_mag = torch.log1p(torch.abs(S))   # [F, T]

        # 转成 [T, F]
        X_mag = X_mag.transpose(0, 1).contiguous()
        D_mag = D_mag.transpose(0, 1).contiguous()
        S_mag = S_mag.transpose(0, 1).contiguous()

        # 输入 [2, T, F]
        input_feat = torch.stack([D_mag, X_mag], dim=0)

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

        return input_feat, S_mag, meta

    def __getitem__(self, idx: int):
        sample = self.build_valid_sample(idx)
        return self.sample_to_example(sample)