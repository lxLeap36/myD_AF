import numpy as np

from .common import (
    to_1d_float32,
    repeat_or_crop,
    apply_rir,
    joint_peak_normalize,
    build_sample,
)


def generate_farend_single_talk(far_end, rir, fs=16000, normalize=True, peak=0.95):
    """
    生成 far-end single-talk 场景

    场景定义：
        s(n) = 0
        v(n) = 0
        d(n) = y(n) = x(n) * h(n)

    适合评估：
        - 纯回声抵消能力
        - ERLE
        - 收敛速度
        - NMIS / MSD（若已知真实路径）
    """
    x = to_1d_float32(far_end)
    h = to_1d_float32(rir)

    y = apply_rir(x, h)
    s = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    d = y.copy()

    masks = {
        "farend_single_talk_mask": np.ones(len(x), dtype=np.float32),
        "double_talk_mask": np.zeros(len(x), dtype=np.float32),
    }

    meta = {
        "scenario": "farend_single_talk",
        "fs": fs,
        "duration_sec": len(x) / fs,
    }

    sample = build_sample(x=x, s=s, v=v, h=h, y=y, d=d, masks=masks, meta=meta)

    if normalize:
        sample = joint_peak_normalize(sample, peak=peak)
        # 联合归一化，在 x / s / v / y / d 这些信号里找一个全局最大幅值，然后缩放所有信号，使得这个最大幅值等于 peak
    return sample