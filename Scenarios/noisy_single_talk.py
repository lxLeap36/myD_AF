import numpy as np

from .common import (
    to_1d_float32,
    repeat_or_crop,
    apply_rir,
    scale_noise_to_snr,
    joint_peak_normalize,
    build_sample,
)


def generate_noisy_single_talk(
    far_end,
    rir,
    noise,
    snr_db=15.0,
    fs=16000,
    normalize=True,
    peak=0.95,
):
    """
    生成 noisy single-talk 场景

    场景定义：
        s(n) = 0
        v(n) != 0
        d(n) = y(n) + v(n)
        y(n) = x(n) * h(n)

    这里的 single-talk 默认指 far-end single-talk + 背景噪声

    适合评估：
        - 噪声存在时的回声抵消能力
        - ERLE 是否明显下降
        - 传统算法的鲁棒性
    """
    x = to_1d_float32(far_end)
    h = to_1d_float32(rir)
    noise = repeat_or_crop(noise, len(x))

    y = apply_rir(x, h)
    v = scale_noise_to_snr(noise, y, snr_db=snr_db)
    s = np.zeros_like(x, dtype=np.float32)
    d = y + v

    masks = {
        "farend_single_talk_mask": np.ones(len(x), dtype=np.float32),
        "double_talk_mask": np.zeros(len(x), dtype=np.float32),
    }

    meta = {
        "scenario": "noisy_single_talk",
        "fs": fs,
        "duration_sec": len(x) / fs,
        "snr_db": snr_db,
    }

    sample = build_sample(x=x, s=s, v=v, h=h, y=y, d=d, masks=masks, meta=meta)

    if normalize:
        sample = joint_peak_normalize(sample, peak=peak)

    return sample