import numpy as np

from .common import (
    to_1d_float32,
    repeat_or_crop,
    apply_rir,
    scale_near_to_ser,
    joint_peak_normalize,
    build_sample,
    default_double_talk_segments,
)


def generate_double_talk(
    far_end,
    near_end,
    rir,
    fs=16000,
    ser_db=0.0,
    segments=None,
    normalize=True,
    peak=0.95,
):
    """
    生成 double-talk 场景

    场景定义：
        far-end 一直存在
        near-end 只在某些时间段存在
        v(n) = 0
        d(n) = y(n) + s(n)

    参数：
        ser_db:
            控制双讲段里 near-end 和 echo 的相对强弱
            SER = 10 * log10(P_near / P_echo)

        segments:
            一个列表，形如：
            [
                {"type": "fst", "start": 0.0, "end": 2.0},
                {"type": "dt",  "start": 2.0, "end": 4.0},
            ]
            如果不提供，就使用默认分段
    """
    x = to_1d_float32(far_end)
    near_src = to_1d_float32(near_end)
    h = to_1d_float32(rir)

    duration_sec = len(x) / fs
    if segments is None:
        segments = default_double_talk_segments(duration_sec)

    y = apply_rir(x, h)
    s = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)

    # 近端语音源如果太短就重复
    near_src = repeat_or_crop(near_src, len(x) * 2)

    # 记录双讲 mask
    dt_mask = np.zeros(len(x), dtype=np.float32)
    fst_mask = np.zeros(len(x), dtype=np.float32)

    near_cursor = 0

    for seg in segments:
        seg_type = seg["type"]
        start = int(seg["start"] * fs)
        end = int(seg["end"] * fs)

        start = max(0, min(start, len(x)))
        end = max(0, min(end, len(x)))

        if end <= start:
            continue

        if seg_type == "dt":
            seg_len = end - start
            chunk = near_src[near_cursor: near_cursor + seg_len]

            # 如果 chunk 不够，再补一段
            if len(chunk) < seg_len:
                chunk = repeat_or_crop(near_src, seg_len)

            s[start:end] = chunk[:seg_len]
            dt_mask[start:end] = 1.0
            near_cursor += seg_len

        elif seg_type == "fst":
            fst_mask[start:end] = 1.0

    # 只在双讲区缩放 near-end，使其与 echo 满足目标 SER
    if np.any(dt_mask > 0):
        idx = dt_mask > 0
        s[idx] = scale_near_to_ser(s[idx], y[idx], ser_db=ser_db)

    d = y + s

    masks = {
        "double_talk_mask": dt_mask,
        "farend_single_talk_mask": fst_mask,
    }

    meta = {
        "scenario": "double_talk",
        "fs": fs,
        "duration_sec": duration_sec,
        "ser_db": ser_db,
        "segments": segments,
    }

    sample = build_sample(x=x, s=s, v=v, h=h, y=y, d=d, masks=masks, meta=meta)

    if normalize:
        sample = joint_peak_normalize(sample, peak=peak)

    return sample