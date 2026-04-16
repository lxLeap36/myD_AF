import numpy as np

from .common import (
    to_1d_float32,
    apply_rir,
    joint_peak_normalize,
    build_sample,
)


def generate_path_change(
    far_end,
    rir_before,
    rir_after,
    change_time_sec,
    fs=16000,
    normalize=True,
    peak=0.95,
):
    """
    生成 path change 场景

    场景定义：
        s(n) = 0
        v(n) = 0
        在某个时刻 change_time_sec，回声路径从 h1 切换到 h2

    说明：
        这里采用“整段卷积后拼接”的方式近似构造路径突变：
            - change 前用 h1 的输出
            - change 后用 h2 的输出

        这样写实现简单，足够用于 V1 场景验证和重收敛测试。
    """
    x = to_1d_float32(far_end)
    h1 = to_1d_float32(rir_before)
    h2 = to_1d_float32(rir_after)

    change_sample = int(change_time_sec * fs)
    change_sample = max(0, min(change_sample, len(x)))

    y1 = apply_rir(x, h1)
    y2 = apply_rir(x, h2)

    # 路径切换：前半段取 y1，后半段取 y2
    y = np.zeros_like(x, dtype=np.float32)
    y[:change_sample] = y1[:change_sample]
    y[change_sample:] = y2[change_sample:]

    s = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    d = y.copy()

    before_mask = np.zeros(len(x), dtype=np.float32)
    after_mask = np.zeros(len(x), dtype=np.float32)
    before_mask[:change_sample] = 1.0
    after_mask[change_sample:] = 1.0

    masks = {
        "before_change_mask": before_mask,
        "after_change_mask": after_mask,
        "farend_single_talk_mask": np.ones(len(x), dtype=np.float32),
    }

    meta = {
        "scenario": "path_change",
        "fs": fs,
        "duration_sec": len(x) / fs,
        "change_time_sec": change_time_sec,
        "change_sample": change_sample,
    }

    # 这里 h 用 dict 保存，因为路径前后不一样
    sample = build_sample(
        x=x,
        s=s,
        v=v,
        h={"before": h1, "after": h2},
        y=y,
        d=d,
        masks=masks,
        meta=meta,
    )

    if normalize:
        sample = joint_peak_normalize(sample, peak=peak)

    return sample