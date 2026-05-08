import numpy as np

from Tools.nonlinear_distortion import apply_nonlinear_distortion
from .common import (
    to_1d_float32,
    repeat_or_crop,
    apply_rir,
    joint_peak_normalize,
    build_sample,
)


def generate_nonlinear_farend_single_talk(far_end, rir, fs=16000, normalize=True, peak=0.95, delta_1=4, delta_2=4):
    """
    生成包含非线性失真的 far-end single-talk 场景

    场景定义：
        s(n) = 0
        v(n) = 0
        x_nl(n) = apply_nonlinear(x(n))  <-- 模拟扬声器/功放非线性破音
        d(n) = y(n) = x_nl(n) * h(n)     <-- 回声路径卷积发生在失真之后

    适合评估：
        - 非线性回声消除能力 (KLMS, 深度学习模型等)
    """
    x = to_1d_float32(far_end)
    h = to_1d_float32(rir)

    # ========================================================
    # 核心新增：在卷积 RIR 之前，远端信号先经过非线性扬声器失真
    # 注意：自适应滤波器看到的参考信号依然是原始的 x，而不是 x_nl
    # ========================================================
    x_nl = apply_nonlinear_distortion(x, delta_1=delta_1, delta_2=delta_2)

    # 房间的真实回声是由失真后的声音 x_nl 与 RIR 卷积产生的
    y = apply_rir(x_nl, h)

    s = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    d = y.copy()

    masks = {
        "farend_single_talk_mask": np.ones(len(x), dtype=np.float32),
        "double_talk_mask": np.zeros(len(x), dtype=np.float32),
    }

    meta = {
        "scenario": "nonlinear_farend_single_talk",
        "fs": fs,
        "duration_sec": len(x) / fs,
        "delta_1": delta_1,
        "delta_2": delta_2,
    }

    sample = build_sample(x=x, s=s, v=v, h=h, y=y, d=d, masks=masks, meta=meta)

    if normalize:
        sample = joint_peak_normalize(sample, peak=peak)

    return sample