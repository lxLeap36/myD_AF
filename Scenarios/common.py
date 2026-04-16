import numpy as np
from scipy.signal import fftconvolve


def to_1d_float32(x):
    """
    将输入转成 1 维 float32 numpy 数组
    """
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError("Input signal must be 1-D after squeeze().")
    return x


def repeat_or_crop(x, target_len):
    """
    将信号裁剪或重复到目标长度
    """
    x = to_1d_float32(x)

    if len(x) == target_len:
        return x.copy()

    if len(x) > target_len:
        return x[:target_len].copy()

    # 长度不够时重复拼接
    repeat_times = int(np.ceil(target_len / len(x)))
    x = np.tile(x, repeat_times)
    return x[:target_len].copy()


def rms(x, eps=1e-8):
    """
    计算均方根
    """
    x = to_1d_float32(x)
    return np.sqrt(np.mean(x ** 2) + eps)


def apply_rir(x, h):
    """
    用 RIR / 回声路径对 far-end 做卷积，生成 echo

    y(n) = x(n) * h(n)
    """
    x = to_1d_float32(x)
    h = to_1d_float32(h)

    y = fftconvolve(x, h, mode="full")[:len(x)]
    return y.astype(np.float32)


def scale_noise_to_snr(noise, reference, snr_db, eps=1e-8):
    """
    按指定 SNR 缩放噪声，使得：
        SNR = 10 * log10( P_reference / P_noise )

    常用于 noisy_single_talk：
        reference 可以取 echo y
    """
    noise = to_1d_float32(noise)
    reference = to_1d_float32(reference)

    ref_power = np.mean(reference ** 2) + eps
    noise_power = np.mean(noise ** 2) + eps

    target_noise_power = ref_power / (10 ** (snr_db / 10.0))
    gain = np.sqrt(target_noise_power / noise_power)

    return (noise * gain).astype(np.float32)


def scale_near_to_ser(near, echo, ser_db, eps=1e-8):
    """
    按指定 SER 缩放 near-end，使得：
        SER = 10 * log10( P_near / P_echo )

    常用于 double_talk：
        near 为近端语音
        echo 为回声
    """
    near = to_1d_float32(near)
    echo = to_1d_float32(echo)

    near_power = np.mean(near ** 2) + eps
    echo_power = np.mean(echo ** 2) + eps

    target_near_power = echo_power * (10 ** (ser_db / 10.0))
    gain = np.sqrt(target_near_power / near_power)

    return (near * gain).astype(np.float32)


def joint_peak_normalize(sample_dict, keys=("x", "s", "v", "y", "d"), peak=0.95, eps=1e-8):
    """
    对多个信号一起做联合峰值归一化，保证它们相对比例不变
    """
    max_amp = 0.0
    for k in keys:
        if k in sample_dict:
            sig = np.asarray(sample_dict[k], dtype=np.float32)
            if sig.size > 0:
                max_amp = max(max_amp, float(np.max(np.abs(sig))))

    if max_amp < eps:
        return sample_dict

    scale = peak / max_amp
    for k in keys:
        if k in sample_dict:
            sample_dict[k] = (sample_dict[k] * scale).astype(np.float32)

    return sample_dict


def build_sample(x, s, v, h, y, d, masks=None, meta=None):
    """
    统一打包场景输出
    """
    if masks is None:
        masks = {}
    if meta is None:
        meta = {}

    return {
        "x": x.astype(np.float32),   # far-end
        "s": s.astype(np.float32),   # near-end
        "v": v.astype(np.float32),   # noise
        "h": h,                      # 回声路径；path_change 场景下可以是 dict
        "y": y.astype(np.float32),   # echo
        "d": d.astype(np.float32),   # mic
        "masks": masks,              # 各种时段 mask
        "meta": meta,                # 场景元信息
    }


def default_double_talk_segments(duration_sec):
    """
    默认的双讲分段：
    - 前 25%: far-end single-talk
    - 中间 25%: double-talk
    - 再 25%: far-end single-talk
    - 最后 25%: double-talk
    """
    t1 = 0.25 * duration_sec
    t2 = 0.50 * duration_sec
    t3 = 0.75 * duration_sec

    return [
        {"type": "fst", "start": 0.0, "end": t1},
        {"type": "dt",  "start": t1, "end": t2},
        {"type": "fst", "start": t2, "end": t3},
        {"type": "dt",  "start": t3, "end": duration_sec},
    ]