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


def _random_partition(total, n_parts, min_value, rng):
    """
    把 total 随机分成 n_parts 份，每份至少 min_value。
    """
    if n_parts <= 0:
        return []

    base = np.full(n_parts, float(min_value), dtype=np.float32)
    remain = float(total) - float(np.sum(base))
    if remain < 0:
        raise ValueError("total is too small for the requested partition constraints.")

    if remain == 0:
        return base.tolist()

    w = rng.random(n_parts).astype(np.float32)
    w = w / (np.sum(w) + 1e-8)
    parts = base + remain * w
    return parts.tolist()


def sample_random_double_talk_segments(
    duration_sec,
    seed=None,
    rng=None,
    num_dt_range=(1, 3),
    total_dt_ratio_range=(0.35, 0.75),
    min_dt_sec=0.30,
    min_fst_sec=0.15,
):
    """
    为 double-talk 场景随机生成分段。

    设计思路：
    - 保持“fst / dt 交替”这个基本形式
    - 但 dt 的段数、各段时长、fst 各段时长都随机
    - 不再固定为 25% / 25% / 25% / 25%
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    duration_sec = float(duration_sec)

    num_dt = int(rng.integers(num_dt_range[0], num_dt_range[1] + 1))
    num_fst = num_dt + 1   # 仍然交替：fst-dt-fst-dt-...-fst

    dt_ratio = float(rng.uniform(total_dt_ratio_range[0], total_dt_ratio_range[1]))
    total_dt = duration_sec * dt_ratio
    total_fst = duration_sec - total_dt

    # 可行性检查
    if total_dt < num_dt * min_dt_sec:
        total_dt = num_dt * min_dt_sec
        total_fst = duration_sec - total_dt

    if total_fst < num_fst * min_fst_sec:
        total_fst = num_fst * min_fst_sec
        total_dt = duration_sec - total_fst

    if total_dt <= 0 or total_fst <= 0:
        # 实在不够分，就退回默认模板
        return default_double_talk_segments(duration_sec)

    dt_lens = _random_partition(total_dt, num_dt, min_dt_sec, rng)
    fst_lens = _random_partition(total_fst, num_fst, min_fst_sec, rng)

    segments = []
    t = 0.0
    for i in range(num_dt):
        # fst
        fst_len = fst_lens[i]
        segments.append({
            "type": "fst",
            "start": t,
            "end": min(duration_sec, t + fst_len),
        })
        t += fst_len

        # dt
        dt_len = dt_lens[i]
        segments.append({
            "type": "dt",
            "start": t,
            "end": min(duration_sec, t + dt_len),
        })
        t += dt_len

    # 最后一个 fst
    last_fst = fst_lens[-1]
    segments.append({
        "type": "fst",
        "start": t,
        "end": duration_sec,
    })

    # 清理一下边界
    cleaned = []
    for seg in segments:
        s = max(0.0, float(seg["start"]))
        e = min(duration_sec, float(seg["end"]))
        if e > s:
            cleaned.append({"type": seg["type"], "start": s, "end": e})

    # 保证最后结束在 duration_sec
    if len(cleaned) > 0:
        cleaned[-1]["end"] = duration_sec

    return cleaned