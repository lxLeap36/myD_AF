import numpy as np


def _resolve_length(length=None, fs=16000, duration_sec=None):
    """
    统一解析目标长度：
    - 如果直接给了 length，就用 length
    - 否则根据 fs 和 duration_sec 计算
    """
    if length is None:
        if duration_sec is None:
            raise ValueError("Either length or duration_sec must be provided.")
        length = int(round(fs * duration_sec))

    if length <= 0:
        raise ValueError("length must be a positive integer.")

    return int(length)


def _normalize_rms(x, target_rms=1.0, eps=1e-8):
    """
    基础 RMS 归一化。
    这里只负责输出一个可控幅度的“原始脉冲噪声”，
    最终 SNR 仍建议交给你已有的 scale_noise_to_snr(...) 去做。
    """
    x = np.asarray(x, dtype=np.float32)
    cur_rms = np.sqrt(np.mean(x ** 2) + eps)
    x = x / cur_rms * target_rms
    return x.astype(np.float32)


def generate_impulse_noise(
    length=None,
    fs=16000,
    duration_sec=None,
    mode="point",
    num_impulses=None,
    impulse_prob=1e-4,
    amplitude_range=(3.0, 8.0),
    signed=True,
    background_std=0.0,
    normalize=True,
    target_rms=1.0,
    seed=None,
):
    """
    生成脉冲噪声（impulse noise）。

    参数:
        length:
            目标采样点数。如果给了，就优先使用。
        fs:
            采样率。只有在通过 duration_sec 推长度时才会用到。
        duration_sec:
            目标时长（秒）。当 length=None 时使用。
        mode:
            "point"  -> 单点脉冲
            "burst"  -> 短脉冲串
        num_impulses:
            脉冲个数。如果不给，就按 impulse_prob * length 自动计算。
        impulse_prob:
            每个采样点成为脉冲起点的概率（仅在 num_impulses=None 时起作用）。
        amplitude_range:
            脉冲幅值范围 (min_amp, max_amp)
        signed:
            是否允许正负脉冲
        background_std:
            是否叠加一个很小的高斯背景。默认 0，表示纯脉冲噪声。
            如果想更接近真实“点击/爆裂”噪声，可设为 0.01 左右。
        normalize:
            是否做 RMS 归一化
        target_rms:
            归一化目标 RMS
        seed:
            随机种子

    返回:
        noise: np.ndarray, shape=(length,), dtype=float32
    """
    length = _resolve_length(length=length, fs=fs, duration_sec=duration_sec)
    rng = np.random.default_rng(seed)

    noise = np.zeros(length, dtype=np.float32)

    # 可选：加一点很弱的高斯背景，让噪声不至于“除了脉冲全是零”
    if background_std > 0:
        noise += rng.normal(loc=0.0, scale=background_std, size=length).astype(np.float32)

    # 自动决定脉冲数
    if num_impulses is None:
        num_impulses = max(1, int(round(length * impulse_prob)))

    num_impulses = min(num_impulses, length)

    # 随机选择脉冲位置
    positions = rng.choice(length, size=num_impulses, replace=False)

    # 随机脉冲幅值
    amps = rng.uniform(amplitude_range[0], amplitude_range[1], size=num_impulses)

    if signed:
        signs = rng.choice([-1.0, 1.0], size=num_impulses)
        amps = amps * signs

    if mode == "point":
        # 单点脉冲：最简单，也最适合先做鲁棒性测试
        noise[positions] += amps.astype(np.float32)

    elif mode == "burst":
        # 短脉冲串：更像 click/pop 一类短时突发干扰
        for pos, amp in zip(positions, amps):
            burst_len = int(rng.integers(2, 11))  # 2~10 个采样点
            end = min(length, pos + burst_len)

            seg_len = end - pos
            if seg_len <= 0:
                continue

            # 做一个简单衰减，不让 burst 太生硬
            decay = np.linspace(1.0, 0.3, seg_len, dtype=np.float32)
            noise[pos:end] += float(amp) * decay

    else:
        raise ValueError("mode must be 'point' or 'burst'.")

    if normalize:
        noise = _normalize_rms(noise, target_rms=target_rms)

    return noise.astype(np.float32)


if __name__ == "__main__":
    # 简单测试：生成 10 秒单点脉冲噪声
    noise = generate_impulse_noise(duration_sec=10, fs=16000, mode="point", seed=42)
    print(noise.shape, noise.dtype, np.mean(noise), np.std(noise))