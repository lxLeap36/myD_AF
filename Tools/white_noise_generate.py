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
    将噪声归一化到指定 RMS。
    注意：
    这里只做基础归一化，后续真正的 SNR 缩放仍由场景函数里的
    scale_noise_to_snr(...) 完成。
    """
    x = np.asarray(x, dtype=np.float32)
    cur_rms = np.sqrt(np.mean(x ** 2) + eps)
    x = x / cur_rms * target_rms
    return x.astype(np.float32)


def generate_white_noise(
    length=None,
    fs=16000,
    duration_sec=None,
    mean=0.0,
    std=1.0,
    noise_type="gaussian",
    normalize=True,
    target_rms=1.0,
    seed=None,
):
    """
    生成白噪声，默认是零均值高斯白噪声。

    参数:
        length:
            目标采样点数。如果给了，就优先使用。
        fs:
            采样率。只有在通过 duration_sec 推长度时才会用到。
        duration_sec:
            目标时长（秒）。当 length=None 时使用。
        mean:
            噪声均值，默认 0。
        std:
            原始噪声标准差，默认 1。
        noise_type:
            可选 "gaussian" 或 "uniform"。
            当前推荐默认 "gaussian"。
        normalize:
            是否做 RMS 归一化。建议保持 True，方便后续统一用你的 SNR 函数缩放。
        target_rms:
            归一化后的 RMS。
        seed:
            随机种子，便于复现。

    返回:
        noise: np.ndarray, shape=(length,), dtype=float32
    """
    length = _resolve_length(length=length, fs=fs, duration_sec=duration_sec)
    rng = np.random.default_rng(seed)

    if noise_type == "gaussian":
        # 零均值高斯白噪声：最常见、最推荐
        noise = rng.normal(loc=mean, scale=std, size=length)
    elif noise_type == "uniform":
        # 也可以生成均匀白噪声，但默认不推荐当主基准
        # 这里用方差匹配的方式，使其和 std 对应
        half_range = np.sqrt(3.0) * std
        noise = rng.uniform(low=mean - half_range, high=mean + half_range, size=length)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'.")

    noise = noise.astype(np.float32)

    if normalize:
        noise = _normalize_rms(noise, target_rms=target_rms)

    return noise


if __name__ == "__main__":
    # 简单测试：生成 10 秒高斯白噪声
    noise = generate_white_noise(duration_sec=10, fs=16000, seed=42)
    print(noise.shape, noise.dtype, np.mean(noise), np.std(noise))