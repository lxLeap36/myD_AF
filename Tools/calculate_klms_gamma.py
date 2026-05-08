import numpy as np
from scipy.spatial.distance import pdist
import soundfile as sf
import os


def estimate_optimal_klms_gamma(audio_signal, filter_length=512, sample_size=2000):
    # 1. 模拟你系统中的 peak=0.95 归一化，保证计算尺度与实际运行完全一致
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = (audio_signal / max_val) * 0.95

    # 2. 剔除绝对静音段以防干扰中位数计算
    energy = audio_signal ** 2
    active_indices = np.where(energy > 1e-4)[0]
    active_signal = audio_signal if len(active_indices) < sample_size else audio_signal

    # 3. 构造自适应滤波器的输入矩阵 (Sliding window / Delay line)
    N = len(active_signal)
    shape = (N - filter_length + 1, filter_length)
    strides = (active_signal.strides[0], active_signal.strides[0])
    X = np.lib.stride_tricks.as_strided(active_signal, shape=shape, strides=strides)

    # 4. 随机采样 2000 个向量以控制计算量
    np.random.seed(42)
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # 5. 计算成对距离并取中位数
    print(f"正在计算欧氏距离平方...")
    sq_distances = pdist(X_sample, metric='sqeuclidean')
    median_sq_dist = np.median(sq_distances)

    recommended_gamma = 1.0 / median_sq_dist
    print(f"\n💡 针对你的音频，推荐的 KLMS kernel_param (gamma) 为: {recommended_gamma:.6f}")
    return recommended_gamma


if __name__ == "__main__":
    # ⚠️ 替换为你本地那半小时的音频路径D:\pyProject\myD_AF\Dataset\clean_speech_test1\speaker05_25min.wav
    audio_file_path = r"D:\pyProject\myD_AF\Dataset\clean_speech_test1\speaker05_25min.wav"

    if os.path.exists(audio_file_path):
        print(f"✅ 找到音频: {audio_file_path}")
        # 获取音频的采样率
        info = sf.info(audio_file_path)
        fs = info.samplerate

        # 核心：只读取前 10 秒！绝不会卡死，且足够进行统计
        read_frames = min(int(10 * fs), info.frames)
        audio_data, _ = sf.read(audio_file_path, frames=read_frames, dtype='float32')

        # 如果是双声道，只取单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        estimate_optimal_klms_gamma(audio_data, filter_length=512)
    else:
        print(f"❌ 找不到文件: {audio_file_path}，请修改路径。")