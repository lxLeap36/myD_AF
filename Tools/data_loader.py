from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


AUDIO_EXTS = {".wav", ".flac"}


def _make_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """
    统一随机数入口：
    - 如果外部传入 rng，就直接使用
    - 否则根据 seed 创建新的 Generator

    这样可以保证：
    - 你在主程序里统一 set_seed 后仍能复现
    - 也可以单独给 loader 传 seed 来固定切片位置
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _to_1d_float32(x):
    """
    转成 1D float32 numpy 数组
    """
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError("Audio must be 1-D after squeeze().")
    return x


def list_audio_files(root_dir: str | Path, exts=None):
    """
    递归列出目录中的所有音频文件
    """
    if exts is None:
        exts = AUDIO_EXTS

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"No audio files found in: {root_dir}")

    return files


def load_audio_segment(
    path: str | Path,
    start_sample: int,
    num_samples: int,
    fs: int = 16000,
    mono: bool = True,
):
    """
    从音频文件中读取一个片段，而不是把整段 30min+ 语音全部读进内存。

    参数:
        path:
            音频文件路径
        start_sample:
            起始采样点（相对于原始文件采样率）
        num_samples:
            读取多少个采样点（相对于原始文件采样率）
        fs:
            目标采样率
        mono:
            多通道时是否转成单通道
    """
    path = Path(path)
    info = sf.info(str(path))

    sr = info.samplerate
    total_frames = info.frames

    start_sample = max(0, min(start_sample, total_frames))
    stop_sample = min(total_frames, start_sample + num_samples)

    wav, read_sr = sf.read(
        str(path),
        start=start_sample,
        stop=stop_sample,
        always_2d=False
    )

    wav = np.asarray(wav, dtype=np.float32)

    # 多通道转单通道
    if wav.ndim == 2 and mono:
        wav = np.mean(wav, axis=1)

    wav = _to_1d_float32(wav)

    if read_sr != fs:
        wav = resample_poly(wav, up=fs, down=read_sr).astype(np.float32)

    return wav


def trim_edges_silence(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    min_keep_sec: float = 0,
):
    """
    轻度静音处理：只裁首尾静音，不动中间自然停顿。
    """
    wav = _to_1d_float32(wav)

    if len(wav) == 0:
        return wav

    peak = np.max(np.abs(wav)) + 1e-8
    thr = peak * (10.0 ** (silence_threshold_db / 20.0))

    active = np.where(np.abs(wav) > thr)[0]

    if len(active) == 0:
        keep_len = min(len(wav), int(round(fs * min_keep_sec)))
        return wav[:keep_len].copy()

    start = int(active[0])
    end = int(active[-1]) + 1
    return wav[start:end].astype(np.float32)


def compress_long_internal_silence(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    max_silence_sec: float = 0.001,
):
    """
    active 模式下使用：
    压缩过长内部静音，但保留短暂停顿。
    """
    wav = _to_1d_float32(wav)

    if len(wav) == 0:
        return wav

    peak = np.max(np.abs(wav)) + 1e-8
    thr = peak * (10.0 ** (silence_threshold_db / 20.0))
    silent_mask = np.abs(wav) <= thr

    max_silence_len = int(round(max_silence_sec * fs))

    out = []
    i = 0
    n = len(wav)

    while i < n:
        if not silent_mask[i]:
            out.append(wav[i])
            i += 1
        else:
            j = i
            while j < n and silent_mask[j]:
                j += 1

            seg = wav[i:j]
            if len(seg) > max_silence_len:
                seg = seg[:max_silence_len]

            out.append(seg)
            i = j

    out = np.concatenate(
        [np.atleast_1d(x).astype(np.float32) for x in out], axis=0
    )
    return out.astype(np.float32)


def preprocess_speech(
    wav,
    fs: int = 16000,
    trim_mode: str = "light",
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.001,
):
    """
    统一预处理接口

    trim_mode:
        - "none"  : 不处理静音
        - "light" : 只裁首尾静音
        - "active": 裁首尾静音 + 压缩过长内部静音
    """
    wav = _to_1d_float32(wav)

    if trim_mode == "none":
        return wav

    if trim_mode == "light":
        return trim_edges_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
        )

    if trim_mode == "active":
        wav = trim_edges_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
        )
        wav = compress_long_internal_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
            max_silence_sec=max_internal_silence_sec,
        )
        return wav

    raise ValueError("trim_mode must be one of: 'none', 'light', 'active'")


def _segment_activity_ratio(
    wav,
    silence_threshold_db: float = -40.0,
):
    """
    计算当前片段的“活跃占比”：
    有多少采样点高于静音阈值。

    用来避免随机裁出来一整段都太安静。
    """
    wav = _to_1d_float32(wav)

    if len(wav) == 0:
        return 0.0

    peak = np.max(np.abs(wav)) + 1e-8
    thr = peak * (10.0 ** (silence_threshold_db / 20.0))
    active = np.mean(np.abs(wav) > thr)
    return float(active)


def _segment_rms_db(wav, eps: float = 1e-8):
    """
    计算片段整体 RMS(dB)
    """
    wav = _to_1d_float32(wav)
    rms = np.sqrt(np.mean(wav ** 2) + eps)
    return float(20.0 * np.log10(rms + eps))


def sample_speech(
    source_dir: str | Path,
    target_duration_sec: float = 20.0,
    fs: int = 16000,
    trim_mode: str = "light",
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.001,
    min_activity_ratio: float = 0.05,
    min_rms_db: float = -45.0,
    max_trials: int = 200,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    从“长语音文件”中随机裁出一段目标时长的语音。

    适用前提：
    - 目录里放的是长语音（例如 >30min）
    - 不是一堆 10s 小文件

    逻辑：
    1. 随机选一个文件
    2. 在该文件里随机选起点
    3. 裁出目标时长片段
    4. 做轻度/active 预处理
    5. 如果片段太安静，则重试

    返回:
        speech: 1D float32 numpy array
        meta:   本次裁剪的元信息
    """
    rng = _make_rng(seed=seed, rng=rng)
    files = list_audio_files(source_dir)

    target_len = int(round(target_duration_sec * fs))

    for _ in range(max_trials):
        file_path = files[int(rng.integers(0, len(files)))]
        info = sf.info(str(file_path))
        file_sr = info.samplerate
        total_frames = info.frames

        # 为了读出目标长度，先换算成原始采样率下需要的采样点数
        needed_frames = int(np.ceil(target_duration_sec * file_sr))

        # 文件太短则跳过（你现在说会放 >30min，理论上不会遇到）
        if total_frames < needed_frames:
            continue

        max_start = total_frames - needed_frames
        start_sample = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

        wav = load_audio_segment(
            path=file_path,
            start_sample=start_sample,
            num_samples=needed_frames,
            fs=fs,
            mono=True,
        )

        wav = preprocess_speech(
            wav,
            fs=fs,
            trim_mode=trim_mode,
            silence_threshold_db=silence_threshold_db,
            max_internal_silence_sec=max_internal_silence_sec,
        )

        # 如果预处理后太短，则跳过重试
        if len(wav) < target_len:
            continue

        # 如果预处理后仍然比目标长，则在内部再随机裁一次
        if len(wav) > target_len:
            max_start2 = len(wav) - target_len
            start2 = int(rng.integers(0, max_start2 + 1)) if max_start2 > 0 else 0
            wav = wav[start2:start2 + target_len]

        wav = wav.astype(np.float32)

        activity_ratio = _segment_activity_ratio(
            wav,
            silence_threshold_db=silence_threshold_db,
        )
        rms_db = _segment_rms_db(wav)

        # 太安静则重试
        if activity_ratio < min_activity_ratio:
            continue
        if rms_db < min_rms_db:
            continue

        meta = {
            "source_dir": str(source_dir),
            "file_path": str(file_path),
            "fs": fs,
            "target_duration_sec": target_duration_sec,
            "target_len": target_len,
            "trim_mode": trim_mode,
            "silence_threshold_db": silence_threshold_db,
            "max_internal_silence_sec": max_internal_silence_sec,
            "activity_ratio": activity_ratio,
            "rms_db": rms_db,
            "start_sample_in_raw_file": start_sample,
        }

        return wav, meta

    raise RuntimeError(
        "Failed to sample a valid speech segment. "
        "Please relax min_activity_ratio / min_rms_db or check your dataset."
    )


def sample_far_end(
    source_dir: str | Path,
    target_duration_sec: float = 15.0,
    fs: int = 16000,
    trim_mode: str = "active",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    far-end 采样包装器

    建议:
    - far-end 默认用 active，更容易保证激励充分，
      对 LMS / NLMS / RLS 看收敛更友好。
    """
    return sample_speech(
        source_dir=source_dir,
        target_duration_sec=target_duration_sec,
        fs=fs,
        trim_mode=trim_mode,
        seed=seed,
        rng=rng,
    )


def sample_near_end(
    source_dir: str | Path,
    target_duration_sec: float = 20.0,
    fs: int = 16000,
    trim_mode: str = "light",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    near-end 采样包装器

    建议:
    - near-end 默认用 light，保留更多自然停顿，
      更接近真实说话节奏。
    """
    return sample_speech(
        source_dir=source_dir,
        target_duration_sec=target_duration_sec,
        fs=fs,
        trim_mode=trim_mode,
        seed=seed,
        rng=rng,
    )


if __name__ == "__main__":
    # 简单自测
    far_end, far_meta = sample_far_end(
        source_dir="../Dataset/clean_speech1",
        target_duration_sec=15.0,
        fs=16000,
        trim_mode="active",
        seed=42,
    )

    near_end, near_meta = sample_near_end(
        source_dir="../Dataset/clean_speech2",
        target_duration_sec=20.0,
        fs=16000,
        trim_mode="light",
        seed=123,
    )

    print("far_end:", far_end.shape, far_end.dtype)
    print("near_end:", near_end.shape, near_end.dtype)
    print("far_meta:", far_meta)
    print("near_meta:", near_meta)