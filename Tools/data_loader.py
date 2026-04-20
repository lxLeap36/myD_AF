from __future__ import annotations

from pathlib import Path
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


# === 新增：基于短时帧 RMS 的帧级静音判定与压缩实现 ===
def _frame_params(fs: int, frame_len_ms: float = 25.0, hop_len_ms: float = 10.0):
    """
    计算帧长与帧移（样本数），并保证至少为 1
    """
    frame_len = max(1, int(round(frame_len_ms * fs / 1000.0)))
    hop_len = max(1, int(round(hop_len_ms * fs / 1000.0)))
    return frame_len, hop_len


def _frame_rms(wav: np.ndarray, frame_len: int, hop_len: int, eps: float = 1e-12):
    """
    计算每帧 RMS（按 frame_len 和 hop_len 分帧）。
    返回:
        rms: 1-D numpy array, 每帧的 rms（非 dB）
        starts: 每帧对应的起始采样点索引

    说明：如果 wav 比一帧还短，则返回单帧（RMS 为整段 RMS），starts=[0]
    """
    wav = _to_1d_float32(wav)
    N = len(wav)

    if N == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    if N <= frame_len:
        rms = np.sqrt(np.mean(wav ** 2) + eps)
        return np.array([rms], dtype=np.float32), np.array([0], dtype=np.int32)

    starts = np.arange(0, N - frame_len + 1, hop_len, dtype=np.int32)
    rms_list = []
    for s in starts:
        frame = wav[s:s + frame_len]
        rms_list.append(np.sqrt(np.mean(frame ** 2) + eps))

    return np.asarray(rms_list, dtype=np.float32), starts


def _frame_activity_mask(rms: np.ndarray, energy_threshold_ratio: float = 0.1):
    """
    基于帧 RMS 的活跃帧掩码。采用相对阈值：thr = max(rms) * energy_threshold_ratio
    返回 boolean 数组，True 表示 active（有人声）
    """
    if rms.size == 0:
        return np.array([], dtype=bool)
    max_r = float(np.max(rms))
    thr = max_r * float(energy_threshold_ratio)
    # 防止 thr 为 0（全静音）导致所有帧都 False
    if thr <= 0.0:
        return rms > 0.0
    return rms > thr


def compress_long_internal_silence_framewise(
    wav: np.ndarray,
    fs: int = 16000,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
    max_internal_silence_sec: float = 0.3,
):
    """
    基于帧级 RMS 的内部静音压缩。

    算法：
    1) 按 frame_len/hop_len 划分帧，计算每帧 RMS
    2) 以 max_frame_rms * energy_threshold_ratio 作为静音阈值，得到静音/活跃帧 mask
    3) 找到连续的静音帧段，对于每一段：
       - 如果静音时长 <= max_internal_silence_sec，则保留完整片段
       - 否则把该静音片段裁切为 max_internal_silence_sec 长度（以采样点为单位截断），保留片段开头部分
    4) 按原始采样点拼接保留片段并返回

    说明：
    - 这里采用帧到采样点的直接映射（每帧对应起始采样点 start，帧覆盖到 start+frame_len），
      压缩时按采样点截取整个静音段，保证输出仍为 1D waveform。
    - 该方法比逐采样阈值更鲁棒，不会把零交叉的瞬时低幅破坏为静音段。
    """
    wav = _to_1d_float32(wav)
    if len(wav) == 0:
        return wav

    frame_len, hop_len = _frame_params(fs, frame_len_ms, hop_len_ms)
    rms, starts = _frame_rms(wav, frame_len, hop_len)

    # 若帧数为 1（短音频），直接返回原始 wav
    if rms.size <= 1:
        return wav

    active_mask = _frame_activity_mask(rms, energy_threshold_ratio=energy_threshold_ratio)
    silent_mask = ~active_mask

    # 找出静音帧的连续区间（基于帧索引）
    out_segments = []
    n_frames = len(rms)
    cur = 0
    N = len(wav)
    max_silence_len_samples = int(round(max_internal_silence_sec * fs))

    while cur < n_frames:
        if active_mask[cur]:
            # 活跃帧，从这个帧的起始采样点开始向后，直到遇到静音帧
            start_sample = int(starts[cur])
            # 找下一静音帧索引
            j = cur + 1
            while j < n_frames and active_mask[j]:
                j += 1
            # j 是第一个静音帧或 n_frames
            # 活跃段结束位置采样点：若 j==0 -> end = starts[0]+frame_len else end = starts[j-1]+frame_len
            end_frame_idx = max(cur, j - 1)
            end_sample = int(starts[end_frame_idx]) + frame_len
            out_segments.append(wav[start_sample:end_sample])
            cur = j
        else:
            # 静音帧段，从 cur 开始
            i = cur
            while i < n_frames and silent_mask[i]:
                i += 1
            # 静音段帧索引范围 [cur, i-1]
            sil_start_sample = int(starts[cur])
            sil_end_frame_idx = i - 1
            sil_end_sample = int(starts[sil_end_frame_idx]) + frame_len

            sil_len = sil_end_sample - sil_start_sample
            if sil_len <= 0:
                cur = i
                continue

            if sil_len <= max_silence_len_samples:
                # 保留整段静音
                out_segments.append(wav[sil_start_sample:sil_end_sample])
            else:
                # 压缩：保留静音段开头的一小段，长度为 max_silence_len_samples
                out_segments.append(wav[sil_start_sample:sil_start_sample + max_silence_len_samples])

            cur = i

    if len(out_segments) == 0:
        return np.array([], dtype=np.float32)

    out = np.concatenate([np.atleast_1d(seg).astype(np.float32) for seg in out_segments], axis=0)
    return out.astype(np.float32)


# === 结束新增函数 ===


def compress_long_internal_silence(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    max_silence_sec: float = 0.001,
):
    """
    兼容接口包装：保留原函数签名，但内部改为基于帧的实现，使用默认帧参数。

    旧接口保留用于兼容，但内部使用 frame-based 方法以获得更好的语音连贯性。
    """
    # 将旧的参数名映射到新的帧级函数参数，默认阈值与帧参数
    # silence_threshold_db 仍然保留作为相对能量阈值的辅助（转换成 ratio）
    # 将 silence_threshold_db -> energy_threshold_ratio（大致映射）
    # 经验映射： silence_threshold_db = -40dB -> ratio ~ 0.01; -20dB -> 0.1
    # 这里采用一个简单映射： ratio = 10^(silence_db/20)
    energy_threshold_ratio = 10.0 ** (silence_threshold_db / 20.0)

    # 由于 frame-level 判定比 sample-level 更保守，给出稍大的默认 max_internal_silence_sec
    # 如果调用者传入很小的值（例如 0.001），仍然尊重
    return compress_long_internal_silence_framewise(
        wav=wav,
        fs=fs,
        frame_len_ms=25.0,
        hop_len_ms=10.0,
        energy_threshold_ratio=energy_threshold_ratio,
        max_internal_silence_sec=max_silence_sec,
    )


def preprocess_speech(
    wav,
    fs: int = 16000,
    trim_mode: str = "light",
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.001,
    # 新增帧级参数，并提供合理默认
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
):
    """
    统一预处理接口

    trim_mode:
        - "none"  : 不处理静音
        - "light" : 只裁首尾静音
        - "active": 裁首尾静音 + 压缩过长内部静音（基于短时帧 RMS）

    说明：
    - 将内部静音压缩由采样点级别改为帧级（RMS），以避免将短时零交叉误判为静音。
    - active 模式下使用 frame_len_ms/hop_len_ms 与 energy_threshold_ratio 计算帧 RMS 并判定静音帧。
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

        # 将 silence_threshold_db 转成能量比供帧级判定使用（与 compress_long_internal_silence 中相同）
        energy_ratio = 10.0 ** (silence_threshold_db / 20.0)

        wav = compress_long_internal_silence_framewise(
            wav,
            fs=fs,
            frame_len_ms=frame_len_ms,
            hop_len_ms=hop_len_ms,
            energy_threshold_ratio=energy_ratio,
            max_internal_silence_sec=max_internal_silence_sec,
        )
        return wav

    raise ValueError("trim_mode must be one of: 'none', 'light', 'active'")


def _segment_activity_ratio(
    wav,
    silence_threshold_db: float = -40.0,
):
    """
    兼容旧接口：调用帧级实现并返回活跃帧比例（0~1）。

    旧接口按采样点比例计算；现在改为按帧级 RMS 计算，能更合理地衡量该片段中"说话时间占比"。
    """
    # 默认帧参数与 preprocess 中一致
    return _segment_activity_ratio_framewise(
        wav,
        fs=16000,
        frame_len_ms=25.0,
        hop_len_ms=10.0,
        energy_threshold_ratio=10.0 ** (silence_threshold_db / 20.0),
    )


def _segment_activity_ratio_framewise(
    wav,
    fs: int = 16000,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
):
    """
    基于短时帧 RMS 的活跃度估计。

    返回： active_frames / total_frames（float）
    说明：这比逐样本点的阈值比例更能反映语音占比。
    """
    wav = _to_1d_float32(wav)
    if len(wav) == 0:
        return 0.0

    frame_len, hop_len = _frame_params(fs, frame_len_ms, hop_len_ms)
    rms, _ = _frame_rms(wav, frame_len, hop_len)
    if rms.size == 0:
        return 0.0

    mask = _frame_activity_mask(rms, energy_threshold_ratio=energy_threshold_ratio)
    if mask.size == 0:
        return 0.0
    return float(np.sum(mask) / float(mask.size))


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
    min_activity_ratio: float = 0.20,
    min_rms_db: float = -45.0,
    max_trials: int = 20,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    # 新增帧级参数，保留默认值，向下传递到 preprocess
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
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
            frame_len_ms=frame_len_ms,
            hop_len_ms=hop_len_ms,
            energy_threshold_ratio=energy_threshold_ratio,
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
    # 将 max_internal_silence_sec 向下传递以支持全局 cfg 配置
    max_internal_silence_sec: float = 0.001,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
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
        max_internal_silence_sec=max_internal_silence_sec,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        energy_threshold_ratio=energy_threshold_ratio,
        rng=rng,
    )


def sample_near_end(
    source_dir: str | Path,
    target_duration_sec: float = 20.0,
    fs: int = 16000,
    trim_mode: str = "active",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    # 支持向下传递 max_internal_silence_sec / 帧参数
    max_internal_silence_sec: float = 0.001,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
):
    """
    near-end 采样包装器

    建议:
    - near-end 默认用 active，以便压缩过长停顿（frame-level RMS 判定），
      但你可以改回 'light' 来保留更多自然短停顿。
    """
    return sample_speech(
        source_dir=source_dir,
        target_duration_sec=target_duration_sec,
        fs=fs,
        trim_mode=trim_mode,
        seed=seed,
        rng=rng,
        max_internal_silence_sec=max_internal_silence_sec,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        energy_threshold_ratio=energy_threshold_ratio,
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