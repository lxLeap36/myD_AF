from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


AUDIO_EXTS = {".wav", ".flac"}


def _make_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """
    统一随机数入口：
    - 如果外部传入 rng，就直接使用
    - 否则根据 seed 创建新的 Generator
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _to_1d_float32(x):
    """转成 1D float32 numpy 数组。"""
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError("Audio must be 1-D after squeeze().")
    return x


def list_audio_files(root_dir: str | Path, exts=None):
    """递归列出目录中的所有音频文件。"""
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
    从音频文件中读取一个片段，而不是把整段长语音全部读进内存。
    """
    path = Path(path)
    info = sf.info(str(path))

    total_frames = info.frames

    start_sample = max(0, min(start_sample, total_frames))
    stop_sample = min(total_frames, start_sample + num_samples)

    wav, read_sr = sf.read(
        str(path),
        start=start_sample,
        stop=stop_sample,
        always_2d=False,
    )

    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2 and mono:
        wav = np.mean(wav, axis=1)

    wav = _to_1d_float32(wav)

    if read_sr != fs:
        wav = resample_poly(wav, up=fs, down=read_sr).astype(np.float32)

    return wav


def _frame_params(fs: int, frame_len_ms: float = 25.0, hop_len_ms: float = 10.0) -> Tuple[int, int]:
    """根据毫秒配置计算帧长与帧移（单位：采样点）。"""
    frame_len = max(1, int(round(frame_len_ms * fs / 1000.0)))
    hop_len = max(1, int(round(hop_len_ms * fs / 1000.0)))
    return frame_len, hop_len


def _frame_rms(
    wav: np.ndarray,
    frame_len: int,
    hop_len: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每帧 RMS，并返回每帧起始位置。

    返回:
        rms:    [num_frames]
        starts: [num_frames]
    """
    wav = _to_1d_float32(wav)
    n = len(wav)

    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    if n <= frame_len:
        rms = np.sqrt(np.mean(wav ** 2) + eps)
        return np.array([rms], dtype=np.float32), np.array([0], dtype=np.int32)

    starts = list(range(0, n - frame_len + 1, hop_len))
    last_start = n - frame_len
    if starts[-1] != last_start:
        starts.append(last_start)
    starts = np.asarray(starts, dtype=np.int32)

    rms_list = []
    for s in starts:
        frame = wav[s:s + frame_len]
        rms_list.append(np.sqrt(np.mean(frame ** 2) + eps))
    return np.asarray(rms_list, dtype=np.float32), starts


def _energy_threshold_ratio_from_db(silence_threshold_db: float) -> float:
    """
    把 dB 相对阈值换成线性比例。
    例如:
        -20 dB -> 0.1
        -40 dB -> 0.01
    """
    return float(10.0 ** (silence_threshold_db / 20.0))


def _frame_activity_mask(
    rms: np.ndarray,
    silence_threshold_db: float = -40.0,
    energy_threshold_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    基于帧 RMS 的活跃帧掩码。
    threshold = max_frame_rms * ratio
    """
    if rms.size == 0:
        return np.array([], dtype=bool)

    if energy_threshold_ratio is None:
        energy_threshold_ratio = _energy_threshold_ratio_from_db(silence_threshold_db)

    max_rms = float(np.max(rms))
    if max_rms <= 0.0:
        return np.zeros_like(rms, dtype=bool)

    thr = max_rms * float(energy_threshold_ratio)
    return rms > thr


def _active_masks_from_frame_rms(
    wav: np.ndarray,
    fs: int = 16000,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    silence_threshold_db: float = -40.0,
    energy_threshold_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    先按帧做活动检测，再映射成 sample 级 active mask。

    返回:
        sample_active_mask: [N]
        frame_active_mask:  [num_frames]
    """
    wav = _to_1d_float32(wav)
    n = len(wav)

    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    frame_len, hop_len = _frame_params(fs, frame_len_ms, hop_len_ms)
    rms, starts = _frame_rms(wav, frame_len, hop_len)
    frame_active = _frame_activity_mask(
        rms,
        silence_threshold_db=silence_threshold_db,
        energy_threshold_ratio=energy_threshold_ratio,
    )

    sample_active = np.zeros(n, dtype=bool)
    for is_active, s in zip(frame_active, starts):
        if not is_active:
            continue
        e = min(int(s) + frame_len, n)
        sample_active[int(s):e] = True

    return sample_active, frame_active


def _find_runs(mask: np.ndarray):
    """
    找到布尔掩码的连续区段。
    返回:
        [(value, start, end_exclusive), ...]
    """
    if mask.size == 0:
        return []

    runs = []
    start = 0
    cur = bool(mask[0])

    for i in range(1, len(mask)):
        val = bool(mask[i])
        if val != cur:
            runs.append((cur, start, i))
            start = i
            cur = val

    runs.append((cur, start, len(mask)))
    return runs


def trim_edges_silence(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    min_keep_sec: float = 0.0,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """
    帧级首尾静音裁剪。

    说明:
    - 这里不再按逐采样点 abs(wav) > thr 判静音，而是按短时帧 RMS 判定。
    - 这样更符合语音“停顿”的概念，也不会把零交叉附近的瞬时低幅值误判成静音。
    """
    wav = _to_1d_float32(wav)

    if len(wav) == 0:
        return wav

    sample_active, _ = _active_masks_from_frame_rms(
        wav,
        fs=fs,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        silence_threshold_db=silence_threshold_db,
        energy_threshold_ratio=energy_threshold_ratio,
    )

    active_idx = np.flatnonzero(sample_active)
    if active_idx.size == 0:
        keep_len = min(len(wav), max(1, int(round(fs * min_keep_sec)))) if min_keep_sec > 0 else 0
        return wav[:keep_len].astype(np.float32).copy()

    start = int(active_idx[0])
    end = int(active_idx[-1]) + 1
    return wav[start:end].astype(np.float32).copy()


def compress_long_internal_silence(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    max_silence_sec: float = 0.3,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """
    帧级内部静音压缩。

    流程:
    1) 用短时帧 RMS 判定每一小段时间是否 active
    2) 将 active 帧映射到 sample 级 active mask
    3) 对连续静音段进行压缩：
       - 如果静音段不长，则完整保留（保留自然停顿）
       - 如果静音段太长，则只保留前 max_silence_sec
    """
    wav = _to_1d_float32(wav)
    if len(wav) == 0:
        return wav

    sample_active, _ = _active_masks_from_frame_rms(
        wav,
        fs=fs,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        silence_threshold_db=silence_threshold_db,
        energy_threshold_ratio=energy_threshold_ratio,
    )

    if sample_active.size == 0:
        return wav

    max_silence_len = max(0, int(round(max_silence_sec * fs)))
    runs = _find_runs(sample_active)

    kept_segments = []
    for is_active, start, end in runs:
        seg = wav[start:end]
        if is_active:
            kept_segments.append(seg)
            continue

        if max_silence_len <= 0:
            continue

        if len(seg) <= max_silence_len:
            kept_segments.append(seg)
        else:
            kept_segments.append(seg[:max_silence_len])

    if not kept_segments:
        return np.array([], dtype=np.float32)

    return np.concatenate(kept_segments, axis=0).astype(np.float32)


def preprocess_speech(
    wav,
    fs: int = 16000,
    trim_mode: str = "light",
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.3,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """
    统一预处理接口。

    trim_mode:
        - "none"  : 不处理静音
        - "light" : 只裁首尾静音（帧级）
        - "active": 裁首尾静音（帧级） + 压缩内部长静音（帧级）
    """
    wav = _to_1d_float32(wav)

    if trim_mode == "none":
        return wav

    if trim_mode == "light":
        return trim_edges_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
            frame_len_ms=frame_len_ms,
            hop_len_ms=hop_len_ms,
            energy_threshold_ratio=energy_threshold_ratio,
        )

    if trim_mode == "active":
        wav = trim_edges_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
            frame_len_ms=frame_len_ms,
            hop_len_ms=hop_len_ms,
            energy_threshold_ratio=energy_threshold_ratio,
        )
        wav = compress_long_internal_silence(
            wav,
            fs=fs,
            silence_threshold_db=silence_threshold_db,
            max_silence_sec=max_internal_silence_sec,
            frame_len_ms=frame_len_ms,
            hop_len_ms=hop_len_ms,
            energy_threshold_ratio=energy_threshold_ratio,
        )
        return wav

    raise ValueError("trim_mode must be one of: 'none', 'light', 'active'")


def _segment_activity_ratio(
    wav,
    fs: int = 16000,
    silence_threshold_db: float = -40.0,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """
    基于帧 RMS 的活跃度估计。

    返回:
        active_frames / total_frames

    这比按采样点阈值比例更贴近“说话时间占比”。
    """
    wav = _to_1d_float32(wav)
    if len(wav) == 0:
        return 0.0

    _, frame_active = _active_masks_from_frame_rms(
        wav,
        fs=fs,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        silence_threshold_db=silence_threshold_db,
        energy_threshold_ratio=energy_threshold_ratio,
    )
    if frame_active.size == 0:
        return 0.0
    return float(np.mean(frame_active.astype(np.float32)))


def _segment_rms_db(wav, eps: float = 1e-8):
    """计算片段整体 RMS(dB)。"""
    wav = _to_1d_float32(wav)
    rms = np.sqrt(np.mean(wav ** 2) + eps)
    return float(20.0 * np.log10(rms + eps))


def sample_speech(
    source_dir: str | Path,
    target_duration_sec: float = 20.0,
    fs: int = 16000,
    trim_mode: str = "light",
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.3,
    min_activity_ratio: float = 0.10,
    min_rms_db: float = -45.0,
    max_trials: int = 50,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """
    从长语音文件中随机裁出一段目标时长的语音。

    逻辑:
    1. 随机选文件
    2. 随机选起点
    3. 裁出目标时长片段
    4. 做静音预处理
    5. 如果预处理后片段太短、活跃度太低或 RMS 太低，则重试
    """
    rng = _make_rng(seed=seed, rng=rng)
    files = list_audio_files(source_dir)

    target_len = int(round(target_duration_sec * fs))

    last_error = None
    for _ in range(max_trials):
        try:
            file_path = files[int(rng.integers(0, len(files)))]
            info = sf.info(str(file_path))
            file_sr = info.samplerate
            total_frames = info.frames

            needed_frames = int(np.ceil(target_duration_sec * file_sr))
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

            if len(wav) < target_len:
                continue

            if len(wav) > target_len:
                max_start2 = len(wav) - target_len
                start2 = int(rng.integers(0, max_start2 + 1)) if max_start2 > 0 else 0
                wav = wav[start2:start2 + target_len]

            wav = wav.astype(np.float32)

            activity_ratio = _segment_activity_ratio(
                wav,
                fs=fs,
                silence_threshold_db=silence_threshold_db,
                frame_len_ms=frame_len_ms,
                hop_len_ms=hop_len_ms,
                energy_threshold_ratio=energy_threshold_ratio,
            )
            rms_db = _segment_rms_db(wav)

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
                "frame_len_ms": frame_len_ms,
                "hop_len_ms": hop_len_ms,
                "energy_threshold_ratio": energy_threshold_ratio,
                "activity_ratio": activity_ratio,
                "rms_db": rms_db,
                "start_sample_in_raw_file": start_sample,
            }
            return wav, meta

        except Exception as exc:
            last_error = exc
            continue

    msg = (
        "Failed to sample a valid speech segment. "
        "Please relax min_activity_ratio / min_rms_db / max_internal_silence_sec "
        "or check your dataset."
    )
    if last_error is not None:
        msg += f" Last error: {last_error}"
    raise RuntimeError(msg)


def sample_far_end(
    source_dir: str | Path,
    target_duration_sec: float = 15.0,
    fs: int = 16000,
    trim_mode: str = "active",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    max_internal_silence_sec: float = 0.3,
    silence_threshold_db: float = -40.0,
    min_activity_ratio: float = 0.10,
    min_rms_db: float = -45.0,
    max_trials: int = 50,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """far-end 采样包装器。"""
    return sample_speech(
        source_dir=source_dir,
        target_duration_sec=target_duration_sec,
        fs=fs,
        trim_mode=trim_mode,
        silence_threshold_db=silence_threshold_db,
        max_internal_silence_sec=max_internal_silence_sec,
        min_activity_ratio=min_activity_ratio,
        min_rms_db=min_rms_db,
        max_trials=max_trials,
        seed=seed,
        rng=rng,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        energy_threshold_ratio=energy_threshold_ratio,
    )


def sample_near_end(
    source_dir: str | Path,
    target_duration_sec: float = 20.0,
    fs: int = 16000,
    trim_mode: str = "active",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    max_internal_silence_sec: float = 0.3,
    silence_threshold_db: float = -40.0,
    min_activity_ratio: float = 0.15,
    min_rms_db: float = -45.0,
    max_trials: int = 50,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    energy_threshold_ratio: Optional[float] = None,
):
    """near-end 采样包装器。"""
    return sample_speech(
        source_dir=source_dir,
        target_duration_sec=target_duration_sec,
        fs=fs,
        trim_mode=trim_mode,
        silence_threshold_db=silence_threshold_db,
        max_internal_silence_sec=max_internal_silence_sec,
        min_activity_ratio=min_activity_ratio,
        min_rms_db=min_rms_db,
        max_trials=max_trials,
        seed=seed,
        rng=rng,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms,
        energy_threshold_ratio=energy_threshold_ratio,
    )


if __name__ == "__main__":
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
        trim_mode="active",
        seed=123,
    )

    print("far_end:", far_end.shape, far_end.dtype)
    print("near_end:", near_end.shape, near_end.dtype)
    print("far_meta:", far_meta)
    print("near_meta:", near_meta)
