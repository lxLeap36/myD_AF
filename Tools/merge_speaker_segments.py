from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf


# =========================
# 路径配置
# =========================
INPUT_DIR = Path(r"/Dataset/04")
OUTPUT_DIR = Path(r"D:\pyProject\myD_AF\Dataset\clean_speech2")
OUTPUT_NAME = "speaker04_2min.wav"


# =========================
# 参数配置
# =========================
SILENCE_THRESHOLD_DB = -40.0   # 静音阈值（相对峰值）
MAX_INTERNAL_SILENCE_SEC = 0.001 # 内部连续静音超过这个时长就裁短
MIN_KEEP_SEC = 0             # 如果整段都很安静，至少保留这么长
CROSSFADE_MS = 10.0            # 片段拼接处做一个很短的淡入淡出，减少爆音


def to_1d_float32(x: np.ndarray) -> np.ndarray:
    """
    转成 1D float32。
    如果是多通道，就简单平均成单通道。
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError("Audio must be 1-D after processing.")

    return x.astype(np.float32)


def trim_edges_silence(
    wav: np.ndarray,
    fs: int,
    silence_threshold_db: float = -40.0,
    min_keep_sec: float = 0,
) -> np.ndarray:
    """
    只裁掉首尾长静音，不动中间正常停顿。
    """
    wav = to_1d_float32(wav)

    if len(wav) == 0:
        return wav

    peak = np.max(np.abs(wav)) + 1e-8
    thr = peak * (10.0 ** (silence_threshold_db / 20.0))

    active_idx = np.where(np.abs(wav) > thr)[0]

    # 如果整段都接近静音，就至少保留前 min_keep_sec
    if len(active_idx) == 0:
        keep_len = min(len(wav), int(round(fs * min_keep_sec)))
        return wav[:keep_len].copy()

    start = int(active_idx[0])
    end = int(active_idx[-1]) + 1
    return wav[start:end].astype(np.float32)


def compress_long_internal_silence(
    wav: np.ndarray,
    fs: int,
    silence_threshold_db: float = -40.0,
    max_silence_sec: float = 0.001,
) -> np.ndarray:
    """
    压缩过长内部静音：
    - 短暂停顿保留
    - 超过 max_silence_sec 的连续静音裁短到这个长度
    """
    wav = to_1d_float32(wav)

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
            # 非静音逐样本保留
            out.append(wav[i])
            i += 1
        else:
            # 找到一整段连续静音
            j = i
            while j < n and silent_mask[j]:
                j += 1

            seg = wav[i:j]
            if len(seg) > max_silence_len:
                seg = seg[:max_silence_len]

            out.append(seg)
            i = j

    out = np.concatenate(
        [np.atleast_1d(x).astype(np.float32) for x in out],
        axis=0
    )
    return out.astype(np.float32)


def preprocess_clip(
    wav: np.ndarray,
    fs: int,
    silence_threshold_db: float = -40.0,
    max_internal_silence_sec: float = 0.001,
) -> np.ndarray:
    """
    单个片段预处理：
    1. 去首尾长静音
    2. 压缩内部过长静音
    """
    wav = trim_edges_silence(
        wav,
        fs=fs,
        silence_threshold_db=silence_threshold_db,
        min_keep_sec=MIN_KEEP_SEC,
    )

    wav = compress_long_internal_silence(
        wav,
        fs=fs,
        silence_threshold_db=silence_threshold_db,
        max_silence_sec=max_internal_silence_sec,
    )

    return wav.astype(np.float32)


def concat_with_crossfade(
    segments: list[np.ndarray],
    fs: int,
    crossfade_ms: float = 10.0,
) -> np.ndarray:
    """
    多段音频拼接。
    在边界做一个很短的交叉淡化，减少拼接处“啪”的爆音。
    """
    if len(segments) == 0:
        return np.zeros(0, dtype=np.float32)

    if len(segments) == 1:
        return to_1d_float32(segments[0])

    fade_len = int(round(fs * crossfade_ms / 1000.0))
    fade_len = max(0, fade_len)

    out = to_1d_float32(segments[0]).copy()

    for seg in segments[1:]:
        seg = to_1d_float32(seg)

        # 如果太短，就直接拼
        if fade_len == 0 or len(out) < fade_len or len(seg) < fade_len:
            out = np.concatenate([out, seg], axis=0).astype(np.float32)
            continue

        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

        mixed = out[-fade_len:] * fade_out + seg[:fade_len] * fade_in
        out = np.concatenate([out[:-fade_len], mixed, seg[fade_len:]], axis=0).astype(np.float32)

    return out.astype(np.float32)


def joint_peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    最后做一次整体峰值归一化，避免幅度过大。
    """
    wav = to_1d_float32(wav)

    max_amp = np.max(np.abs(wav)) + 1e-8
    wav = wav / max_amp * peak
    return wav.astype(np.float32)


def merge_speaker_segments(
    input_dir: Path,
    output_dir: Path,
    output_name: str,
) -> Path:
    """
    主函数：
    - 读取目录下所有 flac
    - 逐个预处理
    - 拼接成长音频
    - 写出 wav
    """
    flac_files = sorted(input_dir.glob("*.flac"))
    if len(flac_files) == 0:
        raise RuntimeError(f"No .flac files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    sample_rate = None

    for file_path in flac_files:
        wav, sr = sf.read(str(file_path), always_2d=False)
        wav = to_1d_float32(wav)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {file_path} has sr={sr}, expected {sample_rate}"
            )

        wav = preprocess_clip(
            wav,
            fs=sample_rate,
            silence_threshold_db=SILENCE_THRESHOLD_DB,
            max_internal_silence_sec=MAX_INTERNAL_SILENCE_SEC,
        )

        # 过滤空片段
        if len(wav) > 0:
            segments.append(wav)

    if len(segments) == 0:
        raise RuntimeError("All segments became empty after preprocessing.")

    merged = concat_with_crossfade(
        segments,
        fs=sample_rate,
        crossfade_ms=CROSSFADE_MS,
    )

    merged = joint_peak_normalize(merged, peak=0.95)

    output_path = output_dir / output_name
    sf.write(str(output_path), merged, sample_rate)

    total_sec = len(merged) / sample_rate
    print(f"Done.")
    print(f"Input clips : {len(flac_files)}")
    print(f"Output file : {output_path}")
    print(f"Sample rate : {sample_rate}")
    print(f"Duration    : {total_sec:.2f} sec ({total_sec/60:.2f} min)")

    return output_path


if __name__ == "__main__":
    merge_speaker_segments(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
    )