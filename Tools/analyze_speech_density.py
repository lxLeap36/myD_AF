import os
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:\pyProject\myD_AF\Dataset")

WAVS = [
    ROOT / "clean_speech1" / "speaker01_23min.wav",
    ROOT / "clean_speech1" / "speaker03_24min.wav",
    ROOT / "clean_speech2" / "speaker02_19min.wav",
    ROOT / "clean_speech2" / "speaker04_23min.wav",
    ROOT / "clean_speech_test1" / "speaker05_25min.wav",
    ROOT / "clean_speech_test2" / "speaker06_24min.wav",
]

OUT_DIR = ROOT / "_density_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_MS = 25.0
HOP_MS = 10.0
THRESHOLD_DB = -40.0   # 相对峰值阈值，可试 -35 / -40 / -45
MIN_ACTIVE_FRAMES = 3  # 简单去毛刺
MIN_SILENT_FRAMES = 3


def to_mono_float32(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return np.squeeze(x).astype(np.float32)


def frame_signal(x, frame_len, hop_len):
    n = len(x)
    if n < frame_len:
        pad = frame_len - n
        x = np.pad(x, (0, pad))
        n = len(x)

    num_frames = 1 + (n - frame_len) // hop_len
    frames = np.stack(
        [x[i * hop_len : i * hop_len + frame_len] for i in range(num_frames)],
        axis=0
    )
    return frames


def compute_rms_db(x, sr, frame_ms=25.0, hop_ms=10.0):
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = int(round(sr * hop_ms / 1000.0))
    frames = frame_signal(x, frame_len, hop_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    times = (np.arange(len(rms_db)) * hop_len + frame_len / 2) / sr
    return rms_db, times, frame_len, hop_len


def smooth_binary_mask(mask, min_active=3, min_silent=3):
    mask = mask.astype(np.int32).copy()

    def fill_short_runs(arr, target_value, min_len):
        n = len(arr)
        i = 0
        while i < n:
            if arr[i] != target_value:
                i += 1
                continue
            j = i
            while j < n and arr[j] == target_value:
                j += 1
            if (j - i) < min_len:
                arr[i:j] = 1 - target_value
            i = j
        return arr

    # 先填短 active，再填短 silent
    mask = fill_short_runs(mask, 1, min_active)
    mask = fill_short_runs(mask, 0, min_silent)
    return mask.astype(np.int32)


def binary_runs(mask):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        v = mask[i]
        j = i
        while j < n and mask[j] == v:
            j += 1
        runs.append((v, i, j))
        i = j
    return runs


def analyze_one(wav_path):
    x, sr = sf.read(str(wav_path), always_2d=False)
    x = to_mono_float32(x)

    rms_db, times, frame_len, hop_len = compute_rms_db(x, sr, FRAME_MS, HOP_MS)

    peak = np.max(np.abs(x)) + 1e-12
    peak_db = 20.0 * np.log10(peak + 1e-12)
    vad_thr_db = peak_db + THRESHOLD_DB

    vad = (rms_db > vad_thr_db).astype(np.int32)
    vad = smooth_binary_mask(vad, MIN_ACTIVE_FRAMES, MIN_SILENT_FRAMES)

    runs = binary_runs(vad)
    speech_runs = []
    silence_runs = []

    frame_sec = hop_len / sr
    for v, s, e in runs:
        dur = (e - s) * frame_sec
        if v == 1:
            speech_runs.append(dur)
        else:
            silence_runs.append(dur)

    activity_ratio = float(np.mean(vad))

    stats = {
        "file": wav_path.name,
        "sr": sr,
        "duration_sec": len(x) / sr,
        "activity_ratio": activity_ratio,
        "num_speech_segments": len(speech_runs),
        "num_silence_segments": len(silence_runs),
        "mean_speech_sec": float(np.mean(speech_runs)) if speech_runs else 0.0,
        "mean_silence_sec": float(np.mean(silence_runs)) if silence_runs else 0.0,
        "median_silence_sec": float(np.median(silence_runs)) if silence_runs else 0.0,
        "p90_silence_sec": float(np.percentile(silence_runs, 90)) if silence_runs else 0.0,
        "max_silence_sec": float(np.max(silence_runs)) if silence_runs else 0.0,
    }

    return x, sr, rms_db, times, vad, stats, speech_runs, silence_runs


def save_visuals(wav_path, x, sr, rms_db, times, vad, silence_runs):
    name = wav_path.stem

    # 1) 波形 + RMS + VAD
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    t = np.arange(len(x)) / sr
    ax1.plot(t, x, linewidth=0.5)
    ax1.set_title(f"{name} - waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(times, rms_db, linewidth=1.0)
    ax2.set_title(f"{name} - frame RMS (dB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("dB")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.step(times, vad, where="mid")
    ax3.set_title(f"{name} - VAD mask")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Active")
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_overview.png", dpi=150)
    plt.close(fig)

    # 2) 停顿长度分布
    fig = plt.figure(figsize=(8, 4.5))
    if len(silence_runs) > 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(silence_runs, bins=40)
        ax.set_title(f"{name} - silence duration histogram")
        ax.set_xlabel("Silence duration (s)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_silence_hist.png", dpi=150)
    plt.close(fig)


def main():
    all_stats = []

    for wav_path in WAVS:
        if not wav_path.exists():
            print(f"[skip] not found: {wav_path}")
            continue

        x, sr, rms_db, times, vad, stats, speech_runs, silence_runs = analyze_one(wav_path)
        save_visuals(wav_path, x, sr, rms_db, times, vad, silence_runs)
        all_stats.append(stats)

    if not all_stats:
        print("No wav files found.")
        return

    # 打印汇总表
    print("\n===== Speech density summary =====")
    headers = [
        "file", "activity_ratio",
        "num_speech_segments", "num_silence_segments",
        "mean_speech_sec", "mean_silence_sec",
        "median_silence_sec", "p90_silence_sec", "max_silence_sec"
    ]
    print("\t".join(headers))
    for s in all_stats:
        print("\t".join([
            s["file"],
            f"{s['activity_ratio']:.4f}",
            str(s["num_speech_segments"]),
            str(s["num_silence_segments"]),
            f"{s['mean_speech_sec']:.3f}",
            f"{s['mean_silence_sec']:.3f}",
            f"{s['median_silence_sec']:.3f}",
            f"{s['p90_silence_sec']:.3f}",
            f"{s['max_silence_sec']:.3f}",
        ]))


if __name__ == "__main__":
    main()