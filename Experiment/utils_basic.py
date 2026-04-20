from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from Algorithms.LMS import LMSFilter
from Algorithms.NLMS import NLMSFilter
from Algorithms.RLS import RLSFilter

from Scenarios.farend_single_talk import generate_farend_single_talk
from Scenarios.noisy_single_talk import generate_noisy_single_talk
from Scenarios.double_talk import generate_double_talk
from Scenarios.path_change import generate_path_change

from Tools.data_loader import (
    sample_far_end,
    sample_near_end,
    list_audio_files,
    load_audio_segment,
)
from Tools.white_noise_generate import generate_white_noise
from Tools.impulse_noise_generate import generate_impulse_noise


def to_numpy_1d(x):
    """
    把 Tensor / list / numpy 标量/向量 都转成 1D numpy
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("Expected a 1-D signal after squeeze().")
    return x.astype(np.float32)


def load_audio_full(path: Union[str, Path], fs: int = 16000, mono: bool = True):
    """
    读取整条音频
    适合：
    - RIR
    - 较短的 babble wav
    """
    wav, sr = sf.read(str(path), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2 and mono:
        wav = np.mean(wav, axis=1)

    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        raise ValueError("Loaded audio must be 1-D after squeeze().")

    if sr != fs:
        wav = resample_poly(wav, up=fs, down=sr).astype(np.float32)

    return wav.astype(np.float32)


def find_first_audio_file(directory: Path):
    """
    找目录中的第一个音频文件
    """
    candidates = []
    for ext in [".wav", ".flac"]:
        candidates.extend(sorted(directory.rglob(f"*{ext}")))

    if len(candidates) == 0:
        raise RuntimeError(f"No audio files found in {directory}")

    return candidates[0]


def load_rir(cfg):
    """
    读取 RIR：
    - 如果 cfg["rir_path"] 指定了路径，优先使用
    - 否则取 rir_dir 下第一个音频文件
    """
    rir_path = cfg["rir_path"]

    if rir_path is None:
        rir_path = find_first_audio_file(Path(cfg["rir_dir"]))

    rir = load_audio_full(rir_path, fs=cfg["fs"], mono=True)
    return rir, str(rir_path)


def sample_babble_noise_segment(noise_dir, duration_sec, fs, seed=None):
    """
    从 babble_noise 目录中随机选一个文件，并裁一段指定时长
    """
    rng = np.random.default_rng(seed)
    files = list_audio_files(noise_dir)

    file_path = files[int(rng.integers(0, len(files)))]
    info = sf.info(str(file_path))
    file_sr = info.samplerate
    total_frames = info.frames

    needed_frames = int(np.ceil(duration_sec * file_sr))
    if total_frames < needed_frames:
        start_sample = 0
        needed_frames = total_frames
    else:
        max_start = total_frames - needed_frames
        start_sample = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

    noise = load_audio_segment(
        path=file_path,
        start_sample=start_sample,
        num_samples=needed_frames,
        fs=fs,
        mono=True,
    )

    meta = {
        "file_path": str(file_path),
        "start_sample_in_raw_file": start_sample,
        "duration_sec": duration_sec,
    }
    return noise.astype(np.float32), meta


def build_algorithm(name: str, cfg):
    """
    根据算法名构建算法实例
    """
    params = cfg["alg_params"][name]

    if name == "lms":
        return LMSFilter(
            filter_length=params["filter_length"],
            step_size=params["step_size"],
        )

    if name == "nlms":
        return NLMSFilter(
            filter_length=params["filter_length"],
            step_size=params["step_size"],
            epsilon=params["epsilon"],
        )

    if name == "rls":
        return RLSFilter(
            filter_length=params["filter_length"],
            lambda_=params["lambda_"],
            delta=params["delta"],
        )

    raise ValueError(f"Unsupported algorithm name: {name}")


def build_scenario(cfg):
    """
    根据配置构造 sample dict
    """
    seed = cfg["seed"]
    fs = cfg["fs"]
    duration_sec = cfg["duration_sec"]
    scenario_name = cfg["scenario_name"]

    far_end, far_meta = sample_far_end(
        source_dir=cfg["far_speech_dir"],
        target_duration_sec=duration_sec,
        fs=fs,
        trim_mode="active",
        seed=seed,
        max_internal_silence_sec=cfg["max_internal_silence_sec"],
    )

    near_end = None
    near_meta = None

    rir, rir_path = load_rir(cfg)

    extra_meta = {
        "far_meta": far_meta,
        "near_meta": near_meta,
        "rir_path": rir_path,
    }

    if scenario_name == "farend_single_talk":
        sample = generate_farend_single_talk(
            far_end=far_end,
            rir=rir,
            fs=fs,
            normalize=True,
            peak=0.95,
        )
        sample["meta"]["extra"] = extra_meta
        return sample

    if scenario_name == "noisy_single_talk":
        noise_type = cfg["noise_type"]

        if noise_type == "white":
            noise = generate_white_noise(
                duration_sec=duration_sec,
                fs=fs,
                noise_type="gaussian",
                seed=seed + 100,
            )
            noise_meta = {"noise_type": "white_gaussian"}

        elif noise_type == "impulse":
            noise = generate_impulse_noise(
                duration_sec=duration_sec,
                fs=fs,
                mode="point",
                seed=seed + 200,
            )
            noise_meta = {"noise_type": "impulse"}

        elif noise_type == "babble":
            noise, noise_meta = sample_babble_noise_segment(
                noise_dir=cfg["babble_noise_dir"],
                duration_sec=duration_sec,
                fs=fs,
                seed=seed + 300,
            )
            noise_meta["noise_type"] = "babble"

        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

        sample = generate_noisy_single_talk(
            far_end=far_end,
            rir=rir,
            noise=noise,
            snr_db=cfg["snr_db"],
            fs=fs,
            normalize=True,
            peak=0.95,
        )
        extra_meta["noise_meta"] = noise_meta
        sample["meta"]["extra"] = extra_meta
        return sample

    if scenario_name == "double_talk":
        near_end, near_meta = sample_near_end(
            source_dir=cfg["near_speech_dir"],
            target_duration_sec=duration_sec,
            fs=fs,
            trim_mode="active",
            seed=seed + 1,
            max_internal_silence_sec=cfg["max_internal_silence_sec"],
        )

        sample = generate_double_talk(
            far_end=far_end,
            near_end=near_end,
            rir=rir,
            fs=fs,
            ser_db=cfg["ser_db"],
            segments=None,
            normalize=True,
            peak=0.95,
        )
        extra_meta["near_meta"] = near_meta
        sample["meta"]["extra"] = extra_meta
        return sample

    if scenario_name == "path_change":
        rir_files = []
        for ext in [".wav", ".flac"]:
            rir_files.extend(sorted(Path(cfg["rir_dir"]).rglob(f"*{ext}")))

        if len(rir_files) >= 2:
            rir_before = load_audio_full(rir_files[0], fs=fs, mono=True)
            rir_after = load_audio_full(rir_files[1], fs=fs, mono=True)
            extra_meta["rir_before_path"] = str(rir_files[0])
            extra_meta["rir_after_path"] = str(rir_files[1])
        else:
            rir_before = rir
            rir_after = rir
            extra_meta["rir_before_path"] = rir_path
            extra_meta["rir_after_path"] = rir_path

        sample = generate_path_change(
            far_end=far_end,
            rir_before=rir_before,
            rir_after=rir_after,
            change_time_sec=cfg["change_time_sec"],
            fs=fs,
            normalize=True,
            peak=0.95,
        )
        sample["meta"]["extra"] = extra_meta
        return sample

    raise ValueError(f"Unsupported scenario_name: {scenario_name}")


def run_algorithm_on_sample(algo, sample):
    """
    运行算法，返回误差信号 e
    """
    algo.reset()
    e = algo.process(sample["x"], sample["d"])
    return to_numpy_1d(e)