from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from Algorithms.TorchAdaptiveFilters import (
    TorchLMSFilter,
    TorchNLMSFilter,
    TorchRLSFilter,
)
from Algorithms.LMS import LMSFilter
from Algorithms.NLMS import NLMSFilter
from Algorithms.RLS import RLSFilter
from Algorithms.DLHybridAEC import DLHybridAEC

from Scenarios.farend_single_talk import generate_farend_single_talk
from Scenarios.noisy_single_talk import generate_noisy_single_talk
from Scenarios.double_talk import generate_double_talk
from Scenarios.path_change import generate_path_change
from Scenarios.common import sample_random_double_talk_segments

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


def sample_rir(cfg, seed=None):
    """
    读取 RIR：
    - 如果 cfg["rir_path"] 指定了路径，优先使用
    - 否则从 rir_dir 下随机采样一条 RIR
    """
    rir_path = cfg.get("rir_path", None)

    if rir_path is None:
        rir_files = list_audio_files(cfg["rir_dir"])
        if len(rir_files) == 0:
            raise RuntimeError(f"No RIR files found in {cfg['rir_dir']}")

        rng = np.random.default_rng(seed)
        rir_path = rir_files[int(rng.integers(0, len(rir_files)))]

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

    if name == "torch_lms":
        return TorchLMSFilter(
            filter_length=params["filter_length"],
            step_size=params["step_size"],
            device=params.get("device", cfg.get("device", "cuda")),
        )

    if name == "torch_nlms":
        return TorchNLMSFilter(
            filter_length=params["filter_length"],
            step_size=params["step_size"],
            epsilon=params["epsilon"],
            device=params.get("device", cfg.get("device", "cuda")),
        )

    if name == "torch_rls":
        return TorchRLSFilter(
            filter_length=params["filter_length"],
            lambda_=params["lambda_"],
            delta=params["delta"],
            device=params.get("device", cfg.get("device", "cuda")),
        )

    if name == "dl_hybrid":
        return DLHybridAEC(
            checkpoint_path=params["checkpoint_path"],
            device=params.get("device", cfg.get("device", "cuda")),
            stft_cfg=params.get("stft", None),
            beta_residual=params.get("beta_residual", None),
            strict=params.get("strict", True),
        )

    raise ValueError(f"Unsupported algorithm name: {name}")


def _get_preprocess_cfg(cfg, which: str):
    """
    读取 speech_preprocess 配置。
    which 取 "far" 或 "near"。

    如果配置里没有 speech_preprocess，则回退到旧逻辑默认值，
    这样旧的 config_basic.py / config_dl.py 不会直接失效。
    """
    speech_pp = cfg.get("speech_preprocess", {})
    pp = speech_pp.get(which, {})

    return {
        "trim_mode": pp.get("trim_mode", "active"),
        "silence_threshold_db": pp.get("silence_threshold_db", -40.0),
        "max_internal_silence_sec": pp.get(
            "max_internal_silence_sec",
            cfg.get("max_internal_silence_sec", 0.3),
        ),
        "min_activity_ratio": pp.get(
            "min_activity_ratio",
            0.10 if which == "far" else 0.15,
        ),
        "min_rms_db": pp.get("min_rms_db", -45.0),
        "max_trials": pp.get("max_trials", 50),
        "frame_len_ms": pp.get("frame_len_ms", 25.0),
        "hop_len_ms": pp.get("hop_len_ms", 10.0),
        "energy_threshold_ratio": pp.get("energy_threshold_ratio", None),
    }


def build_scenario(cfg):
    """
    根据配置构造 sample dict
    """
    seed = cfg["seed"]
    fs = cfg["fs"]
    duration_sec = cfg["duration_sec"]
    scenario_name = cfg["scenario_name"]

    far_pp = _get_preprocess_cfg(cfg, "far")
    near_pp = _get_preprocess_cfg(cfg, "near")

    far_end, far_meta = sample_far_end(
        source_dir=cfg["far_speech_dir"],
        target_duration_sec=duration_sec,
        fs=fs,
        trim_mode=far_pp["trim_mode"],
        seed=seed,
        max_internal_silence_sec=far_pp["max_internal_silence_sec"],
        silence_threshold_db=far_pp["silence_threshold_db"],
        min_activity_ratio=far_pp["min_activity_ratio"],
        min_rms_db=far_pp["min_rms_db"],
        max_trials=far_pp["max_trials"],
        frame_len_ms=far_pp["frame_len_ms"],
        hop_len_ms=far_pp["hop_len_ms"],
        energy_threshold_ratio=far_pp["energy_threshold_ratio"],
    )

    near_end = None
    near_meta = None

    rir, rir_path = sample_rir(cfg, seed=seed + 10)

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
            trim_mode=near_pp["trim_mode"],
            seed=seed + 1,
            max_internal_silence_sec=near_pp["max_internal_silence_sec"],
            silence_threshold_db=near_pp["silence_threshold_db"],
            min_activity_ratio=near_pp["min_activity_ratio"],
            min_rms_db=near_pp["min_rms_db"],
            max_trials=near_pp["max_trials"],
            frame_len_ms=near_pp["frame_len_ms"],
            hop_len_ms=near_pp["hop_len_ms"],
            energy_threshold_ratio=near_pp["energy_threshold_ratio"],
        )

        seg_cfg = cfg.get("double_talk_segment_cfg", {})
        seg_mode = seg_cfg.get("mode", "random")

        if seg_mode == "random":
            segments = sample_random_double_talk_segments(
                duration_sec=duration_sec,
                seed=seed + 20,
                num_dt_range=seg_cfg.get("num_dt_range", (1, 3)),
                total_dt_ratio_range=seg_cfg.get("total_dt_ratio_range", (0.35, 0.75)),
                min_dt_sec=seg_cfg.get("min_dt_sec", 0.30),
                min_fst_sec=seg_cfg.get("min_fst_sec", 0.15),
            )
        else:
            segments = None

        sample = generate_double_talk(
            far_end=far_end,
            near_end=near_end,
            rir=rir,
            fs=fs,
            ser_db=cfg["ser_db"],
            segments=segments,
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