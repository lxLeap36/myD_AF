# -*- coding: utf-8 -*-
"""
Complexity / runtime profiler for basic AEC platform.

统计内容：
1. 参数量：
   - LMS/NLMS/RLS：自适应滤波器权重个数 len(weights)
   - DLHybrid：神经网络 trainable parameters

2. 运行状态量：
   - LMS/NLMS：weights + input buffer
   - RLS：weights + input buffer + P matrix
   - DLHybrid：model parameters / buffers

3. 推理时间：
   - 先 warmup，不计时
   - 再 timed repeats，统计 mean / median / std / min / max
   - CUDA 下计时前后 synchronize，避免异步执行导致时间不准

4. 显存：
   - CUDA 下记录 peak memory
   - 同时记录相对 baseline 的 extra peak memory
"""

import gc
import time
from typing import Dict, Tuple, Any

import numpy as np

try:
    import torch
except Exception:
    torch = None

class NoOpHistory:
    """
    用来替代 weight_history / error_history 的轻量对象。
    算法内部仍然可以调用 .append(...)，但不会真正存东西。
    """
    def append(self, item):
        pass

    def clear(self):
        pass

    def __len__(self):
        return 0


def disable_algorithm_history(algo):
    """
    禁用算法内部逐采样点历史记录，避免 OOM。

    对 LMS / NLMS / RLS：
        weight_history.append(...)
        error_history.append(...)

    会被替换成 NoOpHistory.append(...)，不再真正保存数组。
    """
    if hasattr(algo, "weight_history"):
        algo.weight_history = NoOpHistory()

    if hasattr(algo, "error_history"):
        algo.error_history = NoOpHistory()

def _to_numpy_1d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {x.shape}")
    return x.astype(np.float32)


def _get_algo_device(algo):
    """
    返回算法设备字符串。
    传统算法没有 device，就认为是 cpu。
    """
    dev = getattr(algo, "device", None)
    if dev is not None:
        return str(dev)

    dev_name = getattr(algo, "device_name", None)
    if dev_name is not None:
        return str(dev_name)

    return "cpu"


def _get_cuda_device_if_any(algo):
    """
    如果算法在 CUDA 上运行，返回 torch.device；否则返回 None。
    """
    if torch is None or not torch.cuda.is_available():
        return None

    dev = getattr(algo, "device", None)
    if dev is not None:
        dev = torch.device(dev)
        if dev.type == "cuda":
            return dev

    dev_name = getattr(algo, "device_name", None)
    if dev_name is not None:
        dev = torch.device(dev_name)
        if dev.type == "cuda":
            return dev

    return None


def _cuda_synchronize(device):
    if torch is not None and device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def get_static_complexity_info(algo) -> Dict[str, Any]:
    """
    不运行算法，只统计静态复杂度信息。
    """
    info = {
        "device": _get_algo_device(algo),

        # 对传统算法：param_count = len(weights)
        # 对深度模型：param_count = trainable parameters
        "param_count": None,
        "trainable_param_count": None,

        # 运行时状态量，比如 RLS 的 P 矩阵。
        "state_count": None,

        # 估计的参数/状态内存，不等于真实运行显存。
        "param_memory_mb": None,
        "state_memory_mb": None,
    }

    # ===== 深度模型 =====
    model = getattr(algo, "model", None)
    if model is not None and torch is not None:
        trainable = 0
        total = 0
        param_bytes = 0

        for p in model.parameters():
            n = p.numel()
            total += n
            param_bytes += n * p.element_size()
            if p.requires_grad:
                trainable += n

        buffer_count = 0
        buffer_bytes = 0
        for b in model.buffers():
            n = b.numel()
            buffer_count += n
            buffer_bytes += n * b.element_size()

        info["param_count"] = int(total)
        info["trainable_param_count"] = int(trainable)
        info["state_count"] = int(buffer_count)
        info["param_memory_mb"] = float(param_bytes / (1024 ** 2))
        info["state_memory_mb"] = float(buffer_bytes / (1024 ** 2))
        return info

    # ===== 传统算法 =====
    weights = getattr(algo, "weights", None)
    if weights is not None:
        w = np.asarray(weights)
        info["param_count"] = int(w.size)
        info["trainable_param_count"] = int(w.size)
        info["param_memory_mb"] = float(w.nbytes / (1024 ** 2))

    state_count = 0
    state_bytes = 0

    for attr in ["weights", "input_buffer", "x_buffer", "P"]:
        val = getattr(algo, attr, None)
        if val is None:
            continue
        arr = np.asarray(val)
        state_count += int(arr.size)
        state_bytes += int(arr.nbytes)

    info["state_count"] = int(state_count)
    info["state_memory_mb"] = float(state_bytes / (1024 ** 2))

    return info


def profile_algorithm_on_sample(algo, sample, cfg) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    对一个算法在同一条 sample 上做：
    - warmup
    - timed repeats
    - 返回最后一次 timed run 的输出 e
    - 返回复杂度统计 dict

    注意：
    每次 warmup / timed run 之前都会 algo.reset()，
    所以 LMS/NLMS/RLS 不会因为重复运行而“接着上一次权重继续学”。
    """
    complexity_cfg = cfg.get("complexity", {})

    warmup_runs = int(complexity_cfg.get("warmup_runs", 2))
    timed_runs = int(complexity_cfg.get("timed_runs", 5))
    measure_cuda_memory = bool(complexity_cfg.get("measure_cuda_memory", True))

    warmup_runs = max(warmup_runs, 0)
    timed_runs = max(timed_runs, 1)

    x = _to_numpy_1d(sample["x"])
    d = _to_numpy_1d(sample["d"])

    static_info = get_static_complexity_info(algo)

    cuda_device = _get_cuda_device_if_any(algo)
    use_cuda_measure = measure_cuda_memory and cuda_device is not None

    # 尽量减少 Python 垃圾对象对测量的影响。
    gc.collect()

    # ===== Warmup：不计时 =====
    # 这里会触发 CUDA context / cuDNN / 内存池等初始化，
    # 避免把“第一次冷启动”算进推理时间。
    for _ in range(warmup_runs):
        algo.reset()
        disable_algorithm_history(algo)
        _ = algo.process(x, d)
        _cuda_synchronize(cuda_device)

    times_sec = []
    e_last = None

    cuda_peak_memory_mb_list = []
    cuda_extra_memory_mb_list = []

    # ===== Timed runs =====
    for _ in range(timed_runs):
        algo.reset()
        disable_algorithm_history(algo)

        if use_cuda_measure:
            _cuda_synchronize(cuda_device)
            baseline_alloc = torch.cuda.memory_allocated(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        else:
            baseline_alloc = None

        _cuda_synchronize(cuda_device)
        t0 = time.perf_counter()

        e_last = algo.process(x, d)

        _cuda_synchronize(cuda_device)
        t1 = time.perf_counter()

        times_sec.append(t1 - t0)

        if use_cuda_measure:
            peak_alloc = torch.cuda.max_memory_allocated(cuda_device)
            extra_peak = max(0, peak_alloc - baseline_alloc)

            cuda_peak_memory_mb_list.append(float(peak_alloc / (1024 ** 2)))
            cuda_extra_memory_mb_list.append(float(extra_peak / (1024 ** 2)))

    times_np = np.asarray(times_sec, dtype=np.float64)

    signal_sec = len(x) / float(cfg["fs"])

    runtime_info = {
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),

        "time_mean_ms": float(np.mean(times_np) * 1000.0),
        "time_median_ms": float(np.median(times_np) * 1000.0),
        "time_std_ms": float(np.std(times_np) * 1000.0),
        "time_min_ms": float(np.min(times_np) * 1000.0),
        "time_max_ms": float(np.max(times_np) * 1000.0),

        # real-time factor:
        # rtf < 1 表示处理速度快于实时。
        "rtf_mean": float(np.mean(times_np) / signal_sec),
        "rtf_median": float(np.median(times_np) / signal_sec),

        "signal_length": int(len(x)),
        "signal_sec": float(signal_sec),
    }

    if use_cuda_measure:
        runtime_info["cuda_peak_memory_mb"] = float(np.max(cuda_peak_memory_mb_list))
        runtime_info["cuda_extra_peak_memory_mb"] = float(np.max(cuda_extra_memory_mb_list))
    else:
        runtime_info["cuda_peak_memory_mb"] = None
        runtime_info["cuda_extra_peak_memory_mb"] = None

    info = {}
    info.update(static_info)
    info.update(runtime_info)

    return _to_numpy_1d(e_last), info