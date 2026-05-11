import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # AutoDL / 服务器 / headless 环境必须用 Agg
import matplotlib.pyplot as plt

import numpy as np

from Metrics.erle_metric import compute_erle, compute_erle_curve
from Metrics.pesq_metric import compute_pesq
from Metrics.si_sdr_metric import compute_si_sdr
from Metrics.convergence_metric import compute_learning_curve_db


def tensor_to_float(x):
    """
    把 Tensor / ndarray / 标量 统一转成 python float
    """
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    x = np.asarray(x).squeeze()
    return float(x.item())


def to_numpy_1d(x):
    """
    把 Tensor / list / numpy 都转成 1D numpy
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("Expected a 1-D signal after squeeze().")
    return x.astype(np.float32)

def compute_residual_basic_metrics(residual, eps=1e-12, clip_abs=1e10):
    """
    远端单讲 / 残余回声场景下的基础指标。

    稳定版：
    - 用 float64 计算平方，避免 float32 直接 overflow
    - nan / inf 会先替换成有限值
    - 极端大值会裁剪，避免一个发散算法把整轮评估弄崩

    residual_rms:
        残余信号 RMS，越小越好。

    residual_mse:
        残余信号均方能量，越小越好。

    residual_dbfs:
        以 full-scale=1 为参考的 RMS dBFS。
    """
    residual_raw = np.asarray(residual).squeeze()

    if residual_raw.ndim != 1:
        raise ValueError(f"Expected 1-D residual, got shape {residual_raw.shape}")

    # 记录原始输出是否已经异常，方便后续判断是不是算法发散
    finite_mask = np.isfinite(residual_raw)
    finite_ratio = float(np.mean(finite_mask)) if residual_raw.size > 0 else 0.0

    if np.any(finite_mask):
        max_abs_raw = float(np.max(np.abs(residual_raw[finite_mask])))
    else:
        max_abs_raw = float("inf")

    # 用 float64 做后续计算，避免 float32 平方溢出
    residual64 = np.asarray(residual_raw, dtype=np.float64)

    # nan / inf 先替换成有限值
    residual64 = np.nan_to_num(
        residual64,
        nan=0.0,
        posinf=clip_abs,
        neginf=-clip_abs,
    )

    # 再裁剪极端发散值
    residual64 = np.clip(residual64, -clip_abs, clip_abs)

    residual_mse = float(np.mean(residual64 * residual64))
    residual_rms = float(np.sqrt(residual_mse + eps))
    residual_dbfs = float(20.0 * np.log10(residual_rms + eps))
    residual_peak = float(np.max(np.abs(residual64))) if residual64.size > 0 else 0.0

    # 如果原始值非有限，或者原始峰值已经超过裁剪阈值，就认为该输出异常
    residual_was_clipped = bool((finite_ratio < 1.0) or (max_abs_raw > clip_abs))

    return {
        "residual_mse": residual_mse,
        "residual_rms": residual_rms,
        "residual_dbfs": residual_dbfs,
        "residual_peak": residual_peak,

        # 诊断字段
        "residual_finite_ratio": finite_ratio,
        "residual_max_abs_raw": max_abs_raw,
        "residual_was_clipped": residual_was_clipped,
    }

def _is_valid_audio_for_objective_metric(x, *, min_rms=1e-6, max_abs=50.0):
    """
    判断一条音频是否适合送进 PESQ / SI-SDR 这类客观指标。

    返回:
        ok: bool
        reason: str
    """
    x = np.asarray(x, dtype=np.float32).squeeze()

    if x.ndim != 1:
        return False, f"not_1d_shape_{x.shape}"

    if x.size == 0:
        return False, "empty"

    if not np.all(np.isfinite(x)):
        return False, "has_nan_or_inf"

    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    peak = float(np.max(np.abs(x)))

    if rms < min_rms:
        return False, f"too_silent_rms_{rms:.3e}"

    if peak > max_abs:
        return False, f"too_large_peak_{peak:.3e}"

    return True, "ok"


def _safe_for_metric(x, *, peak=0.99):
    """
    仅用于 PESQ / SI-SDR 前的保护：
    - nan/inf -> 0
    - 如果峰值超过 peak，则整体缩放到 peak
    """
    x = np.asarray(x, dtype=np.float32).squeeze()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    max_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if max_abs > peak:
        x = x / (max_abs + 1e-12) * peak

    return x.astype(np.float32)

def evaluate_sample(sample, e, scenario_name, fs, cfg):
    """
    按场景正确解释 e，并计算指标
    """
    y = to_numpy_1d(sample["y"])
    s = to_numpy_1d(sample["s"])
    v = to_numpy_1d(sample["v"])

    # 不同场景下，原始误差 e 的含义不同
    if scenario_name in ["farend_single_talk", "path_change", "nonlinear_farend_single_talk"]:
        residual_for_erle = e.copy()
        residual_metrics = compute_residual_basic_metrics(residual_for_erle)

    elif scenario_name == "noisy_single_talk":
        residual_for_erle = e - v
        residual_metrics = compute_residual_basic_metrics(residual_for_erle)

    elif scenario_name == "double_talk":
        residual_for_erle = e - s
        residual_metrics = None

    else:
        raise ValueError(f"Unsupported scenario_name: {scenario_name}")

    # ===== 整体 ERLE =====
    erle_value = tensor_to_float(compute_erle(echo=y, residual=residual_for_erle))

    # ===== ERLE 曲线 =====
    erle_curve = compute_erle_curve(
        echo=y,
        residual=residual_for_erle,
        frame_size=cfg["erle_frame_size"],
        hop_size=cfg["erle_hop_size"],
    )
    erle_curve = to_numpy_1d(erle_curve)

    # ===== 收敛曲线 =====
    # 注意：这里用 residual_for_erle，而不是原始 e
    # 因为 noisy/double-talk 下，e 里混了噪声或近端语音
    convergence_curve_db = compute_learning_curve_db(
        residual_for_erle,
        window_size=cfg["curve_window_size"],
    )
    convergence_curve_db = to_numpy_1d(convergence_curve_db)

    pesq_value = None
    si_sdr_value = None

    # 第一版只在 double-talk 下把 e 当近端恢复输出
    if scenario_name == "double_talk":
        # PESQ / SI-SDR 只在输出是有效语音时计算。
        # 如果某个算法发散、静音、nan/inf，直接跳过，避免整轮实验被一个异常输出打断。
        metric_e = _safe_for_metric(e)
        metric_s = _safe_for_metric(s)

        ok_s, reason_s = _is_valid_audio_for_objective_metric(metric_s)
        ok_e, reason_e = _is_valid_audio_for_objective_metric(metric_e)

        if ok_s and ok_e:
            try:
                pesq_value = tensor_to_float(compute_pesq(clean=metric_s, enhanced=metric_e, fs=fs))
            except Exception as ex:
                print(f"[PESQ ERROR] {type(ex).__name__}: {ex}")
                pesq_value = None

            try:
                si_sdr_value = tensor_to_float(compute_si_sdr(clean=metric_s, enhanced=metric_e))
            except Exception as ex:
                print(f"[SI-SDR ERROR] {type(ex).__name__}: {ex}")
                si_sdr_value = None
        else:
            print(
                f"[METRIC SKIP] PESQ/SI-SDR skipped. "
                f"clean_valid={ok_s}({reason_s}), enhanced_valid={ok_e}({reason_e})"
            )
            pesq_value = None
            si_sdr_value = None

    result = {
        "error_signal": e.astype(np.float32),
        "residual_for_erle": residual_for_erle.astype(np.float32),
        "erle": erle_value,
        "erle_curve": erle_curve.astype(np.float32),
        "convergence_curve_db": convergence_curve_db.astype(np.float32),
        "pesq": pesq_value,
        "si_sdr": si_sdr_value,

        # 新增：远端单讲 / 残余回声指标
        "residual_mse": residual_metrics["residual_mse"],
        "residual_rms": residual_metrics["residual_rms"],
        "residual_dbfs": residual_metrics["residual_dbfs"],
        "residual_peak": residual_metrics["residual_peak"],

        # 新增：诊断字段，用来判断是否发散 / 裁剪
        "residual_finite_ratio": residual_metrics["residual_finite_ratio"],
        "residual_max_abs_raw": residual_metrics["residual_max_abs_raw"],
        "residual_was_clipped": residual_metrics["residual_was_clipped"],
    }

    return result

def _pad_or_crop_pair(a, b, max_len=None):
    """
    为了画路径对比图，把真实路径和估计路径整理成可直接对比的同长度数组。

    规则：
    - 先转成 1D numpy
    - 默认补零到同长度
    - 如果给了 max_len，则最多只保留前 max_len 个点
    """
    a = to_numpy_1d(a)
    b = to_numpy_1d(b)

    target_len = max(len(a), len(b))
    if max_len is not None:
        target_len = min(target_len, int(max_len))

    a_out = np.zeros(target_len, dtype=np.float32)
    b_out = np.zeros(target_len, dtype=np.float32)

    a_copy_len = min(len(a), target_len)
    b_copy_len = min(len(b), target_len)

    a_out[:a_copy_len] = a[:a_copy_len]
    b_out[:b_copy_len] = b[:b_copy_len]

    return a_out, b_out


def _slice_signal_for_plot(x, fs, max_sec=None):
    """
    信号画图时可选只截前 max_sec 秒，避免整段太密。
    """
    x = to_numpy_1d(x)

    if max_sec is None:
        return x

    n = min(len(x), int(round(max_sec * fs)))
    return x[:n]

def print_summary(results, sample, cfg):
    """
    打印实验结果概览
    """
    print("=" * 60)
    print(f"Scenario: {cfg['scenario_name']}")
    print(f"Signal length: {len(sample['x'])} samples ({len(sample['x']) / cfg['fs']:.2f} s)")
    print("=" * 60)

    for alg_name, res in results.items():
        print(f"[{alg_name}]")
        print(f"  ERLE   : {res['erle']:.4f} dB")
        if res.get("residual_rms", None) is not None:
            print(f"  Residual RMS  : {res['residual_rms']:.6e}")

        if res.get("residual_dbfs", None) is not None:
            print(f"  Residual dBFS : {res['residual_dbfs']:.4f} dB")

        if res.get("residual_mse", None) is not None:
            print(f"  Residual MSE  : {res['residual_mse']:.6e}")

        if res["pesq"] is not None:
            print(f"  PESQ   : {res['pesq']:.4f}")

        if res["si_sdr"] is not None:
            print(f"  SI-SDR : {res['si_sdr']:.4f} dB")

        comp = res.get("complexity", None)
        if comp is not None:
            param_count = comp.get("param_count", None)
            trainable_param_count = comp.get("trainable_param_count", None)
            state_count = comp.get("state_count", None)

            if param_count is not None:
                print(f"  Params : {param_count}")

            if trainable_param_count is not None and trainable_param_count != param_count:
                print(f"  Trainable params : {trainable_param_count}")

            if state_count is not None:
                print(f"  Runtime states   : {state_count}")

            if comp.get("time_median_ms", None) is not None:
                print(
                    f"  Time   : median={comp['time_median_ms']:.3f} ms, "
                    f"mean={comp['time_mean_ms']:.3f} ms, "
                    f"RTF={comp['rtf_median']:.4f}"
                )

            if comp.get("cuda_peak_memory_mb", None) is not None:
                print(
                    f"  CUDA peak memory : {comp['cuda_peak_memory_mb']:.2f} MB "
                    f"(extra {comp['cuda_extra_peak_memory_mb']:.2f} MB)"
                )

        print("-" * 40)


def plot_curves(results, cfg, out_dir: Path):
    """
    画两张图：
    1. 收敛曲线（dB）
    2. ERLE 曲线（dB）
    """
    fs = cfg["fs"]
    scenario_name = cfg["scenario_name"]

    # ===== 图1：收敛曲线 =====
    plt.figure(figsize=(10, 4.5))
    for alg_name, res in results.items():
        curve = res["convergence_curve_db"]
        t = np.arange(len(curve)) / fs
        plt.plot(t, curve, label=alg_name)

    if scenario_name == "path_change":
        plt.axvline(cfg["change_time_sec"], linestyle="--", label="path change")

    plt.xlabel("Time (s)")
    plt.ylabel("Learning Curve (dB)")
    plt.title(f"Convergence Curve - {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if cfg["save_fig"]:
        plt.savefig(out_dir / "convergence_curve.png", dpi=200)
    plt.close()

    # ===== 图2：ERLE 曲线 =====
    plt.figure(figsize=(10, 4.5))
    for alg_name, res in results.items():
        curve = res["erle_curve"]
        frame_size = cfg["erle_frame_size"]
        hop_size = cfg["erle_hop_size"]
        t = (np.arange(len(curve)) * hop_size + frame_size / 2) / fs
        plt.plot(t, curve, label=alg_name)

    if scenario_name == "path_change":
        plt.axvline(cfg["change_time_sec"], linestyle="--", label="path change")

    plt.xlabel("Time (s)")
    plt.ylabel("ERLE (dB)")
    plt.title(f"ERLE Curve - {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if cfg["save_fig"]:
        plt.savefig(out_dir / "erle_curve.png", dpi=200)
    plt.close()

def plot_signal_comparison(results, sample, cfg, out_dir: Path):
    """
    每个算法各画一张信号对比图：
    - far-end x(n)
    - true echo y(n)
    - microphone d(n)
    - AEC output / error e(n)

    说明：
    - x / y / d 对同一个 sample 来说是相同的
    - 但 e(n) 会因算法不同而不同
    """
    if not cfg.get("plot_signal_waveforms", True):
        return

    fs = cfg["fs"]
    max_sec = cfg.get("signal_plot_max_sec", None)

    x = _slice_signal_for_plot(sample["x"], fs=fs, max_sec=max_sec)
    y = _slice_signal_for_plot(sample["y"], fs=fs, max_sec=max_sec)
    d = _slice_signal_for_plot(sample["d"], fs=fs, max_sec=max_sec)

    for alg_name, res in results.items():
        e = _slice_signal_for_plot(res["error_signal"], fs=fs, max_sec=max_sec)

        n = min(len(x), len(y), len(d), len(e))
        t = np.arange(n) / fs

        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(t, x[:n])
        axes[0].set_ylabel("Amp")
        axes[0].set_title("Far-end signal x(n)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, y[:n])
        axes[1].set_ylabel("Amp")
        axes[1].set_title("True echo y(n)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, d[:n])
        axes[2].set_ylabel("Amp")
        axes[2].set_title("Microphone signal d(n)")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, e[:n])
        axes[3].set_ylabel("Amp")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_title(f"AEC output / error signal e(n) - {alg_name}")
        axes[3].grid(True, alpha=0.3)

        fig.suptitle(f"Signal Comparison - {alg_name}", fontsize=14)
        fig.tight_layout()

        if cfg["save_fig"]:
            fig.savefig(out_dir / f"signal_compare_{alg_name.lower()}.png", dpi=200)

        plt.close(fig)

def plot_path_comparison(results, sample, cfg, out_dir: Path):
    """
    画回声路径（RIR）与算法最终估计路径（weights）的对比图。

    固定路径场景：
        - 真实 h vs 估计 w

    path_change 场景：
        - subplot 1: h_before vs w_est
        - subplot 2: h_after  vs w_est

    说明：
    - 对线性 AEC 基线而言，最终估计路径直接取 algo.weights
    - 对 noisy / double-talk 场景，估计会更容易有偏差，但仍可画
    """
    if not cfg.get("plot_path_compare", True):
        return

    max_len = cfg.get("path_plot_max_len", None)
    scenario_name = cfg["scenario_name"]
    h = sample["h"]

    for alg_name, res in results.items():
        # 如果没有最终估计权重，跳过
        if "estimated_weights" not in res or res["estimated_weights"] is None:
            continue

        # 最终估计权重（fallback）
        w_est = to_numpy_1d(res["estimated_weights"])

        # 如果算法有记录权重历史（每个样本的权重快照），我们可以从中抽取变换前的估计路径
        weight_history = res.get("weight_history", None)
        if weight_history is not None:
            # weight_history 可能已经是 numpy array，也可能是 list；统一成 numpy
            weight_history = np.asarray(weight_history)

        # ===== path_change：前后两条真实路径 =====
        if scenario_name == "path_change" and isinstance(h, dict):
            h_before = h["before"]
            h_after = h["after"]

            # 默认：在路径改变处使用 weight_history 中改变前的快照作为 "before" 的估计路径，
            # 而使用最终的 weights 作为 "after" 的估计路径（即算法在改变后重新学习的结果）。
            if weight_history is not None and weight_history.ndim >= 2:
                fs = cfg["fs"]
                change_time = float(cfg.get("change_time_sec", 0.0))
                change_idx = int(round(change_time * fs))
                # 对索引做边界检查，取改变前的最后一个样本的权重（或最接近的可用样本）
                pre_idx = max(min(change_idx - 1, weight_history.shape[0] - 1), 0)
                w_before = weight_history[pre_idx]
                # 估计改变后的路径我们保留最终权重（w_est）作为 after 的估计
                w_after = w_est
            else:
                # 无 weight_history 时，退回到原始行为：用最终权重绘制两个子图（仍然有信息，但无法展示改变前学习情况）
                w_before = w_est
                w_after = w_est

            h_before_plot, w_before_plot = _pad_or_crop_pair(h_before, w_before, max_len=max_len)
            h_after_plot, w_after_plot = _pad_or_crop_pair(h_after, w_after, max_len=max_len)

            x1 = np.arange(len(h_before_plot))
            x2 = np.arange(len(h_after_plot))

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

            axes[0].plot(x1, h_before_plot, label="True path before change")
            axes[0].plot(x1, w_before_plot, label="Estimated path", linestyle="--")
            axes[0].set_title(f"Path Comparison (Before Change) - {alg_name}")
            axes[0].set_xlabel("Tap index")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(x2, h_after_plot, label="True path after change")
            axes[1].plot(x2, w_after_plot, label="Estimated path", linestyle="--")
            axes[1].set_title(f"Path Comparison (After Change) - {alg_name}")
            axes[1].set_xlabel("Tap index")
            axes[1].set_ylabel("Amplitude")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()

            if cfg["save_fig"]:
                fig.savefig(out_dir / f"path_compare_{alg_name.lower()}.png", dpi=200)

            plt.close(fig)
            continue

        # ===== 其它固定路径场景 =====
        true_h, est_h = _pad_or_crop_pair(h, w_est, max_len=max_len)
        tap_idx = np.arange(len(true_h))

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        ax.plot(tap_idx, true_h, label="True path (RIR)")
        ax.plot(tap_idx, est_h, label="Estimated path", linestyle="--")
        ax.set_title(f"Path Comparison - {alg_name}")
        ax.set_xlabel("Tap index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if cfg["save_fig"]:
            fig.savefig(out_dir / f"path_compare_{alg_name.lower()}.png", dpi=200)

        plt.close(fig)

def _format_table_value(value, digits=4):
    """
    把 None / 数值 / 字符串统一转成表格里的字符串。
    """
    if value is None:
        return "-"

    try:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
    except Exception:
        pass

    return str(value)


def _format_count(value):
    """
    参数量 / 状态量用千分位显示。
    """
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def build_comparison_table_rows(results):
    """
    把 results 整理成表格行。

    每一行对应一个算法：
        Method, ERLE, PESQ, SI-SDR, Params, States, Time, RTF, Memory, Device
    """
    rows = []

    for alg_name, res in results.items():
        comp = res.get("complexity", {}) or {}

        row = {
            "Method": alg_name,
            "ERLE(dB)": _format_table_value(res.get("erle", None), digits=4),

            # far-end single-talk 下 PESQ / SI-SDR 通常为空，这是正常的
            "PESQ": _format_table_value(res.get("pesq", None), digits=4),
            "SI-SDR(dB)": _format_table_value(res.get("si_sdr", None), digits=4),

            # 新增：远端单讲更应该看的残余回声指标
            "Residual RMS": _format_table_value(res.get("residual_rms", None), digits=6),
            "Residual dBFS": _format_table_value(res.get("residual_dbfs", None), digits=2),
            "Residual MSE": _format_table_value(res.get("residual_mse", None), digits=6),
            "Residual Peak": _format_table_value(res.get("residual_peak", None), digits=6),
            "Residual Clipped": _format_table_value(res.get("residual_was_clipped", None), digits=4),

            "Params": _format_count(comp.get("param_count", None)),
            "States": _format_count(comp.get("state_count", None)),

            "Median Time(ms)": _format_table_value(comp.get("time_median_ms", None), digits=3),
            "Mean Time(ms)": _format_table_value(comp.get("time_mean_ms", None), digits=3),
            "RTF": _format_table_value(comp.get("rtf_median", None), digits=5),

            "CUDA Extra Mem(MB)": _format_table_value(
                comp.get("cuda_extra_peak_memory_mb", None),
                digits=2,
            ),
            "CUDA Peak Mem(MB)": _format_table_value(
                comp.get("cuda_peak_memory_mb", None),
                digits=2,
            ),
            "Device": _format_table_value(comp.get("device", None), digits=4),
        }

        rows.append(row)

    return rows


def make_markdown_table(rows):
    """
    生成 Markdown 表格字符串。
    """
    if len(rows) == 0:
        return ""

    headers = list(rows[0].keys())

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    return "\n".join(lines)


def print_comparison_table(results):
    """
    在终端打印统一对比表。
    """
    rows = build_comparison_table_rows(results)
    table = make_markdown_table(rows)

    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)

    if table:
        print(table)
    else:
        print("No rows to display.")

    print("=" * 60 + "\n")


def save_comparison_table(results, out_dir: Path):
    """
    保存：
        comparison_table.md
        comparison_table.csv
    """
    rows = build_comparison_table_rows(results)

    if len(rows) == 0:
        return

    # ===== 保存 Markdown =====
    md_table = make_markdown_table(rows)
    md_path = out_dir / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)
        f.write("\n")

    # ===== 保存 CSV =====
    csv_path = out_dir / "comparison_table.csv"
    headers = list(rows[0].keys())

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def save_results(results, sample, cfg, out_dir: Path):
    """
    保存：
    - summary.json
    - results.npz
    """
    if cfg["save_summary_json"]:
        summary = {
            "config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in cfg.items()
                if k != "alg_params"
            },
            "alg_params": cfg["alg_params"],
            "sample_meta": sample["meta"],
            "metrics": {},
        }

        for alg_name, res in results.items():
            summary["metrics"][alg_name] = {
                "erle": res["erle"],
                "pesq": res["pesq"],
                "si_sdr": res["si_sdr"],

                "residual_mse": res.get("residual_mse", None),
                "residual_rms": res.get("residual_rms", None),
                "residual_dbfs": res.get("residual_dbfs", None),
                "residual_peak": res.get("residual_peak", None),

                "complexity": res.get("complexity", None),
            }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if cfg["save_npz"]:
        save_dict = {
            "x": np.asarray(sample["x"], dtype=np.float32),
            "d": np.asarray(sample["d"], dtype=np.float32),
            "y": np.asarray(sample["y"], dtype=np.float32),
            "s": np.asarray(sample["s"], dtype=np.float32),
            "v": np.asarray(sample["v"], dtype=np.float32),
        }

        for alg_name, res in results.items():
            prefix = alg_name.lower()

            save_dict[f"{prefix}_error"] = res["error_signal"]
            save_dict[f"{prefix}_residual_for_erle"] = res["residual_for_erle"]
            save_dict[f"{prefix}_erle_curve"] = res["erle_curve"]
            save_dict[f"{prefix}_convergence_curve_db"] = res["convergence_curve_db"]

            if res.get("residual_mse", None) is not None:
                save_dict[f"{prefix}_residual_mse"] = np.asarray(
                    res["residual_mse"], dtype=np.float32
                )

            if res.get("residual_rms", None) is not None:
                save_dict[f"{prefix}_residual_rms"] = np.asarray(
                    res["residual_rms"], dtype=np.float32
                )

            if res.get("residual_dbfs", None) is not None:
                save_dict[f"{prefix}_residual_dbfs"] = np.asarray(
                    res["residual_dbfs"], dtype=np.float32
                )

            if res.get("residual_peak", None) is not None:
                save_dict[f"{prefix}_residual_peak"] = np.asarray(
                    res["residual_peak"], dtype=np.float32
                )

            # 新增：统一保存 AEC 输出
            if res.get("aec_output", None) is not None:
                save_dict[f"{prefix}_aec_output"] = np.asarray(
                    res["aec_output"], dtype=np.float32
                )

            # 新增：统一保存派生回声估计 y_hat = d - e
            if res.get("estimated_echo", None) is not None:
                save_dict[f"{prefix}_estimated_echo"] = np.asarray(
                    res["estimated_echo"], dtype=np.float32
                )

            # 传统 LMS/NLMS/RLS 有 estimated_weights，DLHybrid 没有
            if res.get("estimated_weights", None) is not None:
                save_dict[f"{prefix}_estimated_weights"] = np.asarray(
                    res["estimated_weights"], dtype=np.float32
                )
            # 如果存在 weight_history（可能很大），我们也保存下来以便后续分析或绘图恢复
            if res.get("weight_history", None) is not None:
                # 转成 float32 array 保存，shape = (n_samples, filter_length)
                try:
                    save_dict[f"{prefix}_weight_history"] = np.asarray(res["weight_history"], dtype=np.float32)
                except Exception:
                    # 如果转化失败，跳过保存历史以避免写入错误
                    pass

        np.savez_compressed(out_dir / "results.npz", **save_dict)