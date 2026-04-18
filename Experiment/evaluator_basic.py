import json
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
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


def evaluate_sample(sample, e, scenario_name, fs, cfg):
    """
    按场景正确解释 e，并计算指标
    """
    y = to_numpy_1d(sample["y"])
    s = to_numpy_1d(sample["s"])
    v = to_numpy_1d(sample["v"])

    # 不同场景下，原始误差 e 的含义不同
    if scenario_name in ["farend_single_talk", "path_change"]:
        residual_for_erle = e.copy()

    elif scenario_name == "noisy_single_talk":
        residual_for_erle = e - v

    elif scenario_name == "double_talk":
        residual_for_erle = e - s

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
        # try:
        #     pesq_value = tensor_to_float(compute_pesq(clean=s, enhanced=e, fs=fs))
        # except Exception:
        #     pesq_value = None
        try:
            pesq_value = tensor_to_float(compute_pesq(clean=s, enhanced=e, fs=fs))
        except Exception as ex:
            print(f"[PESQ ERROR] {type(ex).__name__}: {ex}")
            pesq_value = None

        try:
            si_sdr_value = tensor_to_float(compute_si_sdr(clean=s, enhanced=e))
        except Exception:
            si_sdr_value = None

    result = {
        "error_signal": e.astype(np.float32),
        "residual_for_erle": residual_for_erle.astype(np.float32),
        "erle": erle_value,
        "erle_curve": erle_curve.astype(np.float32),
        "convergence_curve_db": convergence_curve_db.astype(np.float32),
        "pesq": pesq_value,
        "si_sdr": si_sdr_value,
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
        if res["pesq"] is not None:
            print(f"  PESQ   : {res['pesq']:.4f}")
        if res["si_sdr"] is not None:
            print(f"  SI-SDR : {res['si_sdr']:.4f} dB")
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
    plt.show()

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
    plt.show()

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
            save_dict[f"{prefix}_estimated_weights"] = res["estimated_weights"]
            # 如果存在 weight_history（可能很大），我们也保存下来以便后续分析或绘图恢复
            if res.get("weight_history", None) is not None:
                # 转成 float32 array 保存，shape = (n_samples, filter_length)
                try:
                    save_dict[f"{prefix}_weight_history"] = np.asarray(res["weight_history"], dtype=np.float32)
                except Exception:
                    # 如果转化失败，跳过保存历史以避免写入错误
                    pass

        np.savez_compressed(out_dir / "results.npz", **save_dict)