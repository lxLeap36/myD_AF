from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import os

# 保证能从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Tools.set_seed import set_seed

from config_basic import CONFIG
from utils_basic import build_algorithm, build_scenario, run_algorithm_on_sample
from complexity_basic import (
    profile_algorithm_on_sample,
    get_static_complexity_info,
    disable_algorithm_history,
)
from evaluator_basic import (
    evaluate_sample,
    print_summary,
    print_comparison_table,
    save_comparison_table,
    plot_curves,
    plot_signal_comparison,
    plot_path_comparison,
    save_results,
)


def main():
    """
    运行一次完整的基线实验流程（单样本，多算法比较）。

    步骤：
    1) 通过 CONFIG 固定随机种子，保证可复现
    2) 创建用于保存结果的输出目录（Results/results_basic/<scenario>_timestamp）
    3) 根据场景配置构建相同的 sample（包含 x, d, y, h 等）
    4) 对 CONFIG["algorithms"] 中列出的每个算法：
       - 构建算法实例
       - 用 sample 运行算法，得到误差信号 e(n)
       - 评估该样本得到各种指标（ERLE, PESQ, 收敛曲线等）
       - 保存算法的最终估计权重和权重历史（如果算法有记录）到结果字典
    5) 打印结果汇总
    6) 将结果保存到磁盘（summary.json, results.npz）
    7) 基于评估结果绘图并保存图片

    设计契约（简短）：
    - 输入：全局 CONFIG（在本模块顶端导入），其它模块负责样本生成与算法实现
    - 输出：在磁盘上生成结果文件与图像；在内存中返回 None
    - 错误模式：找不到场景或算法名将抛出异常；IOError 会在写入时暴露
    """

    # 1. 固定随机种子（普通复现模式）
    set_seed(CONFIG["seed"])

    # 2. 创建结果输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "Results" / "results_white_noise_input_room01" / f"{CONFIG['scenario_name']}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 构造同一条 sample，保证三个算法公平比较
    print(f"开始构建场景...")
    sample = build_scenario(CONFIG)

    # 4. 三个算法分别跑
    print(f"开始处理...")
    results = {}
    for name in CONFIG["algorithms"]:
        # 构造算法实例（LMS/NLMS/RLS）
        algo = build_algorithm(name, CONFIG)
        # 运行算法得到误差信号 e(n)
        # 如果开启 complexity，则使用 profile_algorithm_on_sample：
        #   - 先 warmup
        #   - 再正式计时
        #   - 返回最后一次 timed run 的 e
        if CONFIG.get("complexity", {}).get("enable", False):
            e, complexity_info = profile_algorithm_on_sample(algo, sample, CONFIG)
        else:
            algo.reset()
            if not CONFIG.get("record_weight_history", False):
                disable_algorithm_history(algo)

            e = algo.process(sample["x"], sample["d"])
            complexity_info = get_static_complexity_info(algo)

        # 评估该样本（ERLE / 收敛曲线 / PESQ / SI-SDR 等）
        # 评估该样本（ERLE / 收敛曲线 / PESQ / SI-SDR 等）
        res = evaluate_sample(
            sample=sample,
            e=e,
            scenario_name=CONFIG["scenario_name"],
            fs=CONFIG["fs"],
            cfg=CONFIG,
        )

        # ===== 统一保存 AEC 输出和派生回声估计 =====
        # 对传统算法：
        #   e = d - y_hat
        #   所以 y_hat = d - e
        #
        # 对 DLHybrid：
        #   e = s_hat
        #   所以 y_hat = d - s_hat
        #
        # 两者形式上都可以写成：
        #   y_hat = d - e
        d = np.asarray(sample["d"], dtype=np.float32)
        e_np = np.asarray(e, dtype=np.float32)

        res["aec_output"] = e_np.copy()
        res["estimated_echo"] = (d - e_np).astype(np.float32)
        # ===== 保存复杂度信息 =====
        res["complexity"] = complexity_info

        # ===== 保存路径估计：传统算法有，DL 没有 =====
        weights = getattr(algo, "weights", None)
        if weights is not None:
            res["estimated_weights"] = np.asarray(weights, dtype=np.float32).copy()
        else:
            res["estimated_weights"] = None

        # ===== 保存权值历史：传统算法可能有，DL 没有 =====
        # ===== 保存权值历史：默认关闭，避免 OOM =====
        if CONFIG.get("record_weight_history", False):
            weight_history = getattr(algo, "weight_history", None)
            if weight_history is not None and len(weight_history) > 0:
                res["weight_history"] = np.asarray(weight_history, dtype=np.float32)
            else:
                res["weight_history"] = None
        else:
            res["weight_history"] = None

        # 将单个算法的评估结果放入 results
        results[name.upper()] = res

    # 5. 打印结果
    print(f"开始打印结果...")
    print_summary(results, sample, CONFIG)
    print_comparison_table(results)

    # 6. 保存结果
    print(f"开始保存结果...")
    save_results(results, sample, CONFIG, out_dir)
    save_comparison_table(results, out_dir)

    # 7. 画图
    print(f"开始绘图...")
    plot_curves(results, CONFIG, out_dir)
    plot_signal_comparison(results, sample, CONFIG, out_dir)
    plot_path_comparison(results, sample, CONFIG, out_dir)

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()