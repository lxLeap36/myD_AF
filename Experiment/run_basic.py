from datetime import datetime
from pathlib import Path
import sys
import numpy as np

# 保证能从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Tools.set_seed import set_seed

from config_basic import CONFIG
from utils_basic import build_algorithm, build_scenario, run_algorithm_on_sample
from evaluator_basic import (
    evaluate_sample,
    print_summary,
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
    out_dir = ROOT_DIR / "Results" / "results_basic" / f"{CONFIG['scenario_name']}_{timestamp}"
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
        e = run_algorithm_on_sample(algo, sample)

        # 评估该样本（ERLE / 收敛曲线 / PESQ / SI-SDR 等）
        res = evaluate_sample(
            sample=sample,
            e=e,
            scenario_name=CONFIG["scenario_name"],
            fs=CONFIG["fs"],
            cfg=CONFIG,
        )

        # 直接复用算法类里已有的 weights / weight_history
        # 注意：这里我们把最终 weights 复制一份到结果 dict 中，便于保存与比较
        res["estimated_weights"] = np.asarray(algo.weights, dtype=np.float32).copy()

        # 算法如果有 weight_history 属性，则保存下来（可能为空列表）
        if hasattr(algo, "weight_history"):
            # 统一成 numpy array 或者保留为 None
            if len(algo.weight_history) > 0:
                res["weight_history"] = np.asarray(algo.weight_history, dtype=np.float32)
            else:
                res["weight_history"] = None
        else:
            res["weight_history"] = None

        # 将单个算法的评估结果放入 results
        results[name.upper()] = res

    # 5. 打印结果
    print(f"开始打印结果...")
    print_summary(results, sample, CONFIG)

    # 6. 保存结果
    print(f"开始保存结果...")
    save_results(results, sample, CONFIG, out_dir)

    # 7. 画图
    print(f"开始绘图...")
    plot_curves(results, CONFIG, out_dir)
    plot_signal_comparison(results, sample, CONFIG, out_dir)
    plot_path_comparison(results, sample, CONFIG, out_dir)

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()