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
        algo = build_algorithm(name, CONFIG)
        e = run_algorithm_on_sample(algo, sample)

        res = evaluate_sample(
            sample=sample,
            e=e,
            scenario_name=CONFIG["scenario_name"],
            fs=CONFIG["fs"],
            cfg=CONFIG,
        )

        # 直接复用算法类里已有的 weights / weight_history
        res["estimated_weights"] = np.asarray(algo.weights, dtype=np.float32).copy()

        if len(algo.weight_history) > 0:
            res["weight_history"] = np.asarray(algo.weight_history, dtype=np.float32)
        else:
            res["weight_history"] = None

        results[name.upper()] = res

    # 5. 打印结果
    print(f"开始打印结果...")
    print_summary(results, sample, CONFIG)

    # 6. 保存结果
    save_results(results, sample, CONFIG, out_dir)

    # 7. 画图
    plot_curves(results, CONFIG, out_dir)
    plot_signal_comparison(results, sample, CONFIG, out_dir)
    plot_path_comparison(results, sample, CONFIG, out_dir)

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()