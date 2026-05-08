from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG = {
    # ===== 随机种子 / 基本参数 =====
    "seed": 42,
    "fs": 16000,

    # ===== 场景选择 =====
    # 可选：
    # "farend_single_talk"
    # "noisy_single_talk"
    # "double_talk"
    # "path_change"
    "scenario_name": "farend_single_talk",

    # ===== 单条实验语音时长 =====
    "duration_sec": 3.0,

    # ===== 数据路径 =====
    "far_speech_dir": ROOT_DIR / "Dataset" / "clean_speech_test1",
    "near_speech_dir": ROOT_DIR / "Dataset" / "clean_speech_test2",
    "babble_noise_dir": ROOT_DIR / "Dataset" / "Noise" / "babble_noise",
    "rir_dir": ROOT_DIR / "Dataset" / "simulated_rirs" / "smallroom" / "Room200",
    #"rir_dir": ROOT_DIR / "Dataset" / "simulated_rirs" / "easy_8rir",

    # 若手动指定某一条 RIR 文件，填完整路径；否则设为 None
    #"rir_path": ROOT_DIR / "Dataset" / "simulated_rirs" / "easy_8rir",
    #"rir_path": ROOT_DIR / "Dataset" / "simulated_rirs" / "smallroom" / "Room001" / "Room001-00001.wav",
    "rir_path": None,

    # ===== 场景参数 =====
    "snr_db": 15.0,             # noisy_single_talk 用
    "ser_db": 0.0,              # double_talk 用
    "change_time_sec": 7.5,     # path_change 用
    "noise_type": "babble",      # "white" / "impulse" / "babble"

    # ===== 语音预处理参数 =====
    # 这部分和 config_dl.py 中的 speech_preprocess 对齐，
    # 现在 basic 平台也统一从这里调度。
    "speech_preprocess": {
        "far": {
            "trim_mode": "active",
            "silence_threshold_db": -40.0,
            "max_internal_silence_sec": 0.50,
            "min_activity_ratio": 0.05,
            "min_rms_db": -45.0,
            "max_trials": 50,
            "frame_len_ms": 25.0,
            "hop_len_ms": 10.0,
            "energy_threshold_ratio": None,
        },
        "near": {
            "trim_mode": "active",
            "silence_threshold_db": -40.0,
            "max_internal_silence_sec": 0.50,
            "min_activity_ratio": 0.05,
            "min_rms_db": -45.0,
            "max_trials": 80,
            "frame_len_ms": 25.0,
            "hop_len_ms": 10.0,
            "energy_threshold_ratio": None,
        },
    },

    # ===== double-talk 分段参数 =====
    # mode = "random" 时，会调用 sample_random_double_talk_segments()
    # 让 near-end 出现更自然的说话/停顿分布。
    "double_talk_segment_cfg": {
        "mode": "random",
        "num_dt_range": (1, 3),
        "total_dt_ratio_range": (0.35, 0.75),
        "min_dt_sec": 0.30,
        "min_fst_sec": 0.15,
    },

    # ===== 曲线参数 =====
    "curve_window_size": 512,   # 收敛曲线平滑窗口
    "erle_frame_size": 512,
    "erle_hop_size": 512,

    # ===== 输出开关 =====
    "save_npz": True,
    "save_fig": True,
    "save_summary_json": True,
    # ===== 图像开关 =====
    "plot_signal_waveforms": True,
    "plot_path_compare": True,
    # ===== 权值历史记录开关 =====
    # False：不保存逐采样点 weight_history，避免长语音 + 高阶滤波器导致 CPU OOM
    # True ：保存完整 weight_history，仅建议短语音调试或 path_change 细看路径时使用
    "record_weight_history": False,

    # ===== 复杂度 / 推理时间测试 =====
    "complexity": {
        "enable": True,

        # 预热次数：不计入最终时间。
        # 作用：避开第一次 CUDA context 初始化、cuDNN 选择、内存池启动等冷启动开销。
        "warmup_runs": 2,

        # 正式计时次数：最终报告 mean / median / std。
        # 如果 RLS 很慢，可以先改成 3。
        "timed_runs": 5,

        # 是否测 CUDA 显存。只有算法实际在 cuda 上运行时才有效。
        "measure_cuda_memory": True,
    },

    # ===== 显示范围 =====
    # None 表示整段都画；也可以设成 2.0、3.0 这种秒数
    "signal_plot_max_sec": None,
    # 路径对比时最多画多少个 tap；None 表示全画
    "path_plot_max_len": None,

    # ===== 算法列表 =====
    #"algorithms": ["lms", "nlms", "rls"],
    "algorithms": ["lms", "nlms", "klms", "dl_hybrid"],

    # ===== 算法参数 =====
    "alg_params": {
        "lms": {
            "filter_length": 512,
            "step_size": 0.5,
        },
        "nlms": {
            "filter_length": 1024,
            "step_size": 0.8,
            "epsilon": 1e-1,
        },
        "rls": {
            "filter_length": 1024,
            "lambda_": 0.98,
            "delta": 0.1,
        },
        # ================= 新增 KLMS 参数 =================
        "klms": {
            "filter_length": 512,    # 时间延迟嵌入长度
            "step_size": 0.5,        # 学习率 (eta)
            "kernel_param": 0.1,     # 高斯核带宽参数 (gamma/a)
            "budget": 1000,          # 字典大小上限，防止长时间序列导致 OOM
        },
        # ===== PyTorch adaptive filters =====
        "torch_lms": {
            "filter_length": 512,
            "step_size": 0.05,
            "device": "cuda",
        },

        "torch_nlms": {
            "filter_length": 1024,
            "step_size": 0.8,
            "epsilon": 1e-1,
            "device": "cuda",
        },

        "torch_rls": {
            "filter_length": 1024,
            "lambda_": 0.98,
            "delta": 0.1,
            "device": "cuda",
        },
        # ===== V2-4A: deep hybrid model as one platform algorithm =====
        "dl_hybrid": {
            # 按你的训练脚本默认位置：
            # Results/results_dl_hybrid/checkpoints/best_model_hybrid.pt
            "checkpoint_path": str(
                ROOT_DIR / "Results" / "results_dl_hybrid" / "checkpoints" / "best_model_hybrid.pt"
            ),

            # 没有 GPU 就改成 "cpu"
            "device": "cuda",

            # 必须和训练时保持一致
            "stft": {
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 512,
            },

            # None 表示读取 checkpoint 里的 beta_residual
            # 也可以手动覆盖，比如 0.40
            "beta_residual": None,

            "strict": True,
        },
    },
}