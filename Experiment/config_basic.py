from pathlib import Path


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
    "duration_sec": 15.0,

    # ===== 数据路径 =====
    "far_speech_dir": ROOT_DIR / "Dataset" / "clean_speech1",
    "near_speech_dir": ROOT_DIR / "Dataset" / "clean_speech2",
    "babble_noise_dir": ROOT_DIR / "Dataset" / "Noise" / "babble_noise",
    "rir_dir": ROOT_DIR / "Dataset" / "simulated_rirs",

    # 若手动指定某一条 RIR 文件，填完整路径；否则设为 None
    #"rir_path": ROOT_DIR / "Dataset" / "simulated_rirs" / "synthetic_8rir.wav",
    "rir_path": ROOT_DIR / "Dataset" / "simulated_rirs" / "smallroom" / "Room001" / "Room001-00001.wav",

    # ===== 场景参数 =====
    "snr_db": 15.0,             # noisy_single_talk 用
    "ser_db": 0.0,              # double_talk 用
    "change_time_sec": 7.5,     # path_change 用
    "noise_type": "white",      # "white" / "impulse" / "babble"

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

    # ===== 显示范围 =====
    # None 表示整段都画；也可以设成 2.0、3.0 这种秒数
    "signal_plot_max_sec": None,
    # 路径对比时最多画多少个 tap；None 表示全画
    "path_plot_max_len": None,

    # ===== 算法列表 =====
    "algorithms": ["lms", "nlms", "rls"],

    # ===== 算法参数 =====
    "alg_params": {
        "lms": {
            "filter_length": 512,
            "step_size": 0.05,
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
    },
}