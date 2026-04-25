import os


def get_config():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    cfg = {
        # -----------------------------
        # 基础
        # -----------------------------
        "seed": 42,
        "device": "cuda",   # 若无 CUDA，可改成 "cpu"
        "root_dir": root_dir,
        "output_dir": os.path.join(root_dir, "Results", "results_dl_crm_test1"),
        # "output_dir": os.path.join(root_dir, "Results", "results_dl_mask_test2"),  # 测试用，实际训练时改回上面这一行

        # -----------------------------
        # 数据 / 场景
        # -----------------------------
        "sample_rate": 16000,
        "duration_sec": 3.0,

        # 兼容旧逻辑的保底字段
        "max_internal_silence_sec": 0.3,

        # ===== 对齐 Experiment.utils_basic 的 build_scenario 关键字 =====
        "scenario_name": "double_talk",
        "ser_db": 0.0,
        # "far_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech1"),
        # "near_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech2"),
        # "rir_dir": os.path.join(root_dir, "Dataset", "simulated_rirs", "smallroom"),
        # 测试数据路径，实际训练时可改成上面两行
        "far_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech_test1"),
        "near_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech_test2"),
        "rir_dir": os.path.join(root_dir, "Dataset", "simulated_rirs", "smallroom", "Room200"),

        "babble_noise_dir": os.path.join(root_dir, "Dataset", "Noise", "babble_noise"),
        "fs": 16000,
        "rir_path": None,

        # -----------------------------
        # 新增：统一的语音预处理配置
        # far / near 分开配
        # -----------------------------
        # "speech_preprocess": {
        #     "far": {
        #         "trim_mode": "active",
        #         "silence_threshold_db": -40.0,
        #         "max_internal_silence_sec": 0.30,
        #         "min_activity_ratio": 0.10,
        #         "min_rms_db": -45.0,
        #         "max_trials": 50,
        #         "frame_len_ms": 25.0,
        #         "hop_len_ms": 10.0,
        #         "energy_threshold_ratio": None,
        #     },
        #     "near": {
        #         "trim_mode": "active",
        #         "silence_threshold_db": -40.0,
        #         "max_internal_silence_sec": 0.001,
        #         "min_activity_ratio": 0.35,
        #         "min_rms_db": -45.0,
        #         "max_trials": 80,
        #         "frame_len_ms": 25.0,
        #         "hop_len_ms": 10.0,
        #         "energy_threshold_ratio": None,
        #     },
        # },

        # 测试时调整的预处理配置，实际训练时可改回上面这一段
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

        "double_talk_segment_cfg": {
            "mode": "random",
            "num_dt_range": (1, 3),
            "total_dt_ratio_range": (0.35, 0.75),
            "min_dt_sec": 0.30,
            "min_fst_sec": 0.15,
        },

        # -----------------------------
        # STFT
        # -----------------------------
        "stft": {
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
        },

        # -----------------------------
        "train_num_samples": 1000,
        "val_num_samples": 200,

        "batch_size": 8,
        "num_workers": 2,

        "train": {
            "epochs": 40,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "save_every_epoch": True,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 1e-4,
        },

        # -----------------------------
        # 模型
        # -----------------------------
        "model": {
            "lstm_hidden": 128,
        },

        # -----------------------------
        # 推理
        # -----------------------------
        "inference": {
            "checkpoint_path": None,
                #os.path.join(root_dir, "Results", "results_dl_mask", "checkpoints", "best_model_mask.pt"),
                                                                                    # None 时默认找 best_model.pt
            "sample_index": 0,
        },
    }

    return cfg