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
        "output_dir": os.path.join(root_dir, "Results", "results_dl"),

        # -----------------------------
        # 数据 / 场景
        # 这里尽量沿用你现有平台的字段
        # -----------------------------
        "sample_rate": 16000,
        "duration_sec": 3.0,

        # "dataset": {
        #     "clean_speech1_dir": os.path.join(root_dir, "Dataset", "clean_speech1"),
        #     "clean_speech2_dir": os.path.join(root_dir, "Dataset", "clean_speech2"),
        #     "rir_dir": os.path.join(root_dir, "Dataset", "simulated_rirs", "smallroom", "Room001"),
        #     "babble_noise_dir": os.path.join(root_dir, "Dataset", "Noise", "babble_noise"),
        # },

        # ===== 对齐Experiment.utils_basic 的 build_scenario load_rir 中的关键字 =====
        "scenario_name": "double_talk",
        "ser_db": 0.0,  # double_talk 用
        "far_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech1"),
        "near_speech_dir": os.path.join(root_dir, "Dataset", "clean_speech2"),
        "rir_dir": os.path.join(root_dir, "Dataset", "simulated_rirs", "smallroom", "Room001"),
        # "rir_dir": ROOT_DIR / "Dataset" / "simulated_rirs" / "smallroom" / "Room001",
        "babble_noise_dir": os.path.join(root_dir, "Dataset", "Noise", "babble_noise"),
        "fs": 16000,
        "rir_path": None,

        # "scenario": {
        #     "name": "double_talk",
        #     # 下面这些字段名，如果和你现有 build_scenario 要求有少量出入，
        #     # 到时我再按你 repo 实际报错帮你对齐。
        #     "double_talk": {
        #         "ser_db_min": -6.0,
        #         "ser_db_max": 6.0,
        #     },
        # },

        # -----------------------------
        # STFT
        # -----------------------------
        "stft": {
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
        },

        # -----------------------------
        # 训练数据规模
        # -----------------------------
        "train_num_samples": 200,
        "val_num_samples": 40,

        # -----------------------------
        # dataloader
        # -----------------------------
        "batch_size": 4,
        "num_workers": 0,

        # -----------------------------
        # 模型
        # -----------------------------
        "model": {
            "lstm_hidden": 128,
        },

        # -----------------------------
        # 训练
        # -----------------------------
        "train": {
            "epochs": 10,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "save_every_epoch": True,
        },

        # -----------------------------
        # 推理
        # -----------------------------
        "inference": {
            "checkpoint_path": None,  # None 时默认找 best_model.pt
            "sample_index": 0,
        },
    }

    return cfg