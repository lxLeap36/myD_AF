import json
import os
import sys
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# 允许从 repo 根目录运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.losses import WeightedSpectralL1Loss
from Tools.set_seed import set_seed


def move_batch_to_device(batch, device):
    input_feat, target_mag, meta = batch
    input_feat = input_feat.to(device)
    target_mag = target_mag.to(device)
    return input_feat, target_mag, meta


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_feat, target_mag, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad() # 清除上一步的梯度信息，准备计算当前 batch 的梯度

        pred_mag = model(input_feat)                # [B, T, F]
        loss = criterion(pred_mag, target_mag)

        if is_train:
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 按照优化器的更新规则更新模型参数

        bsz = input_feat.size(0)
        total_loss += loss.item() * bsz
        # 在 PyTorch 中，loss 是一个标量张量，其值通常是该 batch 内所有样本损失的平均值，
        # 用 loss.item() * bsz 恢复该 batch 的总损失和（所有样本损失之和）
        total_count += bsz # 总样本数量

    mean_loss = total_loss / max(total_count, 1) # 用 总损失和 / 总样本数 得到所有样本的平均损失。1 是为了避免除以零的情况。
    return mean_loss


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    cfg = get_config()
    # 配置与目录准备
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 随机种子与设备
    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # 数据集
    train_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["train_num_samples"],
        split="train",
    )
    val_set = DoubleTalkSTFTDataset(
        base_cfg=cfg,
        num_samples=cfg["val_num_samples"],
        split="val",
    )

    # # ===== 调试：打印前 20 个训练样本对应的 far / near 文件组合 =====
    # print("\n===== Preview first 20 training samples =====")
    # preview_n = min(20, len(train_set))
    # combo_counter = {}
    #
    # for i in range(preview_n):
    #     raw_sample = train_set._build_one_sample(i)
    #
    #     extra = raw_sample.get("meta", {}).get("extra", {})
    #     far_meta = extra.get("far_meta", {}) or {}
    #     near_meta = extra.get("near_meta", {}) or {}
    #
    #     far_path = far_meta.get("file_path", "N/A")
    #     near_path = near_meta.get("file_path", "N/A")
    #     far_start = far_meta.get("start_sample_in_raw_file", "N/A")
    #     near_start = near_meta.get("start_sample_in_raw_file", "N/A")
    #
    #     combo_key = (far_path, near_path)
    #     combo_counter[combo_key] = combo_counter.get(combo_key, 0) + 1
    #
    #     print(f"[train sample {i:02d}]")
    #     print(f"  far : {far_path}")
    #     print(f"        start_sample_in_raw_file = {far_start}")
    #     print(f"  near: {near_path}")
    #     print(f"        start_sample_in_raw_file = {near_start}")
    #
    # print("\n===== Unique far/near combinations in preview =====")
    # for k, v in combo_counter.items():
    #     print(f"{v:2d} times | far={k[0]} | near={k[1]}")
    # print("=============================================\n")
    #
    # # ===== 统计整个训练集的 far/near 组合分布 =====
    # print("\n===== Count all train combinations =====")
    # all_combo_counter = {}
    #
    # for i in range(len(train_set)):
    #     raw_sample = train_set._build_one_sample(i)
    #
    #     extra = raw_sample.get("meta", {}).get("extra", {})
    #     far_meta = extra.get("far_meta", {}) or {}
    #     near_meta = extra.get("near_meta", {}) or {}
    #
    #     far_path = far_meta.get("file_path", "N/A")
    #     near_path = near_meta.get("file_path", "N/A")
    #
    #     combo_key = (far_path, near_path)
    #     all_combo_counter[combo_key] = all_combo_counter.get(combo_key, 0) + 1
    #
    # for k, v in sorted(all_combo_counter.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{v:3d} times | far={k[0]} | near={k[1]}")
    #
    # print("=======================================\n")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # 用一条样本自动确定 F
    sample_input, sample_target, _ = train_set[0] # 还有meta_dict，额外信息（可选调试）
    _, t0, f0 = sample_input.shape

    # 模型、损失、优化器
    model = CNNLSTMSTFT(
        num_freq_bins=f0,
        lstm_hidden=cfg["model"]["lstm_hidden"],
    ).to(device)

    criterion = WeightedSpectralL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    last_path = os.path.join(ckpt_dir, "last_model.pt")

    epochs = cfg["train"]["epochs"]
    print(f"Starting training for {epochs} epochs on device: {device}.")
    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        # 验证集只读不改，它的权重来源于本 epoch 训练结束时的模型参数。
        """在同一个 epoch 中，run_one_epoch 先以训练模式（optimizer 不为 None）执行，
        模型参数会通过 loss.backward() 和 optimizer.step() 被更新；
            紧接着以验证模式（optimizer=None）执行时：is_train = False → model.train(False) → 模型进入评估模式,
        不会调用 optimizer.zero_grad()、loss.backward()、optimizer.step(),模型权重不发生任何改变
            所以验证集使用的权重正是当前 epoch 训练刚刚完成后的参数（即上一轮训练更新的结果）。
            这样设计的目的是：在每个训练 epoch 结束后，立即用最新的模型在验证集上评估性能，以便判断是否过拟合、是否保存最佳模型等。"""

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(f"[Epoch {epoch:03d}/{epochs:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  -> saved best model to: {best_path}")

        if cfg["train"]["save_every_epoch"]:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "epoch": epoch,
                },
                epoch_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "num_freq_bins": f0,
            "epoch": epochs,
        },
        last_path,
    )

    save_json(history, os.path.join(cfg["output_dir"], "train_history.json"))
    save_json(
        {
            "best_val_loss": float(best_val),
            "best_model_path": best_path,
            "last_model_path": last_path,
            "device_used": device.type,
            "num_freq_bins": int(f0),
            "num_time_frames_example": int(t0),
        },
        os.path.join(cfg["output_dir"], "train_summary.json"),
    )

    print("Training finished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()