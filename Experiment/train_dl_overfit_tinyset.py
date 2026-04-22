# -*- coding: utf-8 -*-
"""
极小集合过拟合测试
---------------------------------
建议保存为：
    Experiment/train_dl_overfit_tinyset.py

用途：
1. 固定 10~20 条样本，不再在线重采样
2. 反复训练到几乎记住
3. 看 train loss 能否压很低
4. 保存每条样本的谱图对比（input / pred / target / abs error）

运行示例：
    python Experiment/train_dl_overfit_tinyset.py

你也可以改下面 main() 里的参数：
    tiny_num_samples=12
    epochs=300
    lr=1e-3
"""

import json
import math
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

# 允许从 repo 根目录运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Experiment.config_dl import get_config
from Models.cnn_lstm_stft import CNNLSTMSTFT
from Training.dataset_doubletalk import DoubleTalkSTFTDataset
from Training.losses import DTMaskedWeightedSpectralL1Loss
from Tools.set_seed import set_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch, device):
    input_feat, target_mag, dt_mask, meta = batch
    input_feat = input_feat.to(device)
    target_mag = target_mag.to(device)
    dt_mask = dt_mask.to(device)
    return input_feat, target_mag, dt_mask, meta


class FixedTinyDoubleTalkDataset(Dataset):
    """
    关键点：
    - 只在初始化时，调用一次底层在线数据集构造样本
    - 把样本缓存到内存
    - 后续每个 epoch 都返回同一批固定样本

    这样才能真正做“极小集合过拟合测试”。
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        num_samples: int,
        split: str = "train",
    ):
        super().__init__()
        self.base_cfg = deepcopy(base_cfg)
        self.num_samples = int(num_samples)
        self.split = split

        self.source_dataset = DoubleTalkSTFTDataset(
            base_cfg=self.base_cfg,
            num_samples=self.num_samples,
            split=split,
        )

        self.cached_examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]] = []
        self._build_and_cache()

    def _build_and_cache(self):
        print(f"Building fixed tiny dataset with {self.num_samples} samples ...")
        for idx in range(self.num_samples):
            # 关键：这里只构造一次
            sample = self.source_dataset.build_valid_sample(idx)
            example = self.source_dataset.sample_to_example(sample)
            self.cached_examples.append(example)

            _, _, _, meta = example
            print(
                f"[cache {idx:02d}] "
                f"far={meta.get('far_path')} | near={meta.get('near_path')} | "
                f"len={meta.get('length')}"
            )

        print("Fixed tiny dataset ready.\n")

    def __len__(self):
        return len(self.cached_examples)

    def __getitem__(self, idx: int):
        input_feat, target_mag, dt_mask, meta = self.cached_examples[idx]

        # 返回 clone，避免训练过程中意外原地修改缓存
        return (
            input_feat.clone(),
            target_mag.clone(),
            dt_mask.clone(),
            deepcopy(meta),
        )


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
        input_feat, target_mag, dt_mask, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        pred_mag = model(input_feat)  # [B, T, F]
        loss = criterion(pred_mag, target_mag, dt_mask)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = input_feat.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_per_sample(
    model: torch.nn.Module,
    dataset: FixedTinyDoubleTalkDataset,
    criterion: torch.nn.Module,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    for idx in range(len(dataset)):
        input_feat, target_mag, dt_mask, meta = dataset[idx]

        input_feat_b = input_feat.unsqueeze(0).to(device)      # [1,2,T,F]
        target_mag_b = target_mag.unsqueeze(0).to(device)      # [1,T,F]
        dt_mask_b = dt_mask.unsqueeze(0).to(device)            # [1,T]

        pred_mag_b = model(input_feat_b)
        loss = criterion(pred_mag_b, target_mag_b, dt_mask_b)

        pred_mag = pred_mag_b[0].detach().cpu()                # [T,F]

        results.append(
            {
                "index": idx,
                "loss": float(loss.item()),
                "meta": meta,
                "input_feat": input_feat.cpu(),                # [2,T,F]
                "target_mag": target_mag.cpu(),                # [T,F]
                "pred_mag": pred_mag.cpu(),                    # [T,F]
                "dt_mask": dt_mask.cpu(),                      # [T]
            }
        )

    return results


def plot_training_curve(history: Dict[str, List[float]], save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["eval_same_set_loss"], label="eval_same_set_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tiny-set overfit loss curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_one_matrix(ax, mat: torch.Tensor, title: str):
    # mat: [T,F]
    im = ax.imshow(
        mat.T.numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Freq bin")
    return im


def save_example_figures(
    per_sample_results: List[Dict[str, Any]],
    save_dir: str,
):
    ensure_dir(save_dir)

    for item in per_sample_results:
        idx = item["index"]
        meta = item["meta"]

        input_feat = item["input_feat"]     # [2,T,F]
        target_mag = item["target_mag"]     # [T,F]
        pred_mag = item["pred_mag"]         # [T,F]
        abs_err = torch.abs(pred_mag - target_mag)

        d_mag = input_feat[0]
        x_mag = input_feat[1]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        im0 = _plot_one_matrix(axes[0, 0], d_mag, "Input: log1p(|D|)")
        im1 = _plot_one_matrix(axes[0, 1], x_mag, "Input: log1p(|X|)")
        axes[0, 2].plot(item["dt_mask"].numpy())
        axes[0, 2].set_title("double-talk frame mask")
        axes[0, 2].set_xlabel("Time frame")
        axes[0, 2].set_ylabel("Mask value")
        axes[0, 2].grid(True, alpha=0.3)

        im2 = _plot_one_matrix(axes[1, 0], target_mag, "Target: log1p(|S|)")
        im3 = _plot_one_matrix(axes[1, 1], pred_mag, "Pred: log1p(|S_hat|)")
        im4 = _plot_one_matrix(axes[1, 2], abs_err, "|Pred - Target|")

        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

        far_name = os.path.basename(meta.get("far_path") or "unknown_far")
        near_name = os.path.basename(meta.get("near_path") or "unknown_near")

        fig.suptitle(
            f"sample_{idx:02d} | "
            f"loss={item['loss']:.6f}\n"
            f"far={far_name} | near={near_name}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        save_path = os.path.join(save_dir, f"sample_{idx:02d}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg = get_config()

    # ========= 你可以在这里改实验参数 =========
    tiny_num_samples = 12       # 建议先 10~12；再试 16 或 20
    epochs = 300                # 过拟合测试建议拉长
    batch_size = 4
    lr = 1e-3
    weight_decay = 0.0
    alpha = 4.0
    dt_weight = 4.0
    non_dt_weight = 0.25
    lstm_hidden = cfg["model"]["lstm_hidden"]   # 默认沿用当前配置
    save_every = 20          # 每多少个 epoch 存一次 checkpoint
    print_every = 1
    # ======================================

    # 单独输出目录，避免和正常训练混在一起
    output_dir = os.path.join(
        cfg["root_dir"],
        "Results",
        f"results_dl_overfit_tiny_{tiny_num_samples}",
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(output_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(fig_dir)

    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # 1) 构建固定 tiny set
    tiny_set = FixedTinyDoubleTalkDataset(
        base_cfg=cfg,
        num_samples=tiny_num_samples,
        split="train",   # 这里直接固定 train split 的前 tiny_num_samples 条
    )

    # 同一批样本既做 train，也做 eval_same_set
    train_loader = DataLoader(
        tiny_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        tiny_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    sample_input, sample_target, sample_dt_mask, _ = tiny_set[0]
    _, t0, f0 = sample_input.shape

    # 2) 模型 / 损失 / 优化器
    model = CNNLSTMSTFT(
        num_freq_bins=f0,
        lstm_hidden=lstm_hidden,
    ).to(device)

    criterion = DTMaskedWeightedSpectralL1Loss(
        alpha=alpha,
        dt_weight=dt_weight,
        non_dt_weight=non_dt_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    print(f"Device: {device}")
    print(f"Tiny samples: {len(tiny_set)}")
    print(f"Example shape: input={tuple(sample_input.shape)}, target={tuple(sample_target.shape)}")
    print(f"Trainable params: {count_parameters(model):,}")
    print("Start overfitting tiny set...\n")

    history = {
        "train_loss": [],
        "eval_same_set_loss": [],
    }

    best_loss = float("inf")
    best_path = os.path.join(ckpt_dir, "best_overfit_model.pt")
    last_path = os.path.join(ckpt_dir, "last_overfit_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
        eval_same_set_loss = run_one_epoch(model, eval_loader, criterion, device, optimizer=None)

        history["train_loss"].append(float(train_loss))
        history["eval_same_set_loss"].append(float(eval_same_set_loss))

        if epoch % print_every == 0:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} "
                f"eval_same_set_loss={eval_same_set_loss:.6f}"
            )

        if eval_same_set_loss < best_loss:
            best_loss = eval_same_set_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "epoch": epoch,
                    "best_eval_same_set_loss": best_loss,
                    "tiny_num_samples": tiny_num_samples,
                },
                best_path,
            )

        if epoch % save_every == 0 or epoch == epochs:
            save_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_freq_bins": f0,
                    "epoch": epoch,
                    "tiny_num_samples": tiny_num_samples,
                },
                save_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "num_freq_bins": f0,
            "epoch": epochs,
            "tiny_num_samples": tiny_num_samples,
        },
        last_path,
    )

    # 3) 保存 loss 曲线
    plot_training_curve(
        history,
        save_path=os.path.join(output_dir, "loss_curve.png"),
    )

    # 4) 逐样本评估并存图
    per_sample_results = evaluate_per_sample(model, tiny_set, criterion, device)
    save_example_figures(per_sample_results, fig_dir)

    # 5) 保存逐样本数字结果
    per_sample_json = {}
    for item in per_sample_results:
        idx = item["index"]
        per_sample_json[f"sample_{idx:02d}"] = {
            "loss": float(item["loss"]),
            "meta": item["meta"],
        }

    summary = {
        "device_used": device.type,
        "tiny_num_samples": tiny_num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "lstm_hidden": lstm_hidden,
        "best_eval_same_set_loss": float(best_loss),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_eval_same_set_loss": float(history["eval_same_set_loss"][-1]),
        "num_freq_bins": int(f0),
        "num_time_frames_example": int(t0),
        "trainable_params": int(count_parameters(model)),
    }

    save_json(history, os.path.join(output_dir, "train_history.json"))
    save_json(summary, os.path.join(output_dir, "summary.json"))
    save_json(per_sample_json, os.path.join(output_dir, "per_sample_metrics.json"))

    print("\nFinished.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Loss curve: {os.path.join(output_dir, 'loss_curve.png')}")
    print(f"Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()