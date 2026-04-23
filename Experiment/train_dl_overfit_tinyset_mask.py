# -*- coding: utf-8 -*-
"""
最小 mask 版本的 tiny-set overfit 实验
-----------------------------------------
建议保存为：
    Experiment/train_dl_overfit_tinyset_mask.py

核心区别：
- 仍然复用你当前的 CNNLSTMSTFT
- 但把它的输出解释为“正的 mask”
- 先与 |D| 相乘得到 |S_hat|
- 再转成 log1p(|S_hat|) 去和 target 的 log1p(|S|) 比较

用途：
1. 固定 10~20 条样本，不再在线重采样
2. 看 mask formulation 是否比 direct magnitude regression 更容易记住 tiny set
3. 保存 loss 曲线和谱图对比

运行：
    python Experiment/train_dl_overfit_tinyset_mask.py
"""

import json
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
    input_feat, target_logmag, dt_mask, meta = batch
    input_feat = input_feat.to(device)
    target_logmag = target_logmag.to(device)
    dt_mask = dt_mask.to(device)
    return input_feat, target_logmag, dt_mask, meta


class FixedTinyDoubleTalkDataset(Dataset):
    """
    一次性固定 tiny set，后续每个 epoch 都重复用同一批样本。
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
        input_feat, target_logmag, dt_mask, meta = self.cached_examples[idx]
        return (
            input_feat.clone(),
            target_logmag.clone(),
            dt_mask.clone(),
            deepcopy(meta),
        )


def reconstruct_logmag_from_mask(
    pred_mask: torch.Tensor,
    input_feat: torch.Tensor,
    eps: float = 1e-8,
):
    """
    pred_mask: [B, T, F], 非负
    input_feat: [B, 2, T, F], 其中 channel0 = log1p(|D|)
    return:
        recon_logmag: [B, T, F] = log1p(pred_mask * |D|)
        d_mag_lin:    [B, T, F]
    """
    d_logmag = input_feat[:, 0, :, :]  # [B,T,F]
    d_mag_lin = torch.expm1(torch.clamp(d_logmag, min=0.0))
    pred_mag_lin = pred_mask * d_mag_lin
    recon_logmag = torch.log1p(torch.clamp(pred_mag_lin, min=0.0) + eps)
    return recon_logmag, d_mag_lin


def build_ideal_mask(
    input_feat: torch.Tensor,
    target_logmag: torch.Tensor,
    eps: float = 1e-8,
):
    """
    ideal mask = |S| / (|D| + eps)
    不做裁剪，保留真实比值，便于观察。
    """
    d_logmag = input_feat[:, 0, :, :]  # [B,T,F]
    d_mag_lin = torch.expm1(torch.clamp(d_logmag, min=0.0))
    target_mag_lin = torch.expm1(torch.clamp(target_logmag, min=0.0))
    ideal_mask = target_mag_lin / (d_mag_lin + eps)
    return ideal_mask


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    max_mask_value: float = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_feat, target_logmag, dt_mask, _ = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        # 这里仍复用你当前模型，但把输出解释成“正 mask”
        pred_mask = model(input_feat)  # [B,T,F], Softplus 保证非负

        if max_mask_value is not None:
            pred_mask = torch.clamp(pred_mask, max=max_mask_value)

        recon_logmag, _ = reconstruct_logmag_from_mask(pred_mask, input_feat)

        loss = criterion(recon_logmag, target_logmag, dt_mask)

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
    max_mask_value: float = None,
):
    model.eval()
    results = []

    for idx in range(len(dataset)):
        input_feat, target_logmag, dt_mask, meta = dataset[idx]

        input_feat_b = input_feat.unsqueeze(0).to(device)
        target_logmag_b = target_logmag.unsqueeze(0).to(device)
        dt_mask_b = dt_mask.unsqueeze(0).to(device)

        pred_mask_b = model(input_feat_b)

        if max_mask_value is not None:
            pred_mask_b = torch.clamp(pred_mask_b, max=max_mask_value)

        recon_logmag_b, _ = reconstruct_logmag_from_mask(pred_mask_b, input_feat_b)
        ideal_mask_b = build_ideal_mask(input_feat_b, target_logmag_b)

        loss = criterion(recon_logmag_b, target_logmag_b, dt_mask_b)

        results.append(
            {
                "index": idx,
                "loss": float(loss.item()),
                "meta": meta,
                "input_feat": input_feat.cpu(),                      # [2,T,F]
                "target_logmag": target_logmag.cpu(),                # [T,F]
                "pred_mask": pred_mask_b[0].detach().cpu(),          # [T,F]
                "ideal_mask": ideal_mask_b[0].detach().cpu(),        # [T,F]
                "recon_logmag": recon_logmag_b[0].detach().cpu(),    # [T,F]
                "dt_mask": dt_mask.cpu(),                            # [T]
            }
        )

    return results


def plot_training_curve(history: Dict[str, List[float]], save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["eval_same_set_loss"], label="eval_same_set_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tiny-set overfit loss curve (mask version)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_one_matrix(ax, mat: torch.Tensor, title: str):
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

        input_feat = item["input_feat"]            # [2,T,F]
        target_logmag = item["target_logmag"]      # [T,F]
        pred_mask = item["pred_mask"]              # [T,F]
        ideal_mask = item["ideal_mask"]            # [T,F]
        recon_logmag = item["recon_logmag"]        # [T,F]
        abs_err = torch.abs(recon_logmag - target_logmag)

        d_logmag = input_feat[0]
        x_logmag = input_feat[1]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        im0 = _plot_one_matrix(axes[0, 0], d_logmag, "Input: log1p(|D|)")
        im1 = _plot_one_matrix(axes[0, 1], x_logmag, "Input: log1p(|X|)")
        axes[0, 2].plot(item["dt_mask"].numpy())
        axes[0, 2].set_title("double-talk frame mask")
        axes[0, 2].set_xlabel("Time frame")
        axes[0, 2].set_ylabel("Mask value")
        axes[0, 2].grid(True, alpha=0.3)

        im2 = _plot_one_matrix(axes[1, 0], target_logmag, "Target: log1p(|S|)")
        im3 = _plot_one_matrix(axes[1, 1], recon_logmag, "Recon: log1p(M * |D|)")
        im4 = _plot_one_matrix(axes[1, 2], abs_err, "|Recon - Target|")

        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

        far_name = os.path.basename(meta.get("far_path") or "unknown_far")
        near_name = os.path.basename(meta.get("near_path") or "unknown_near")

        fig.suptitle(
            f"sample_{idx:02d} | loss={item['loss']:.6f}\n"
            f"far={far_name} | near={near_name}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        save_path = os.path.join(save_dir, f"sample_{idx:02d}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        # 额外单独保存 mask 对比图
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
        im5 = _plot_one_matrix(axes2[0], ideal_mask, "Ideal mask = |S| / |D|")
        im6 = _plot_one_matrix(axes2[1], pred_mask, "Pred mask")
        im7 = _plot_one_matrix(axes2[2], torch.abs(pred_mask - ideal_mask), "|PredMask - IdealMask|")

        fig2.colorbar(im5, ax=axes2[0], fraction=0.046, pad=0.04)
        fig2.colorbar(im6, ax=axes2[1], fraction=0.046, pad=0.04)
        fig2.colorbar(im7, ax=axes2[2], fraction=0.046, pad=0.04)

        fig2.suptitle(
            f"sample_{idx:02d} mask comparison",
            fontsize=11,
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.92])

        save_path2 = os.path.join(save_dir, f"sample_{idx:02d}_mask.png")
        fig2.savefig(save_path2, dpi=150)
        plt.close(fig2)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg = get_config()

    # ========= 你可以先用这组默认参数 =========
    tiny_num_samples = 12
    epochs = 300
    batch_size = 4
    lr = 1e-3
    weight_decay = 0.0

    alpha = 4.0
    dt_weight = 4.0
    non_dt_weight = 0.25

    lstm_hidden = cfg["model"]["lstm_hidden"]

    # 为了避免 mask 过大导致训练数值不稳，可以先给一个较松的上限
    # None 表示不裁剪；建议第一次先用 5.0
    max_mask_value = 5.0

    save_every = 20
    print_every = 1
    # ======================================

    output_dir = os.path.join(
        cfg["root_dir"],
        "Results",
        f"results_dl_overfit_tiny_mask_{tiny_num_samples}",
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

    # 1) 固定 tiny set
    tiny_set = FixedTinyDoubleTalkDataset(
        base_cfg=cfg,
        num_samples=tiny_num_samples,
        split="train",
    )

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
    # 注意：模型结构不变，只改变它的“输出解释方式”
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
    print("Start overfitting tiny set with MASK formulation...\n")

    history = {
        "train_loss": [],
        "eval_same_set_loss": [],
    }

    best_loss = float("inf")
    best_path = os.path.join(ckpt_dir, "best_overfit_mask_model.pt")
    last_path = os.path.join(ckpt_dir, "last_overfit_mask_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(
            model, train_loader, criterion, device, optimizer,
            max_mask_value=max_mask_value
        )
        eval_same_set_loss = run_one_epoch(
            model, eval_loader, criterion, device, optimizer=None,
            max_mask_value=max_mask_value
        )

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
                    "formulation": "mask_times_D_mag",
                    "max_mask_value": max_mask_value,
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
                    "formulation": "mask_times_D_mag",
                    "max_mask_value": max_mask_value,
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
            "formulation": "mask_times_D_mag",
            "max_mask_value": max_mask_value,
        },
        last_path,
    )

    # 3) loss 曲线
    plot_training_curve(
        history,
        save_path=os.path.join(output_dir, "loss_curve.png"),
    )

    # 4) 逐样本评估和存图
    per_sample_results = evaluate_per_sample(
        model=model,
        dataset=tiny_set,
        criterion=criterion,
        device=device,
        max_mask_value=max_mask_value,
    )
    save_example_figures(per_sample_results, fig_dir)

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
        "max_mask_value": max_mask_value,
        "best_eval_same_set_loss": float(best_loss),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_eval_same_set_loss": float(history["eval_same_set_loss"][-1]),
        "num_freq_bins": int(f0),
        "num_time_frames_example": int(t0),
        "trainable_params": int(count_parameters(model)),
        "formulation": "mask_times_D_mag",
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