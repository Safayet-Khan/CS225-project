from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Path setup (same trick as in train.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AudioFuse.models.audiofuse import AudioFuseLCF, count_parameters
from AudioFuse.training.train import (
    DatasetBundle,
    apply_overrides,
    build_optimizer,
    build_physionet_bundle,
    build_scheduler,
    evaluate,
    load_config,
    train_one_epoch,
)
from AudioFuse.utils.logging import (
    CSVLogger,
    dump_json,
    evaluating,
    get_logger,
    make_run_dir,
    save_checkpoint,
    seed_everything,
)
from AudioFuse.utils.losses import build_loss
from AudioFuse.utils.metrics import (
    BinaryMetrics,
    MetricAccumulator,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_fused_embeddings(
    model: AudioFuseLCF,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:

    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with evaluating(model):
        for waveform, spec, target in loader:
            waveform = waveform.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            emb = model.encode(waveform, spec)
            feats.append(emb.detach().cpu().float().numpy())
            targets.append(target.detach().cpu().long().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)


# ---------------------------------------------------------------------------
# SMOTE wrapper
# ---------------------------------------------------------------------------


def apply_smote(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int,
    sampling_strategy: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn is required for the SMOTE variant. "
            "Install it via `pip install imbalanced-learn` and retry."
        ) from e

    n_min = int((labels == 1).sum())
    if n_min < 2:
        # SMOTE needs at least 2 minority samples.
        return embeddings, labels
    k = min(k_neighbors, n_min - 1)
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(embeddings, labels)
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ---------------------------------------------------------------------------
# Frozen-branch head training on augmented embeddings
# ---------------------------------------------------------------------------


def train_head_on_embeddings(
    model: AudioFuseLCF,
    embeddings: np.ndarray,
    labels: np.ndarray,
    val_loader: DataLoader,
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
    device: torch.device,
    csv_logger: CSVLogger,
    logger,
    start_epoch: int,
    total_epochs: int,
    best_ckpt_path: str,
) -> float:
    """Train only the head on the SMOTE-augmented feature set.

    Returns the best validation loss observed during this stage.
    """
    # Freeze the encoders.
    for p in model.spec_encoder.parameters():
        p.requires_grad = False
    for p in model.wave_encoder.parameters():
        p.requires_grad = False

    # Optimiser only on head parameters now.
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    # A fresh cosine schedule for the remaining epochs.
    remaining = max(total_epochs - start_epoch + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
    loss_fn = build_loss(bundle.class_weights).to(device)

    feat_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(embeddings).float(),
            torch.from_numpy(labels).long(),
        ),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    es_cfg = cfg["training"]["early_stopping"]
    best_metric = float("inf") if es_cfg["mode"] == "min" else -float("inf")
    patience_left = int(es_cfg["patience"])

    for epoch in range(start_epoch, total_epochs + 1):
        model.head.train()
        running_loss = 0.0
        n = 0
        acc = MetricAccumulator()
        t0 = time.time()
        for emb, target in feat_loader:
            emb = emb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(emb)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            bs = target.size(0)
            running_loss += loss.item() * bs
            n += bs
            acc.update(logits, target)
        scheduler.step()
        train_loss = running_loss / max(n, 1)
        train_metrics = acc.compute()

        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        csv_logger.log(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics.accuracy,
                "val_acc": val_metrics.accuracy,
                "train_f1": train_metrics.f1,
                "val_f1": val_metrics.f1,
                "train_auc": train_metrics.roc_auc,
                "val_auc": val_metrics.roc_auc,
                "train_mcc": train_metrics.mcc,
                "val_mcc": val_metrics.mcc,
                "elapsed_s": elapsed,
            }
        )
        logger.info(
            "[SMOTE-stage] Epoch %3d/%d | train_loss=%.4f val_loss=%.4f | "
            "val_AUC=%.4f val_F1=%.4f val_MCC=%.4f | %.1fs",
            epoch, total_epochs,
            train_loss, val_loss,
            val_metrics.roc_auc, val_metrics.f1, val_metrics.mcc, elapsed,
        )

        monitor_value = (
            val_loss if es_cfg["monitor"] == "val_loss" else -val_metrics.roc_auc
        )
        improved = (
            monitor_value < best_metric
            if es_cfg["mode"] == "min"
            else monitor_value > best_metric
        )
        if improved:
            best_metric = monitor_value
            patience_left = int(es_cfg["patience"])
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.as_dict(),
                    "model_name": "audiofuse_smote",
                    "stage": "smote",
                    "class_weights": bundle.class_weights,
                },
                best_ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    return float(best_metric)


def train_smote_one_seed(
    seed: int,
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
) -> Dict[str, Any]:
    """Run the full warmup -> SMOTE -> frozen-head pipeline for one seed."""
    seed_everything(seed, deterministic=cfg.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(cfg["logging"]["output_dir"], "audiofuse_smote", seed)
    logger = get_logger(
        f"AudioFuse.smote.seed{seed}",
        log_file=os.path.join(run_dir, "train.log"),
    )
    csv_logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))

    # Build the LCF backbone explicitly (SMOTE variant requires .encode()).
    model = AudioFuseLCF(cfg["model"]).to(device)
    n_params = count_parameters(model)
    logger.info("AudioFuse-LCF backbone built with %d trainable params.", n_params)

    loss_fn = build_loss(bundle.class_weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_loader = DataLoader(
        bundle.train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )
    train_loader_no_shuffle = DataLoader(
        bundle.train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )
    test_loader = DataLoader(
        bundle.test,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    best_ckpt_path = os.path.join(run_dir, "best.pt")

    # ----------------- Stage 1: warm-up ----------------------
    warmup = int(cfg["smote"]["warmup_epochs"])
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0))
    for epoch in range(1, warmup + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, grad_clip, scaler
        )
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0
        csv_logger.log(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics.accuracy,
                "val_acc": val_metrics.accuracy,
                "train_f1": train_metrics.f1,
                "val_f1": val_metrics.f1,
                "train_auc": train_metrics.roc_auc,
                "val_auc": val_metrics.roc_auc,
                "train_mcc": train_metrics.mcc,
                "val_mcc": val_metrics.mcc,
                "elapsed_s": elapsed,
            }
        )
        logger.info(
            "[Warmup ] Epoch %3d/%d | train_loss=%.4f val_loss=%.4f | "
            "val_AUC=%.4f val_F1=%.4f val_MCC=%.4f | %.1fs",
            epoch, warmup,
            train_loss, val_loss,
            val_metrics.roc_auc, val_metrics.f1, val_metrics.mcc, elapsed,
        )

    # ----------------- Stage 2: extract embeddings ----------
    logger.info("Extracting fused embeddings for SMOTE...")
    embeddings, labels = extract_fused_embeddings(model, train_loader_no_shuffle, device)
    logger.info(
        "Pre-SMOTE: shape=%s, pos=%d, neg=%d",
        embeddings.shape,
        int((labels == 1).sum()),
        int((labels == 0).sum()),
    )

    # ----------------- Stage 3: SMOTE ----------------------
    embeddings, labels = apply_smote(
        embeddings,
        labels,
        k_neighbors=int(cfg["smote"]["k_neighbors"]),
        sampling_strategy=str(cfg["smote"]["sampling_strategy"]),
        seed=seed,
    )
    logger.info(
        "Post-SMOTE: shape=%s, pos=%d, neg=%d",
        embeddings.shape,
        int((labels == 1).sum()),
        int((labels == 0).sum()),
    )

    # ----------------- Stage 4: frozen-branch head training ----
    total_epochs = int(cfg["training"]["epochs"])
    train_head_on_embeddings(
        model=model,
        embeddings=embeddings,
        labels=labels,
        val_loader=val_loader,
        cfg=cfg,
        bundle=bundle,
        device=device,
        csv_logger=csv_logger,
        logger=logger,
        start_epoch=warmup + 1,
        total_epochs=total_epochs,
        best_ckpt_path=best_ckpt_path,
    )

    # ----------------- Stage 5: test ------------------------
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)
    logger.info("Test results: loss=%.4f | %s", test_loss, test_metrics)

    summary = {
        "model": "audiofuse_smote",
        "seed": seed,
        "n_params": n_params,
        "test_loss": test_loss,
        "test_metrics": test_metrics.as_dict(),
        "best_ckpt": best_ckpt_path,
    }
    dump_json(os.path.join(run_dir, "summary.json"), summary)
    return summary


def run_smote_multi_seed(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Train the SMOTE variant across all configured seeds."""
    seeds = list(cfg["seeds"])
    summaries = []
    metric_objs: List[BinaryMetrics] = []
    for seed in seeds:
        bundle = build_physionet_bundle(cfg, seed=seed)
        s = train_smote_one_seed(seed, cfg, bundle)
        summaries.append(s)
        m = s["test_metrics"]
        metric_objs.append(
            BinaryMetrics(
                accuracy=m["accuracy"], f1=m["f1"],
                roc_auc=m["roc_auc"], mcc=m["mcc"],
            )
        )
    stats = aggregate_metrics(metric_objs)
    out = {
        "model": "audiofuse_smote",
        "seeds": seeds,
        "per_seed": summaries,
        "aggregate": stats.as_dict(),
    }
    out_path = os.path.join(
        cfg["logging"]["output_dir"], "audiofuse_smote", "aggregate.json"
    )
    dump_json(out_path, out)
    print("\n=== audiofuse_smote aggregate over", len(seeds), "seeds ===")
    print(stats.pretty())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioFuse + SMOTE training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Dotted-key overrides, e.g. smote.warmup_epochs=20.",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Override the seed list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    run_smote_multi_seed(cfg)


if __name__ == "__main__":
    main()
