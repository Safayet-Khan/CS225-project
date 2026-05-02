from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

# Allow ``python AudioFuse/training/train.py`` to also work in addition
# to ``python -m AudioFuse.training.train``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AudioFuse.data.dataset import PCGSegmentDataset
from AudioFuse.data.preprocessing import MelConfig, SegmentConfig
from AudioFuse.data.splits import (
    Recording,
    compute_class_weights,
    index_pascal,
    index_physionet,
    patient_level_split,
)
from AudioFuse.models.audiofuse import (
    available_models,
    build_model,
    count_parameters,
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
# Config loading + CLI overrides
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _coerce(value: str) -> Any:
    """Best-effort conversion of a CLI string to its YAML-equivalent type."""
    lo = value.lower()
    if lo in ("true", "false"):
        return lo == "true"
    if lo in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``key.path=value`` style overrides to a nested dict."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override (expected key=value): {ov}")
        key, value = ov.split("=", 1)
        keys = key.split(".")
        cur: Any = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                raise KeyError(f"Override key not found in config: {key}")
            cur = cur[k]
        if keys[-1] not in cur:
            raise KeyError(f"Override key not found in config: {key}")
        cur[keys[-1]] = _coerce(value)
    return cfg


@dataclass
class DatasetBundle:
    """Train/val/test datasets plus the class weights for the loss."""

    train: PCGSegmentDataset
    val: PCGSegmentDataset
    test: PCGSegmentDataset
    class_weights: Tuple[float, float]
    train_records: List[Recording]


def build_physionet_bundle(cfg: Mapping[str, Any], seed: int) -> DatasetBundle:
    """Index PhysioNet, split by patient, and wrap the splits as Datasets."""
    records = index_physionet(cfg["data"]["physionet_root"], drop_leakage=True)
    split = patient_level_split(
        records,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=seed,
    )

    seg_cfg = SegmentConfig(
        sample_rate=cfg["audio"]["sample_rate"],
        clip_seconds=cfg["audio"]["segmentation"]["clip_seconds"],
        hop_seconds=cfg["audio"]["segmentation"]["hop_seconds"],
        segment_seconds=cfg["audio"]["segment_seconds"],
        pad_short=cfg["audio"]["segmentation"]["pad_short"],
    )
    mel_cfg = MelConfig(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["mel"]["n_fft"],
        hop_length=cfg["audio"]["mel"]["hop_length"],
        n_mels=cfg["audio"]["mel"]["n_mels"],
        fmin=cfg["audio"]["mel"]["fmin"],
        fmax=cfg["audio"]["mel"]["fmax"],
        power=cfg["audio"]["mel"]["power"],
        log_offset=cfg["audio"]["mel"]["log_offset"],
        target_size=tuple(cfg["audio"]["mel"]["target_size"]),
    )

    cache_dir = cfg["data"].get("cache_dir")
    train_ds = PCGSegmentDataset(split.train, seg_cfg, mel_cfg, cache_dir=cache_dir)
    val_ds = PCGSegmentDataset(split.val, seg_cfg, mel_cfg, cache_dir=cache_dir)
    test_ds = PCGSegmentDataset(split.test, seg_cfg, mel_cfg, cache_dir=cache_dir)

    if cfg["training"]["class_weights"]["auto"]:
        cw = compute_class_weights(split.train)
    else:
        cw = (
            float(cfg["training"]["class_weights"]["normal"]),
            float(cfg["training"]["class_weights"]["abnormal"]),
        )

    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        class_weights=cw,
        train_records=split.train,
    )




def build_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = cfg["training"]["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    name = cfg["training"]["scheduler"].lower()
    if name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg["training"]["cosine_T0"]),
            T_mult=int(cfg["training"]["cosine_Tmult"]),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg["training"]["epochs"])
        )
    raise ValueError(f"Unsupported scheduler: {name}")



def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: torch.amp.GradScaler | None,
) -> Tuple[float, BinaryMetrics]:
    model.train()
    acc = MetricAccumulator()
    running_loss = 0.0
    n = 0
    use_amp = scaler is not None

    for waveform, spec, target in loader:
        waveform = waveform.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(waveform, spec)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveform, spec)
            loss = loss_fn(logits, target)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = target.size(0)
        running_loss += loss.item() * bs
        n += bs
        acc.update(logits, target)

    metrics = acc.compute()
    return running_loss / max(n, 1), metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, BinaryMetrics]:
    """Run a full validation/test pass and return ``(loss, metrics)``."""
    with evaluating(model):
        acc = MetricAccumulator()
        running_loss = 0.0
        n = 0
        for waveform, spec, target in loader:
            waveform = waveform.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(waveform, spec)
            loss = loss_fn(logits, target)
            bs = target.size(0)
            running_loss += loss.item() * bs
            n += bs
            acc.update(logits, target)
    return running_loss / max(n, 1), acc.compute()



def train_one_seed(
    model_name: str,
    seed: int,
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
) -> Dict[str, Any]:
    """Train ``model_name`` once with the given random ``seed``.

    Returns a dict with the best validation metrics, the test metrics
    obtained from the best checkpoint, and bookkeeping info.
    """
    seed_everything(seed, deterministic=cfg.get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg["logging"]["output_dir"], model_name, seed)
    logger = get_logger(
        f"AudioFuse.{model_name}.seed{seed}",
        log_file=os.path.join(run_dir, "train.log"),
    )
    csv_logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))

    # Model.
    model = build_model(model_name, cfg["model"]).to(device)
    n_params = count_parameters(model)
    logger.info("Model %s built with %d trainable params.", model_name, n_params)

    # Loss / optimiser / scheduler.
    loss_fn = build_loss(bundle.class_weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info("Class weights: neg=%.4f, pos=%.4f", *bundle.class_weights)

    # Data loaders.
    train_loader = DataLoader(
        bundle.train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
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

    # Early stopping bookkeeping.
    es_cfg = cfg["training"]["early_stopping"]
    best_metric = float("inf") if es_cfg["mode"] == "min" else -float("inf")
    patience_left = int(es_cfg["patience"])
    best_ckpt_path = os.path.join(run_dir, "best.pt")

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, grad_clip, scaler
        )
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        # Track current LR for the CSV row.
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        csv_logger.log(
            {
                "epoch": epoch,
                "lr": current_lr,
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
            "Epoch %3d/%d | lr=%.2e | train_loss=%.4f val_loss=%.4f | "
            "val_AUC=%.4f val_F1=%.4f val_MCC=%.4f | %.1fs",
            epoch, epochs, current_lr,
            train_loss, val_loss,
            val_metrics.roc_auc, val_metrics.f1, val_metrics.mcc,
            elapsed,
        )

        # Early stopping + best-checkpoint tracking.
        monitor_value = val_loss if es_cfg["monitor"] == "val_loss" else -val_metrics.roc_auc
        improved = (
            monitor_value < best_metric if es_cfg["mode"] == "min" else monitor_value > best_metric
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
                    "model_name": model_name,
                    "seed": seed,
                    "class_weights": bundle.class_weights,
                },
                best_ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (no improvement).", epoch)
                break

    # Final test pass using the best checkpoint.
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)
    logger.info("Test results: loss=%.4f | %s", test_loss, test_metrics)

    summary = {
        "model": model_name,
        "seed": seed,
        "n_params": n_params,
        "best_val_loss": float(best_metric) if es_cfg["monitor"] == "val_loss" else float("nan"),
        "test_loss": test_loss,
        "test_metrics": test_metrics.as_dict(),
        "best_ckpt": best_ckpt_path,
    }
    dump_json(os.path.join(run_dir, "summary.json"), summary)
    return summary


def run_multi_seed(model_name: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Train ``model_name`` once per seed in ``cfg['seeds']``."""
    seeds = list(cfg["seeds"])
    summaries = []
    metric_objs: List[BinaryMetrics] = []
    bundle = None
    for seed in seeds:
        # Re-build datasets per seed so the patient-level split is reseeded.
        bundle = build_physionet_bundle(cfg, seed=seed)
        summary = train_one_seed(model_name, seed, cfg, bundle)
        summaries.append(summary)
        m = summary["test_metrics"]
        metric_objs.append(
            BinaryMetrics(
                accuracy=m["accuracy"],
                f1=m["f1"],
                roc_auc=m["roc_auc"],
                mcc=m["mcc"],
            )
        )

    stats = aggregate_metrics(metric_objs)
    out = {
        "model": model_name,
        "seeds": seeds,
        "per_seed": summaries,
        "aggregate": stats.as_dict(),
    }
    out_path = os.path.join(cfg["logging"]["output_dir"], model_name, "aggregate.json")
    dump_json(out_path, out)
    print(f"\n=== {model_name} aggregate over {len(seeds)} seeds ===")
    print(stats.pretty())
    return out



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioFuse training entry point.")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=available_models(),
        help="Which model to train.",
    )
    parser.add_argument(
        "--dataset",
        default="physionet",
        choices=("physionet",),
        help="Which dataset to train on (PASCAL is fine-tuning only).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key overrides, e.g. training.epochs=50. Repeatable.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override the seed list from the config (space-separated ints).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    run_multi_seed(args.model, cfg)


if __name__ == "__main__":
    main()
