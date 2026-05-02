"""
Evaluation entry point for trained AudioFuse models.

Two modes are supported:

    --mode in_domain
        Reload the best checkpoint from a training run and report
        Accuracy / F1 / ROC-AUC / MCC on the held-out PhysioNet test
        set.  (Equivalent to the ``test_metrics`` already saved in
        ``summary.json`` -- this mode just re-runs the evaluation
        independently for verification.)

    --mode pascal
        Domain-generalisation protocol of the comparative-fusion report
        (Sec. IV.B): load PhysioNet-trained weights, fine-tune only the
        classification head on 80% of PASCAL Set B (patient-level
        split), then evaluate on the held-out 20%.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

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
    two_way_patient_split,
)
from AudioFuse.models.audiofuse import (
    AudioFuseLCF,
    EarlyFusion,
    TensorFusion,
    WeightedLateFusion,
    available_models,
    build_model,
)
from AudioFuse.models.baselines import SpectrogramBaseline, WaveformBaseline
from AudioFuse.training.train import evaluate, load_config, apply_overrides
from AudioFuse.utils.logging import (
    dump_json,
    get_logger,
    save_checkpoint,
    seed_everything,
)
from AudioFuse.utils.losses import build_loss
from AudioFuse.utils.metrics import (
    BinaryMetrics,
    MetricAccumulator,
)


# ---------------------------------------------------------------------------
# PASCAL dataset construction
# ---------------------------------------------------------------------------


def _build_pascal_datasets(
    cfg: Mapping[str, Any], seed: int
) -> Tuple[PCGSegmentDataset, PCGSegmentDataset, Tuple[float, float], List[Recording]]:
    """Build (train, eval) PASCAL Set B datasets using a patient-level split."""
    records = index_pascal(cfg["data"]["pascal_root"])
    train_frac = float(cfg["domain_generalization"]["pascal_train_fraction"])
    train_recs, eval_recs = two_way_patient_split(records, train_frac, seed=seed)

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

    cache = cfg["data"].get("cache_dir")
    train_ds = PCGSegmentDataset(train_recs, seg_cfg, mel_cfg, cache_dir=cache)
    eval_ds = PCGSegmentDataset(eval_recs, seg_cfg, mel_cfg, cache_dir=cache)

    cw = compute_class_weights(train_recs)
    return train_ds, eval_ds, cw, train_recs


# ---------------------------------------------------------------------------
# Head freezing
# ---------------------------------------------------------------------------


def _freeze_backbones(model: torch.nn.Module) -> None:
    """Freeze every parameter that is not part of the classification head.

    Knowledge of each model's structure is used here to identify which
    submodules constitute the "head" (the final 192-unit dense + sigmoid
    output, per Sec. III.B of the report).
    """
    if isinstance(model, (SpectrogramBaseline, WaveformBaseline)):
        encoder = model.encoder
        for p in encoder.parameters():
            p.requires_grad = False
    elif isinstance(model, AudioFuseLCF):
        for p in model.spec_encoder.parameters():
            p.requires_grad = False
        for p in model.wave_encoder.parameters():
            p.requires_grad = False
    elif isinstance(model, EarlyFusion):
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False
    elif isinstance(model, TensorFusion):
        for p in model.spec_encoder.parameters():
            p.requires_grad = False
        for p in model.wave_encoder.parameters():
            p.requires_grad = False
        for p in model.proj.parameters():
            p.requires_grad = False
    elif isinstance(model, WeightedLateFusion):
        for p in model.spec_encoder.parameters():
            p.requires_grad = False
        for p in model.wave_encoder.parameters():
            p.requires_grad = False
        # The gate logits sit outside the head, so freeze them too.
        model.gate_logits.requires_grad = False
    else:
        # Fallback: freeze everything except things named "head".
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False


# ---------------------------------------------------------------------------
# PASCAL fine-tune-and-evaluate
# ---------------------------------------------------------------------------


def finetune_on_pascal(
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    seed: int,
    out_dir: str,
) -> Dict[str, Any]:
    """Fine-tune ``model``'s head on PASCAL train split, evaluate on its test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger = get_logger(
        f"AudioFuse.pascal.seed{seed}",
        log_file=os.path.join(out_dir, "finetune.log"),
    )

    train_ds, eval_ds, class_weights, _ = _build_pascal_datasets(cfg, seed)
    logger.info(
        "PASCAL splits: train segments=%d, eval segments=%d, class_weights=%s",
        len(train_ds), len(eval_ds), class_weights,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    # Freeze every parameter outside the classification head.
    if cfg["domain_generalization"]["finetune_only_head"]:
        _freeze_backbones(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters during PASCAL fine-tune: %d",
                sum(p.numel() for p in trainable))

    loss_fn = build_loss(class_weights).to(device)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["domain_generalization"]["finetune_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["domain_generalization"]["finetune_epochs"])
    )

    best_metric = float("inf")
    best_metrics_obj: BinaryMetrics | None = None
    best_ckpt = os.path.join(out_dir, "pascal_best.pt")

    for epoch in range(1, int(cfg["domain_generalization"]["finetune_epochs"]) + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n = 0
        for waveform, spec, target in train_loader:
            waveform = waveform.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveform, spec)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * target.size(0)
            n += target.size(0)
        scheduler.step()
        train_loss = running_loss / max(n, 1)

        eval_loss, eval_metrics = evaluate(model, eval_loader, loss_fn, device)
        elapsed = time.time() - t0
        logger.info(
            "[PASCAL] Epoch %3d | train_loss=%.4f eval_loss=%.4f | "
            "AUC=%.4f F1=%.4f Acc=%.4f MCC=%.4f | %.1fs",
            epoch, train_loss, eval_loss,
            eval_metrics.roc_auc, eval_metrics.f1,
            eval_metrics.accuracy, eval_metrics.mcc, elapsed,
        )
        if eval_loss < best_metric:
            best_metric = eval_loss
            best_metrics_obj = eval_metrics
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "eval_loss": eval_loss,
                    "eval_metrics": eval_metrics.as_dict(),
                    "stage": "pascal",
                },
                best_ckpt,
            )

    assert best_metrics_obj is not None
    return {
        "best_eval_loss": float(best_metric),
        "best_eval_metrics": best_metrics_obj.as_dict(),
        "best_ckpt": best_ckpt,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_for_eval(model_name: str, cfg: Mapping[str, Any], ckpt_path: str) -> torch.nn.Module:
    """Build a model and load weights from ``ckpt_path``."""
    model = build_model(model_name, cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AudioFuse model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--ckpt", required=True, help="Path to a best.pt checkpoint.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("in_domain", "pascal"),
        help="in_domain rerun on PhysioNet test or pascal fine-tune-and-eval.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write outputs to (defaults next to the checkpoint).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    seed_everything(args.seed, deterministic=cfg.get("deterministic", True))

    out_dir = args.out_dir or os.path.dirname(args.ckpt)
    os.makedirs(out_dir, exist_ok=True)

    model = _load_for_eval(args.model, cfg, args.ckpt)

    if args.mode == "in_domain":
        # Re-derive the same PhysioNet test split for the seed and evaluate.
        from AudioFuse.training.train import build_physionet_bundle  # local import
        bundle = build_physionet_bundle(cfg, seed=args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = build_loss(bundle.class_weights).to(device)
        loader = DataLoader(
            bundle.test,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        loss, metrics = evaluate(model, loader, loss_fn, device)
        result = {"mode": "in_domain", "test_loss": loss,
                  "test_metrics": metrics.as_dict()}
    else:
        result = finetune_on_pascal(model, cfg, seed=args.seed, out_dir=out_dir)
        result["mode"] = "pascal"

    dump_json(os.path.join(out_dir, f"eval_{args.mode}.json"), result)
    print(result)


if __name__ == "__main__":
    main()
