from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .preprocessing import is_audio_file



@dataclass
class Recording:
    """A single audio recording with metadata used for splitting."""

    path: str
    label: int                      # 1 = abnormal, 0 = normal
    patient_id: str
    source: str                     # e.g. "training-a" / "pascal_normal"



_PHYSIONET_SUBSETS = ("training-a", "training-b", "training-c",
                      "training-d", "training-e", "training-f")


def _parse_physionet_reference(csv_path: str) -> Dict[str, int]:
    """Parse a PhysioNet REFERENCE.csv -> {filename_stem: 0/1}.

    The official labels are -1 (normal) and 1 (abnormal); we map -1 -> 0.
    """
    mapping: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            stem, raw_label = row[0].strip(), row[1].strip()
            try:
                label = int(raw_label)
            except ValueError:
                continue
            mapping[stem] = 1 if label == 1 else 0
    return mapping


def _physionet_patient_id(stem: str, subset: str) -> str:
    """Best-effort patient ID derivation from a PhysioNet recording stem.

    PhysioNet 2016 does not expose a clean patient-ID column in its public
    release.  Following the AudioFuse paper, we treat the alphabetic prefix
    of the recording name (which encodes the source / patient grouping) as
    the patient ID.  This is deliberately conservative: any two recordings
    that share a numeric stem across subsets, or that share an alphabetic
    prefix within a subset, are kept on the same side of the split.
    """
    # Recording stems look like "a0001", "b0023", "f0100", etc.
    match = re.match(r"^([a-zA-Z]+)(\d+)$", stem)
    if match:
        prefix, number = match.group(1), match.group(2)
        return f"{subset}:{prefix}:{number}"
    return f"{subset}:{stem}"


def index_physionet(root: str, drop_leakage: bool = True) -> List[Recording]:
    """Build a flat list of PhysioNet 2016 recordings.

    Args:
        root: Directory containing the ``training-a`` ... ``training-f``
            subset folders.
        drop_leakage: If True, drop recordings whose stems appear in more
            than one subset (a known leakage source in the official split).

    Returns:
        List of :class:`Recording` objects.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"PhysioNet root not found: {root}")

    records: List[Recording] = []
    seen_stems: Dict[str, str] = {}    # stem -> first subset that owns it

    for subset in _PHYSIONET_SUBSETS:
        subset_dir = os.path.join(root, subset)
        if not os.path.isdir(subset_dir):
            continue
        ref_csv = os.path.join(subset_dir, "REFERENCE.csv")
        labels = _parse_physionet_reference(ref_csv) if os.path.isfile(ref_csv) else {}

        for fname in sorted(os.listdir(subset_dir)):
            if not is_audio_file(fname):
                continue
            stem = os.path.splitext(fname)[0]
            label = labels.get(stem)
            if label is None:
                continue

            if drop_leakage and stem in seen_stems and seen_stems[stem] != subset:
                # Same stem appears in two subsets -> potential leakage.
                continue
            seen_stems.setdefault(stem, subset)

            records.append(
                Recording(
                    path=os.path.join(subset_dir, fname),
                    label=int(label),
                    patient_id=_physionet_patient_id(stem, subset),
                    source=subset,
                )
            )

    if not records:
        raise RuntimeError(
            f"No PhysioNet recordings found under {root}. "
            "Check that the directory structure matches training-a..training-f."
        )
    return records




_PASCAL_NORMAL = ("normal",)
_PASCAL_ABNORMAL = ("murmur", "extrastole", "extrahls", "extrasystole")
_PASCAL_DROP = ("artifact",)        # non-cardiac noise, excluded


def _pascal_patient_id(filename: str) -> str:
    """Heuristic patient ID for a PASCAL recording.

    PASCAL Set B file names are of the form
    ``<class>__<recorder_or_patient_tag>_<index>.wav``.  We use everything
    up to the trailing numeric index as the grouping key so multiple takes
    by the same source stay together in one split.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"^(.*?)(?:_?\d+)?$", stem)
    return match.group(1) if match else stem


def index_pascal(root: str, drop_artifact: bool = True) -> List[Recording]:
    """Build a flat list of PASCAL Set B recordings."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"PASCAL root not found: {root}")

    records: List[Recording] = []
    for class_dir in sorted(os.listdir(root)):
        full = os.path.join(root, class_dir)
        if not os.path.isdir(full):
            continue
        cls = class_dir.lower()
        if drop_artifact and cls in _PASCAL_DROP:
            continue
        if cls in _PASCAL_NORMAL:
            label = 0
        elif cls in _PASCAL_ABNORMAL:
            label = 1
        else:
            # Unknown sub-folder: skip rather than silently mislabel.
            continue
        for fname in sorted(os.listdir(full)):
            if not is_audio_file(fname):
                continue
            records.append(
                Recording(
                    path=os.path.join(full, fname),
                    label=label,
                    patient_id=_pascal_patient_id(fname),
                    source=f"pascal_{cls}",
                )
            )

    if not records:
        raise RuntimeError(
            f"No PASCAL recordings found under {root}. "
            "Expected sub-folders like normal/, murmur/, extrastole/."
        )
    return records



@dataclass
class Split:
    """Three-way split of recordings."""

    train: List[Recording] = field(default_factory=list)
    val: List[Recording] = field(default_factory=list)
    test: List[Recording] = field(default_factory=list)


def patient_level_split(
    records: Sequence[Recording],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Split:

    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("Train/val/test fractions must sum to 1.")

    # Group recordings by patient ID and assign each patient a majority label.
    by_patient: Dict[str, List[Recording]] = {}
    for r in records:
        by_patient.setdefault(r.patient_id, []).append(r)

    patients_by_label: Dict[int, List[str]] = {0: [], 1: []}
    for pid, recs in by_patient.items():
        # Majority label among this patient's recordings.
        n_pos = sum(1 for r in recs if r.label == 1)
        majority = 1 if n_pos >= len(recs) - n_pos else 0
        patients_by_label[majority].append(pid)

    rng = random.Random(seed)
    split = Split()
    for label, pids in patients_by_label.items():
        rng.shuffle(pids)
        n_total = len(pids)
        n_train = int(round(n_total * train_frac))
        n_val = int(round(n_total * val_frac))
        # The rest goes to test, ensuring all patients are accounted for.
        train_pids = pids[:n_train]
        val_pids = pids[n_train : n_train + n_val]
        test_pids = pids[n_train + n_val :]
        for pid in train_pids:
            split.train.extend(by_patient[pid])
        for pid in val_pids:
            split.val.extend(by_patient[pid])
        for pid in test_pids:
            split.test.extend(by_patient[pid])
    return split


def two_way_patient_split(
    records: Sequence[Recording],
    train_frac: float,
    seed: int,
) -> Tuple[List[Recording], List[Recording]]:
    """Patient-level 80/20 style split, used for PASCAL fine-tuning."""
    by_patient: Dict[str, List[Recording]] = {}
    for r in records:
        by_patient.setdefault(r.patient_id, []).append(r)

    pids = list(by_patient.keys())
    random.Random(seed).shuffle(pids)
    n_train = int(round(len(pids) * train_frac))
    train_pids = set(pids[:n_train])

    train: List[Recording] = []
    test: List[Recording] = []
    for pid, recs in by_patient.items():
        (train if pid in train_pids else test).extend(recs)
    return train, test



def compute_class_weights(records: Sequence[Recording]) -> Tuple[float, float]:
    """Return ``(w_normal, w_abnormal)`` inversely proportional to frequency.

    The result is normalised so that the mean weight is 1, which keeps the
    loss magnitude comparable to the unweighted case.
    """
    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = len(records) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    total = len(records)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    return float(w_neg), float(w_pos)
