from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .preprocessing import (
    LogMelSpectrogram,
    MelConfig,
    SegmentConfig,
    load_audio,
    segment_waveform,
)
from .splits import Recording


@dataclass
class _SegmentRef:
    """Pointer from a flat segment index to its source recording + offset."""

    record_index: int       # index into the underlying Recording list
    segment_index: int      # which segment within that recording
    label: int


class PCGSegmentDataset(Dataset):
    """Segment-level Dataset for PhysioNet 2016 / PASCAL Set B.

    Args:
        records: List of :class:`Recording` for this split.
        seg_cfg: Segmentation configuration.
        mel_cfg: Log-Mel configuration.
        cache_dir: Optional directory in which to cache per-recording
            preprocessing results (one ``.pt`` file per recording).
            Disabled if ``None``.
        in_memory: If True, keep all decoded segments in RAM after the
            first access for fast subsequent epochs.
    """

    def __init__(
        self,
        records: Sequence[Recording],
        seg_cfg: SegmentConfig,
        mel_cfg: MelConfig,
        cache_dir: Optional[str] = None,
        in_memory: bool = True,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("PCGSegmentDataset requires at least one recording.")
        self._records: List[Recording] = list(records)
        self._seg_cfg = seg_cfg
        self._mel_cfg = mel_cfg
        self._cache_dir = cache_dir
        self._in_memory = in_memory

        # The Mel transform is small and stateless; one instance is fine.
        self._logmel = LogMelSpectrogram(mel_cfg, device="cpu")

        # Per-recording memoisation of the decoded segments.  Each entry is
        # a list of (waveform, spectrogram) tensor pairs.
        self._mem: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        self._seg_refs: Optional[List[_SegmentRef]] = None

    # ---------------- private ----------------

    def _ensure_index(self) -> None:
        """Materialise every recording once to know exact segment counts.

        This is a one-time cost.  Subsequent access uses the cached
        segments either in-memory or via on-disk pickles.
        """
        if self._seg_refs is not None:
            return
        seg_refs: List[_SegmentRef] = []
        for ri, rec in enumerate(self._records):
            segs = self._load_segments(ri)
            for si in range(len(segs)):
                seg_refs.append(_SegmentRef(ri, si, rec.label))
        self._seg_refs = seg_refs

    def _cache_path(self, record_index: int) -> Optional[str]:
        if self._cache_dir is None:
            return None
        rec = self._records[record_index]
        # Use a stable hash of the absolute path for the cache filename.
        key = str(abs(hash(os.path.abspath(rec.path))))
        return os.path.join(self._cache_dir, f"{key}.pkl")

    def _load_segments(
        self, record_index: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return decoded ``(waveform, spec)`` segments for one recording."""
        if record_index in self._mem:
            return self._mem[record_index]

        cache_path = self._cache_path(record_index)
        if cache_path is not None and os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                segs = pickle.load(f)
        else:
            rec = self._records[record_index]
            wf = load_audio(rec.path, target_sr=self._seg_cfg.sample_rate)
            clips = segment_waveform(wf, self._seg_cfg)
            segs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for clip in clips:
                spec = self._logmel(clip)
                # Store as float32 to keep RAM use predictable.
                segs.append((clip.float(), spec.float()))
            if cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(segs, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self._in_memory:
            self._mem[record_index] = segs
        return segs

    def __len__(self) -> int:
        self._ensure_index()
        assert self._seg_refs is not None
        return len(self._seg_refs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_index()
        assert self._seg_refs is not None
        ref = self._seg_refs[idx]
        segs = self._load_segments(ref.record_index)
        waveform, spec = segs[ref.segment_index]
        # Add a channel dimension to the waveform so it's (1, T).
        waveform = waveform.unsqueeze(0)
        label = torch.tensor(ref.label, dtype=torch.long)
        return waveform, spec, label


    @property
    def labels(self) -> List[int]:
        """All segment-level labels (in the same order as __getitem__)."""
        self._ensure_index()
        assert self._seg_refs is not None
        return [ref.label for ref in self._seg_refs]

    @property
    def num_recordings(self) -> int:
        return len(self._records)
