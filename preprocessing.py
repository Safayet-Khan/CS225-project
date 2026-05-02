from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class MelConfig:
    """Configuration for the log-Mel spectrogram pipeline."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 11025.0
    power: float = 2.0
    log_offset: float = 1e-6
    target_size: Tuple[int, int] = (224, 224)


@dataclass
class SegmentConfig:
    """Configuration for waveform segmentation."""

    sample_rate: int = 22050
    clip_seconds: float = 10.0          # segmentation window
    hop_seconds: float = 5.0            # 5 s overlap
    segment_seconds: float = 5.0        # actual model input length
    pad_short: bool = True


# ---------------------------------------------------------------------------
# Loading and resampling
# ---------------------------------------------------------------------------


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    # Mix down to mono if needed.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform.squeeze(0).contiguous()


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_waveform(waveform: torch.Tensor, cfg: SegmentConfig) -> List[torch.Tensor]:
    sr = cfg.sample_rate
    clip_len = int(round(cfg.clip_seconds * sr))
    hop_len = int(round(cfg.hop_seconds * sr))
    out_len = int(round(cfg.segment_seconds * sr))

    if waveform.numel() == 0:
        return []

    # If the recording is shorter than one clip, optionally pad it once.
    if waveform.numel() < clip_len:
        if not cfg.pad_short:
            return []
        pad_amount = clip_len - waveform.numel()
        waveform = F.pad(waveform, (0, pad_amount))

    starts = list(range(0, waveform.numel() - clip_len + 1, hop_len))
    # Always include the trailing window for long recordings whose last hop
    # would otherwise be dropped.
    if not starts:
        starts = [0]
    elif starts[-1] + clip_len < waveform.numel():
        starts.append(waveform.numel() - clip_len)

    clips: List[torch.Tensor] = []
    for s in starts:
        clip = waveform[s : s + clip_len]
        # Truncate or zero-pad the clip to the model's input length.
        clip = _fix_length(clip, out_len)
        clips.append(clip.contiguous())
    return clips


def _fix_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Centre-crop or right-pad a 1D waveform to ``target_len`` samples."""
    n = waveform.numel()
    if n == target_len:
        return waveform
    if n > target_len:
        start = (n - target_len) // 2
        return waveform[start : start + target_len]
    return F.pad(waveform, (0, target_len - n))


class LogMelSpectrogram:

    def __init__(self, cfg: MelConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._mel = AT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=cfg.power,
        ).to(self.device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 1:
            raise ValueError(
                f"LogMelSpectrogram expects a 1D waveform, got shape {tuple(waveform.shape)}"
            )
        x = waveform.to(self.device).unsqueeze(0)            # (1, T)
        mel = self._mel(x)                                   # (1, n_mels, frames)
        log_mel = torch.log(mel + self.cfg.log_offset)
        # Per-clip standardisation -> stable inputs for the ViT branch.
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        # Resize to 224 x 224 for the ViT.
        log_mel = F.interpolate(
            log_mel.unsqueeze(0),                            # (1, 1, n_mels, frames)
            size=self.cfg.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                         # (1, H, W)
        return log_mel.contiguous()


# ---------------------------------------------------------------------------
# Convenience: end-to-end preprocessing of a single recording
# ---------------------------------------------------------------------------


def preprocess_recording(
    path: str,
    seg_cfg: SegmentConfig,
    mel_cfg: MelConfig,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of ``(waveform, log_mel)`` pairs for one recording.

    Each waveform has shape ``(segment_samples,)`` and each log-Mel has
    shape ``(1, 224, 224)``.  Both branches see the same underlying audio.
    """
    waveform = load_audio(path, target_sr=seg_cfg.sample_rate)
    clips = segment_waveform(waveform, seg_cfg)
    if not clips:
        return []
    logmel = LogMelSpectrogram(mel_cfg)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for clip in clips:
        spec = logmel(clip)
        pairs.append((clip.float(), spec.float()))
    return pairs


# ---------------------------------------------------------------------------
# Helpers used by the splits / dataset code
# ---------------------------------------------------------------------------


def is_audio_file(name: str) -> bool:
    """Return True for recognised audio extensions."""
    return name.lower().endswith((".wav", ".flac", ".aiff", ".au"))


def numpy_from_tensor(x: torch.Tensor) -> np.ndarray:
    """Detach + move to CPU + numpy view."""
    return x.detach().cpu().numpy()
