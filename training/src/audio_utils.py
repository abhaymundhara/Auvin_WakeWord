from __future__ import annotations

import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_pcm(path: Path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)


def pad_leading_silence(pcm: np.ndarray, min_samples: int) -> np.ndarray:
    if len(pcm) >= min_samples:
        return pcm
    pad = min_samples - len(pcm)
    return np.concatenate([np.zeros(pad, dtype=np.float32), pcm])


def augment_clip(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    speed_jitter: float = 0.15,
    pitch_semitones: float = 2.0,
    gain_db: float = 6.0,
) -> np.ndarray:
    out = pcm.copy()
    speed = 1.0 + random.uniform(-speed_jitter, speed_jitter)
    if abs(speed - 1.0) > 0.001:
        out = librosa.effects.time_stretch(out, rate=speed)
    pitch = random.uniform(-pitch_semitones, pitch_semitones)
    if abs(pitch) > 0.01:
        out = librosa.effects.pitch_shift(out, sr=sample_rate, n_steps=pitch)
    gain = 10 ** (random.uniform(-gain_db, gain_db) / 20.0)
    out = np.clip(out * gain, -1.0, 1.0)
    return out.astype(np.float32)


def mix_background_noise(
    speech: np.ndarray,
    background: np.ndarray,
    snr_db: float,
    offset: int = 0,
) -> np.ndarray:
    """Mix background into speech at a target signal-to-noise ratio."""
    if len(background) == 0:
        return speech.astype(np.float32, copy=True)
    if len(background) < len(speech):
        repeats = int(np.ceil(len(speech) / len(background)))
        background = np.tile(background, repeats)
    max_offset = len(background) - len(speech)
    start = max(offset, 0) % (max_offset + 1)
    noise = background[start : start + len(speech)].astype(np.float32, copy=True)

    speech_rms = float(np.sqrt(np.mean(np.square(speech), dtype=np.float64)))
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if speech_rms == 0.0 or noise_rms == 0.0:
        return speech.astype(np.float32, copy=True)

    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    mixed = speech + noise * (target_noise_rms / noise_rms)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)
