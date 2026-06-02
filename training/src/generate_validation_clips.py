from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import augment_clip, pad_leading_silence
from .constants import MIN_CLIP_SAMPLES, SAMPLE_RATE
from .paths import ROOT


def synth_validation_clips() -> None:
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    from .paths import VOICES_DIR

    out_dir = ROOT / "data" / "validation"
    pos_dir = out_dir / "positive"
    neg_dir = out_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    voice = PiperVoice.load(
        str(VOICES_DIR / "en_US-amy-medium.onnx"),
        config_path=str(VOICES_DIR / "en_US-amy-medium.onnx.json"),
    )

    positives = ["auvin", "Auvin", "hey auvin", "Hey Auvin"]
    negatives = ["oven", "often", "avin", "hey siri", "hello there"]

    for i, phrase in enumerate(positives):
        syn_config = SynthesisConfig(
            length_scale=random.uniform(0.85, 1.15),
            noise_scale=random.uniform(0.5, 0.9),
            noise_w_scale=random.uniform(0.5, 0.9),
        )
        chunks = [c.audio_float_array for c in voice.synthesize(phrase, syn_config=syn_config)]
        pcm = pad_leading_silence(np.concatenate(chunks).astype(np.float32), MIN_CLIP_SAMPLES)
        pcm = augment_clip(pcm)
        sf.write(str(pos_dir / f"pos_{i:02d}.wav"), pcm, SAMPLE_RATE)

    for i, phrase in enumerate(negatives):
        chunks = [c.audio_float_array for c in voice.synthesize(phrase)]
        pcm = pad_leading_silence(np.concatenate(chunks).astype(np.float32), MIN_CLIP_SAMPLES)
        sf.write(str(neg_dir / f"neg_{i:02d}.wav"), pcm, SAMPLE_RATE)

    print(f"Wrote validation clips to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    synth_validation_clips()


if __name__ == "__main__":
    main()
