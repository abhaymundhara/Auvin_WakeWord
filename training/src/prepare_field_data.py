from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import librosa
import soundfile as sf

from .audio_utils import augment_clip, load_pcm, pad_leading_silence
from .constants import MIN_CLIP_SAMPLES, SAMPLE_RATE
from .paths import RAW_DIR, ROOT


def numbered_files(directory: Path, prefix: str) -> list[Path]:
    return sorted(directory.glob(f"{prefix}-*.wav"))


def is_training_file(path: Path) -> bool:
    return int(path.stem.rsplit("-", 1)[1]) % 5 != 0


def prepare_field_data(copies_per_positive: int) -> dict[str, int]:
    validation_dir = ROOT / "data" / "validation"
    positive_sources = [
        path
        for path in numbered_files(validation_dir / "positive", "auvin")
        if is_training_file(path)
    ]
    negative_sources = [
        path
        for path in numbered_files(validation_dir / "negative", "bg")
        if is_training_file(path)
    ]
    if not positive_sources or not negative_sources:
        raise SystemExit("Record numbered auvin-*.wav and bg-*.wav field clips first")

    positive_dir = RAW_DIR / "field_positives"
    negative_dir = RAW_DIR / "field_negatives"
    for directory in (positive_dir, negative_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.wav"):
            old.unlink()

    positive_count = 0
    for source in positive_sources:
        pcm = load_pcm(source)
        trimmed, _ = librosa.effects.trim(pcm, top_db=35)
        for copy_index in range(copies_per_positive):
            random.seed(f"{source.name}:{copy_index}")
            augmented = augment_clip(trimmed)
            aligned = pad_leading_silence(augmented, MIN_CLIP_SAMPLES)
            destination = positive_dir / f"{source.stem}-{copy_index:03d}.wav"
            sf.write(destination, aligned, SAMPLE_RATE)
            positive_count += 1

    for source in negative_sources:
        shutil.copy2(source, negative_dir / source.name)

    return {
        "positive_source_clips": len(positive_sources),
        "positive_augmented_clips": positive_count,
        "negative_source_clips": len(negative_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare field clips for training; keep every fifth clip held out"
    )
    parser.add_argument("--copies-per-positive", type=int, default=100)
    args = parser.parse_args()
    print(prepare_field_data(args.copies_per_positive))


if __name__ == "__main__":
    main()
