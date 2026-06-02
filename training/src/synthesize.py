from __future__ import annotations

import argparse
import json
import random
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from piper import PiperVoice
from tqdm import tqdm

from .audio_utils import augment_clip, pad_leading_silence
from .constants import MIN_CLIP_SAMPLES, SAMPLE_RATE
from .paths import CONFIG_PATH, DATA_DIR, RAW_DIR, VOICES_DIR

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

VOICE_PATHS = {
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high",
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en_US-hfc_female-medium": "en/en_US/hfc_female/medium/en_US-hfc_female-medium",
    "en_US-libritts-high": "en/en_US/libritts/high/en_US-libritts-high",
    "en_GB-alan-medium": "en/en_GB/alan/medium/en_GB-alan-medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
}


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_voices(voices: list[str]) -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        rel = VOICE_PATHS[voice]
        for suffix in [".onnx", ".onnx.json"]:
            url = f"{HF_BASE}/{rel}{suffix}"
            dest = VOICES_DIR / f"{voice}{suffix}"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            print(f"Downloading {dest.name}...")
            urllib.request.urlretrieve(url, dest)


def synthesize_one(args: tuple) -> dict | None:
    phrase, voice_name, out_dir, min_samples, aug_cfg, bucket = args
    try:
        model_path = VOICES_DIR / f"{voice_name}.onnx"
        config_path = VOICES_DIR / f"{voice_name}.onnx.json"
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        length_scale = random.uniform(0.75, 1.25)
        noise_scale = random.uniform(0.4, 1.0)
        noise_w_scale = random.uniform(0.4, 1.0)
        audio_chunks = []
        for chunk in voice.synthesize(
            phrase,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        ):
            audio_chunks.append(chunk.audio_float_array)
        pcm = np.concatenate(audio_chunks).astype(np.float32)
        pcm = augment_clip(
            pcm,
            speed_jitter=aug_cfg["speed_jitter"],
            pitch_semitones=aug_cfg["pitch_semitones"],
            gain_db=aug_cfg["gain_db"],
        )
        pcm = pad_leading_silence(pcm, min_samples)
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        sf.write(str(out_path), pcm, SAMPLE_RATE)
        return {"path": str(out_path), "phrase": phrase, "voice": voice_name, "bucket": bucket}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "phrase": phrase, "voice": voice_name}


def count_existing(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*.wav")))


def synthesize_bucket(
    phrases: list[str],
    voices: list[str],
    target_count: int,
    out_dir: Path,
    aug_cfg: dict,
    bucket: str,
    workers: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = count_existing(out_dir)
    remaining = max(0, target_count - existing)
    if remaining == 0:
        print(f"{bucket}: already have {existing} clips")
        return existing

    jobs = []
    for _ in range(remaining):
        phrase = random.choice(phrases)
        voice = random.choice(voices)
        jobs.append((phrase, voice, out_dir, MIN_CLIP_SAMPLES, aug_cfg, bucket))

    created = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(synthesize_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=bucket):
            result = fut.result()
            if result and "path" in result:
                created += 1
            elif result and "error" in result:
                print(f"error: {result}")

    total = count_existing(out_dir)
    print(f"{bucket}: {total} clips ({created} new)")
    return total


def download_librispeech(target_count: int, out_dir: Path) -> int:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = count_existing(out_dir)
    if existing >= target_count:
        print(f"random_negatives: already have {existing} clips")
        return existing

    ds = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
    needed = target_count - existing
    saved = 0
    for row in tqdm(ds, desc="librispeech", total=needed):
        if saved >= needed:
            break
        audio = row["audio"]
        pcm = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != SAMPLE_RATE:
            import librosa

            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=SAMPLE_RATE)
        pcm = pad_leading_silence(pcm, MIN_CLIP_SAMPLES)
        out_path = out_dir / f"ls_{uuid.uuid4().hex}.wav"
        sf.write(str(out_path), pcm, SAMPLE_RATE)
        saved += 1
    total = count_existing(out_dir)
    print(f"random_negatives: {total} clips")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Auvin wake word training data")
    parser.add_argument("--download-voices", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Synthesize 5 clips per bucket only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-librispeech", action="store_true")
    args = parser.parse_args()

    config = load_config()
    voices = config["piper_voices"]
    aug_cfg = config["augmentation"]

    if args.download_voices:
        download_voices(voices)

    pos_target = 5 if args.smoke else config["positives"]["target_count"]
    hard_target = 5 if args.smoke else config["hard_negatives"]["target_count"]
    rand_target = 5 if args.smoke else config["random_negatives"]["target_count"]

    synthesize_bucket(
        config["positives"]["phrases"],
        voices,
        pos_target,
        RAW_DIR / "positives",
        aug_cfg,
        "positives",
        args.workers,
    )
    synthesize_bucket(
        config["hard_negatives"]["phrases"],
        voices,
        hard_target,
        RAW_DIR / "hard_negatives",
        aug_cfg,
        "hard_negatives",
        args.workers,
    )

    if not args.skip_librispeech:
        download_librispeech(rand_target, RAW_DIR / "random_negatives")

    manifest = {
        "positives": count_existing(RAW_DIR / "positives"),
        "hard_negatives": count_existing(RAW_DIR / "hard_negatives"),
        "random_negatives": count_existing(RAW_DIR / "random_negatives"),
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
