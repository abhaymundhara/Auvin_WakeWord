from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio_utils import load_pcm, mix_background_noise
from .paths import CONFIG_PATH, FEATURES_DIR, RAW_DIR
from .streaming_featurizer import StreamingFeaturizer


LABEL_MAP = {
    "positives": (1, "positive"),
    "field_positives": (1, "field_positive"),
    "hard_negatives": (0, "hard_negative"),
    "field_negatives": (0, "field_negative"),
    "random_negatives": (0, "random_negative"),
}


def featurize_file(
    args: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    path_str, label, weight_kind, noise_path, snr_db, noise_offset = args
    try:
        pcm = load_pcm(Path(path_str))
        if noise_path is not None:
            background = load_pcm(Path(noise_path))
            pcm = mix_background_noise(pcm, background, snr_db, noise_offset)
        featurizer = StreamingFeaturizer()
        windows = featurize_clip_worker(pcm, featurizer)
        if windows.shape[0] == 0:
            return None
        labels, weights, kinds = make_targets(windows.shape[0], label, weight_kind)
        return windows, labels, weights, kinds
    except Exception:
        return None


def make_targets(
    window_count: int,
    label: int,
    weight_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros((window_count,), dtype=np.int64)
    weights = np.ones((window_count,), dtype=np.float32)
    kinds = np.full((window_count,), weight_kind, dtype="<U16")

    if label == 1:
        # Synthetic positives are padded with leading silence, so only the
        # final window is aligned with the completed wake phrase.
        kinds.fill("positive_context")
        labels[-1] = 1
        kinds[-1] = "positive"
    elif weight_kind in {"hard_negative", "field_negative"}:
        weights.fill(4.0)

    return labels, weights, kinds


def featurize_clip_worker(pcm: np.ndarray, featurizer: StreamingFeaturizer) -> np.ndarray:
    featurizer.reset()
    featurizer.process_pcm(pcm)
    return featurizer.extract_windows()


def collect_files(raw_dir: Path, augmentation: dict) -> list[tuple]:
    files: list[tuple] = []
    noise_files = sorted((raw_dir / "random_negatives").glob("*.wav"))
    rng = random.Random(42)
    for bucket, (label, kind) in LABEL_MAP.items():
        bucket_dir = raw_dir / bucket
        if not bucket_dir.exists():
            continue
        for wav in bucket_dir.glob("*.wav"):
            noise_path = None
            snr_db = 0.0
            noise_offset = 0
            if (
                label == 1
                and noise_files
                and rng.random() < augmentation["background_noise_probability"]
            ):
                noise = rng.choice(noise_files)
                noise_path = str(noise)
                snr_db = rng.uniform(
                    augmentation["background_snr_db_min"],
                    augmentation["background_snr_db_max"],
                )
                noise_offset = rng.randrange(0, max(noise.stat().st_size // 4, 1))
            files.append((str(wav), label, kind, noise_path, snr_db, noise_offset))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Featurize raw audio into training windows")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0 = all)")
    args = parser.parse_args()

    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    files = collect_files(RAW_DIR, config["augmentation"])
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No wav files found under {RAW_DIR}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_kinds: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(featurize_file, item) for item in files]
        empty = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="featurize"):
            result = fut.result()
            if result is None:
                empty += 1
                continue
            windows, labels, weights, kinds = result
            all_windows.append(windows)
            all_labels.append(labels)
            all_weights.append(weights)
            all_kinds.append(kinds)

    if not all_windows:
        raise SystemExit("No training windows produced. Check padding and featurizer alignment.")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    w = np.concatenate(all_weights, axis=0).astype(np.float32)
    k = np.concatenate(all_kinds, axis=0)

    np.save(FEATURES_DIR / "X.npy", X)
    np.save(FEATURES_DIR / "y.npy", y)
    np.save(FEATURES_DIR / "w.npy", w)
    np.save(FEATURES_DIR / "k.npy", k)

    meta = {
        "windows": int(X.shape[0]),
        "shape": list(X.shape),
        "empty_clips": empty,
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "kinds": {kind: int((k == kind).sum()) for kind in np.unique(k)},
    }
    with (FEATURES_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
