from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio_utils import load_pcm
from .paths import FEATURES_DIR, RAW_DIR
from .streaming_featurizer import StreamingFeaturizer


LABEL_MAP = {
    "positives": (1, "positive"),
    "hard_negatives": (0, "hard_negative"),
    "random_negatives": (0, "random_negative"),
}


def featurize_file(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path_str, label, weight_kind = args
    try:
        pcm = load_pcm(Path(path_str))
        featurizer = StreamingFeaturizer()
        windows = featurize_clip_worker(pcm, featurizer)
        if windows.shape[0] == 0:
            return None
        labels = np.full((windows.shape[0],), label, dtype=np.int64)
        weights = np.full((windows.shape[0],), 1.0 if weight_kind == "positive" else (4.0 if weight_kind == "hard_negative" else 1.0), dtype=np.float32)
        return windows, labels, weights
    except Exception:
        return None


def featurize_clip_worker(pcm: np.ndarray, featurizer: StreamingFeaturizer) -> np.ndarray:
    featurizer.reset()
    featurizer.process_pcm(pcm)
    return featurizer.extract_windows()


def collect_files(raw_dir: Path) -> list[tuple[str, int, str]]:
    files: list[tuple[str, int, str]] = []
    for bucket, (label, kind) in LABEL_MAP.items():
        bucket_dir = raw_dir / bucket
        if not bucket_dir.exists():
            continue
        for wav in bucket_dir.glob("*.wav"):
            files.append((str(wav), label, kind))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Featurize raw audio into training windows")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0 = all)")
    args = parser.parse_args()

    files = collect_files(RAW_DIR)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No wav files found under {RAW_DIR}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(featurize_file, item) for item in files]
        empty = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="featurize"):
            result = fut.result()
            if result is None:
                empty += 1
                continue
            windows, labels, weights = result
            all_windows.append(windows)
            all_labels.append(labels)
            all_weights.append(weights)

    if not all_windows:
        raise SystemExit("No training windows produced. Check padding and featurizer alignment.")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    w = np.concatenate(all_weights, axis=0).astype(np.float32)

    np.save(FEATURES_DIR / "X.npy", X)
    np.save(FEATURES_DIR / "y.npy", y)
    np.save(FEATURES_DIR / "w.npy", w)

    meta = {
        "windows": int(X.shape[0]),
        "shape": list(X.shape),
        "empty_clips": empty,
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
    }
    with (FEATURES_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
