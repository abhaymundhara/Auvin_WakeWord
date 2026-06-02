from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from .paths import BACKBONE_DIR, CLASSIFIER_PATH, ROOT
from .streaming_featurizer import StreamingFeaturizer


def score_clip(pcm: np.ndarray) -> float:
    featurizer = StreamingFeaturizer()
    featurizer.reset()
    offset = 0
    frame = 1280
    max_score = 0.0
    while offset + frame <= len(pcm):
        chunk = pcm[offset : offset + frame]
        featurizer.process_pcm(chunk)
        offset += frame
    windows = featurizer.extract_windows()
    if windows.shape[0] == 0:
        return 0.0

    session = ort.InferenceSession(str(CLASSIFIER_PATH), providers=["CPUExecutionProvider"])
    for i in range(windows.shape[0]):
        x = windows[i : i + 1].astype(np.float32)
        out = session.run(None, {"input": x})[0]
        max_score = max(max_score, float(out.squeeze()))
    return max_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate wake word clips")
    parser.add_argument("--clips-dir", type=Path, default=ROOT / "data" / "validation")
    args = parser.parse_args()

    if not CLASSIFIER_PATH.exists():
        raise SystemExit("Train and export model first")

    results = []
    clips_dir = args.clips_dir
    if not clips_dir.exists():
        clips_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {clips_dir}. Add positive/negative wav files and re-run.")
        return

    for wav in sorted(clips_dir.glob("**/*.wav")):
        pcm, sr = sf.read(str(wav), dtype="float32")
        if sr != 16000:
            import librosa

            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000)
        score = score_clip(pcm)
        label = "positive" if "positive" in wav.parts else "negative"
        results.append({"file": str(wav.relative_to(ROOT)), "score": score, "label": label})

    if not results:
        print("No validation clips found.")
        return

    pos = [r for r in results if r["label"] == "positive"]
    neg = [r for r in results if r["label"] == "negative"]
    summary = {
        "clips": len(results),
        "positive_recall": sum(r["score"] >= 0.5 for r in pos) / max(len(pos), 1),
        "negative_fp_rate": sum(r["score"] >= 0.5 for r in neg) / max(len(neg), 1),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
