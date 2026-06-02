from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from piper import PiperVoice

from .paths import CONFIG_PATH, VOICES_DIR


CANDIDATE_POSITIVES = [
    "auvin",
    "Auvin",
    "hey auvin",
    "Hey Auvin",
    "hey aw vin",
    "oh auvin",
    "au vin",
    "hey au vin",
]

CANDIDATE_HARD_NEGATIVES = [
    "oven",
    "often",
    "avin",
    "a vine",
    "oh vin",
    "hey oven",
    "hey often",
    "hey avin",
    "hey siri",
    "hey google",
    "hey alexa",
    "called auvin",
    "this is auvin",
    "to auvin",
]

DEFAULT_VOICE = "en_US-amy-medium"


def load_voice() -> PiperVoice:
    model_path = VOICES_DIR / f"{DEFAULT_VOICE}.onnx"
    config_path = VOICES_DIR / f"{DEFAULT_VOICE}.onnx.json"
    if not model_path.exists():
        raise SystemExit(
            f"Voice not found at {model_path}. Run synthesize with --download-voices first."
        )
    return PiperVoice.load(str(model_path), config_path=str(config_path))


def phonemize_phrases(phrases: list[str]) -> list[dict]:
    voice = load_voice()
    results = []
    for phrase in phrases:
        phones = voice.phonemize(phrase)
        results.append({"phrase": phrase, "phonemes": phones})
    return results


def group_positives(results: list[dict]) -> tuple[list[str], list[dict]]:
    groups: dict[str, list[str]] = {}
    for item in results:
        key = item["phonemes"]
        groups.setdefault(key, []).append(item["phrase"])
    if not groups:
        return [], results
    primary = max(groups.items(), key=lambda kv: len(kv[1]))[0]
    approved = groups[primary]
    rejected = [r for r in results if r["phrase"] not in approved]
    return approved, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Piper phonemizations for Auvin phrases")
    parser.add_argument("--write-config", action="store_true", help="Write approved lists to auvin.yaml")
    args = parser.parse_args()

    pos_results = phonemize_phrases(CANDIDATE_POSITIVES)
    neg_results = phonemize_phrases(CANDIDATE_HARD_NEGATIVES)
    approved_pos, rejected_pos = group_positives(pos_results)

    report = {
        "positives": pos_results,
        "hard_negatives": neg_results,
        "approved_positives": approved_pos,
        "rejected_positives": rejected_pos,
    }
    print(json.dumps(report, indent=2))

    if rejected_pos:
        print("\nRejected positive variants (inconsistent phonemization):")
        for item in rejected_pos:
            print(f"  - {item['phrase']!r} -> {item['phonemes']}")

    if args.write_config:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "wake_word": "auvin",
            "sample_rate": 16000,
            "positives": {
                "phrases": approved_pos,
                "target_count": 18000,
            },
            "hard_negatives": {
                "phrases": CANDIDATE_HARD_NEGATIVES,
                "target_count": 10000,
            },
            "random_negatives": {
                "dataset": "librispeech",
                "split": "train.clean.100",
                "target_count": 15000,
            },
            "piper_voices": [
                "en_US-amy-medium",
                "en_US-ryan-high",
                "en_US-lessac-medium",
                "en_US-hfc_female-medium",
                "en_US-libritts-high",
                "en_GB-alan-medium",
                "en_GB-jenny_dioco-medium",
            ],
            "augmentation": {
                "speed_jitter": 0.15,
                "pitch_semitones": 2.0,
                "gain_db": 6.0,
                "min_duration_sec": 2.4,
            },
            "training": {
                "hard_negative_weight": 4.0,
                "epochs": 50,
                "patience": 12,
                "batch_size": 1024,
                "learning_rate": 0.001,
            },
            "gates": {
                "recall_min": 0.97,
                "real_speech_fpr_max": 0.0005,
                "hard_negative_fpr_max": 0.03,
                "mean_pos_score_min": 0.95,
                "mean_neg_score_max": 0.02,
            },
        }
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"\nWrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
