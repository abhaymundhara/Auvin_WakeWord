from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from piper import PiperVoice

from .paths import CONFIG_PATH, VOICES_DIR


CANDIDATE_POSITIVES = [
    "aw win",
    "aw-win",
    "hey aw win",
    "hey aw-win",
    "oh aw win",
    "okay aw win",
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
    "Owen",
    "a win",
    "I win",
    "we win",
    "all win",
    "awning",
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
        if isinstance(phones, list):
            phones_key = " ".join(str(p) for p in phones)
        else:
            phones_key = str(phones)
        results.append({"phrase": phrase, "phonemes": phones_key, "phonemes_raw": phones})
    return results


def approve_positives(results: list[dict]) -> tuple[list[str], list[dict]]:
    """Keep phrase variants unless phonemization contains known bad patterns."""
    bad_markers = ["d\u0292a\u026a", "dʒaɪ"]
    approved: list[str] = []
    rejected: list[dict] = []
    for item in results:
        phones = item["phonemes"]
        if any(marker in phones for marker in bad_markers):
            rejected.append(item)
        else:
            approved.append(item["phrase"])
    return approved, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Piper phonemizations for Auvin phrases")
    parser.add_argument("--write-config", action="store_true", help="Write approved lists to auvin.yaml")
    args = parser.parse_args()

    pos_results = phonemize_phrases(CANDIDATE_POSITIVES)
    neg_results = phonemize_phrases(CANDIDATE_HARD_NEGATIVES)
    approved_pos, rejected_pos = approve_positives(pos_results)

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
                "dataset": "openslr/librispeech_asr",
                "subset": "clean",
                "split": "train.100",
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
                "background_noise_probability": 0.25,
                "background_snr_db_min": 15.0,
                "background_snr_db_max": 30.0,
            },
            "training": {
                "hard_negative_weight": 40.0,
                "field_negative_weight": 40.0,
                "epochs": 35,
                "patience": 8,
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
