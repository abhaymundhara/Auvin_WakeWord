from __future__ import annotations

import argparse
import urllib.request

from .paths import VOICES_DIR
from .synthesize import VOICE_PATHS, HF_BASE


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Piper voice models")
    parser.add_argument("--voice", action="append", dest="voices")
    args = parser.parse_args()
    voices = args.voices or list(VOICE_PATHS.keys())
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        rel = VOICE_PATHS[voice]
        for suffix in [".onnx", ".onnx.json"]:
            url = f"{HF_BASE}/{rel}{suffix}"
            dest = VOICES_DIR / f"{voice}{suffix}"
            if dest.exists() and dest.stat().st_size > 0:
                print(f"skip {dest.name}")
                continue
            print(f"download {dest.name}...")
            urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
