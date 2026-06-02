from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .paths import BACKBONE_DIR, BACKBONE_FILES, OPENWAKEWORD_RELEASE


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"download {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download openWakeWord v0.5.1 backbone ONNX models")
    parser.parse_args()
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    for name in BACKBONE_FILES:
        url = f"{OPENWAKEWORD_RELEASE}/{name}"
        download_file(url, BACKBONE_DIR / name)
    print(f"Backbone models ready in {BACKBONE_DIR}")


if __name__ == "__main__":
    main()
