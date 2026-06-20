from __future__ import annotations

import argparse

import torch

from .model import ConvWakeHead
from .paths import CLASSIFIER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained classifier to ONNX")
    parser.parse_args()

    pt_path = CLASSIFIER_PATH.with_suffix(".pt")
    if not pt_path.exists():
        raise SystemExit(f"Missing weights: {pt_path}. Run train first.")

    model = ConvWakeHead()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 16, 96)
    torch.onnx.export(
        model,
        dummy,
        str(CLASSIFIER_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
    )
    size_kb = CLASSIFIER_PATH.stat().st_size / 1024
    print(f"Exported {CLASSIFIER_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
