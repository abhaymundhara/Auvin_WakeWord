from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .paths import BACKBONE_DIR, BACKBONE_FILES


def describe_session(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = []
    for item in session.get_inputs():
        inputs.append(
            {
                "name": item.name,
                "type": item.type,
                "shape": item.shape,
            }
        )
    outputs = []
    for item in session.get_outputs():
        outputs.append(
            {
                "name": item.name,
                "type": item.type,
                "shape": item.shape,
            }
        )
    return {"path": str(path), "inputs": inputs, "outputs": outputs}


def probe_melspec() -> dict:
    path = BACKBONE_DIR / "melspectrogram.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    results = {}

    for label, n_samples in [("isolated_1280", 1280), ("with_context_1760", 1760)]:
        x = np.random.randint(-1000, 1000, (1, n_samples), dtype=np.int16).astype(np.float32)
        out = session.run(None, {"input": x})[0]
        spec = np.squeeze(out)
        spec = spec / 10.0 + 2.0
        results[label] = {"input_samples": n_samples, "mel_frames": int(spec.shape[0])}

    return results


def probe_embedding() -> dict:
    path = BACKBONE_DIR / "embedding_model.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 76, 32, 1).astype(np.float32)
    out = session.run(None, {"input_1": x})[0]
    return {"input_shape": list(x.shape), "output_shape": list(out.shape)}


def probe_vad() -> dict:
    path = BACKBONE_DIR / "silero_vad.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    x = np.zeros((1, 512), dtype=np.float32)
    sr = np.array([16000], dtype=np.int64)
    feeds = {"input": x, "h": h, "c": c, "sr": sr}
    missing = [name for name in feeds if name not in input_names]
    if missing:
        feeds = {name: feeds[name] for name in input_names if name in feeds}
    out = session.run(None, feeds)
    return {
        "input_names": input_names,
        "output_names": [o.name for o in session.get_outputs()],
        "prob_shape": list(out[0].shape) if out else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe openWakeWord ONNX model I/O")
    parser.parse_args()

    missing = [f for f in BACKBONE_FILES if not (BACKBONE_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing backbone files: {missing}. Run download_models first.")

    report = {
        "models": [describe_session(BACKBONE_DIR / f) for f in BACKBONE_FILES],
        "melspec_smoke": probe_melspec(),
        "embedding_smoke": probe_embedding(),
        "vad_smoke": probe_vad(),
        "gates": {},
    }

    mel = report["melspec_smoke"]
    report["gates"]["mel_8_frames_with_context"] = mel["with_context_1760"]["mel_frames"] == 8
    report["gates"]["mel_5_frames_isolated"] = mel["isolated_1280"]["mel_frames"] == 5
    report["gates"]["embedding_96_dim"] = report["embedding_smoke"]["output_shape"][-1] == 96
    report["gates"]["vad_h_c_states"] = "h" in report["vad_smoke"]["input_names"]

    print(json.dumps(report, indent=2))

    failed = [k for k, v in report["gates"].items() if not v]
    if failed:
        raise SystemExit(f"Probe gates failed: {failed}")
    print("All probe gates passed.")


if __name__ == "__main__":
    main()
