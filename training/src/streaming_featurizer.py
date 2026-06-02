from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .constants import (
    CLASSIFIER_FRAMES,
    EMBEDDING_DIM,
    FRAME_SAMPLES,
    MEL_BUFFER_FRAMES,
    MEL_CALIBRATION_OFFSET,
    MEL_CALIBRATION_SCALE,
    MEL_CONTEXT_SAMPLES,
    MEL_FRAMES_PER_STEP,
)
from .paths import BACKBONE_DIR


def calibrate_mel(spec: np.ndarray) -> np.ndarray:
    return spec / MEL_CALIBRATION_SCALE + MEL_CALIBRATION_OFFSET


class StreamingFeaturizer:
    """Streaming featurizer aligned with the TypeScript inference pipeline."""

    def __init__(self, backbone_dir=BACKBONE_DIR):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.mel_session = ort.InferenceSession(
            str(backbone_dir / "melspectrogram.onnx"), sess_options=opts, providers=providers
        )
        self.emb_session = ort.InferenceSession(
            str(backbone_dir / "embedding_model.onnx"), sess_options=opts, providers=providers
        )
        self.reset()

    def reset(self) -> None:
        self.pcm_buffer: list[float] = []
        self.mel_buffer = np.ones((MEL_BUFFER_FRAMES, 32), dtype=np.float32)
        self.embedding_buffer: list[np.ndarray] = []

    def _run_melspec(self, pcm: np.ndarray) -> np.ndarray:
        x = pcm.astype(np.float32)[None, :]
        out = self.mel_session.run(None, {"input": x})[0]
        return calibrate_mel(np.squeeze(out))

    def _run_embedding(self, mel_window: np.ndarray) -> np.ndarray:
        x = mel_window.astype(np.float32)[None, :, :, None]
        out = self.emb_session.run(None, {"input_1": x})[0]
        return np.squeeze(out).astype(np.float32)

    def _append_pcm(self, samples: np.ndarray) -> None:
        self.pcm_buffer.extend(samples.tolist())

    def _process_frame(self) -> None:
        if len(self.pcm_buffer) < FRAME_SAMPLES:
            return
        context = self.pcm_buffer[-(FRAME_SAMPLES + MEL_CONTEXT_SAMPLES) :]
        if len(context) < FRAME_SAMPLES + MEL_CONTEXT_SAMPLES:
            pad = [0.0] * (FRAME_SAMPLES + MEL_CONTEXT_SAMPLES - len(context))
            context = pad + context
        mel = self._run_melspec(np.array(context, dtype=np.float32))
        if mel.shape[0] != MEL_FRAMES_PER_STEP:
            raise ValueError(f"Expected {MEL_FRAMES_PER_STEP} mel frames, got {mel.shape[0]}")
        self.mel_buffer = np.vstack((self.mel_buffer, mel))
        if self.mel_buffer.shape[0] > 970:
            self.mel_buffer = self.mel_buffer[-970:, :]

        ndx = len(self.mel_buffer)
        for i in range(1):
            end = ndx - 8 * i
            start = end - MEL_BUFFER_FRAMES
            if start < 0:
                continue
            window = self.mel_buffer[start:end]
            if window.shape[0] == MEL_BUFFER_FRAMES:
                emb = self._run_embedding(window)
                self.embedding_buffer.append(emb)

    def process_pcm(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32)
        offset = 0
        while offset + FRAME_SAMPLES <= len(pcm):
            chunk = pcm[offset : offset + FRAME_SAMPLES]
            self._append_pcm(chunk)
            self._process_frame()
            offset += FRAME_SAMPLES

    def extract_windows(self) -> np.ndarray:
        if len(self.embedding_buffer) < CLASSIFIER_FRAMES:
            return np.empty((0, CLASSIFIER_FRAMES, EMBEDDING_DIM), dtype=np.float32)
        emb = np.stack(self.embedding_buffer, axis=0)
        windows = []
        for i in range(CLASSIFIER_FRAMES - 1, len(emb)):
            windows.append(emb[i - CLASSIFIER_FRAMES + 1 : i + 1])
        if not windows:
            return np.empty((0, CLASSIFIER_FRAMES, EMBEDDING_DIM), dtype=np.float32)
        return np.stack(windows, axis=0).astype(np.float32)


def featurize_clip(pcm: np.ndarray, featurizer: StreamingFeaturizer | None = None) -> np.ndarray:
    own = featurizer is None
    if own:
        featurizer = StreamingFeaturizer()
    featurizer.reset()
    featurizer.process_pcm(pcm)
    windows = featurizer.extract_windows()
    if own:
        del featurizer
    return windows
