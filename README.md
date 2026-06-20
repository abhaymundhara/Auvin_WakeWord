# Auvin Wake Word

Custom wake-word detector for **"Auvin"** and **"Hey Auvin"**, built on the [openWakeWord](https://github.com/dscripka/openWakeWord) frozen backbone and a tiny Conv1D classifier head (~122 KB). Follows the [Hey GAIA playbook](https://aryanranderiya.com/agent-convos/hey-gaia-wake-word-trained/).

## Layout

- `training/` — Python data synthesis, featurization, training, export
- `packages/wake-word/` — TypeScript inference library (Node / Web / React Native adapters)
- `models/backbone/` — Frozen openWakeWord ONNX (melspec, embedding, Silero VAD)
- `models/auvin.onnx` — Trained classifier (produced by training)
- `data/` — Raw audio and feature tensors (gitignored)

## Quick start

### 1. Python environment

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download backbone + voices, probe models

```bash
python -m src.download_models
python -m src.probe_onnx
python -m src.synthesize --download-voices
python -m src.phonemize_check --write-config
```

### 3. Synthesize training data (43k clips, ~8 GB raw audio)

```bash
# Smoke test first:
python -m src.synthesize --smoke
# Full dataset (18k positive, 10k hard-negative, 15k LibriSpeech):
python -m src.synthesize --download-voices
```

### 4. Featurize, train, export

```bash
python -m src.featurize --workers 8
python -m src.train 2>&1 | tee logs/train.log
python -m src.export_onnx
# Re-evaluate an existing checkpoint without retraining:
python -m src.train --evaluate-only
```

The full feature tensors currently require about 15 GB of disk space.

Training logs go to `training/logs/train.log`. Monitor with `tail -f training/logs/train.log`.

### 5. TypeScript inference

```bash
npm install
npm run build --workspace=@auvin/wake-word
npm test
```

### 6. Validation

```bash
python -m src.generate_validation_clips
python -m src.validate_clips
```

Add your own mic recordings under `data/validation/positive/` and `data/validation/negative/`.

On macOS, `rec` (from SoX) can create correctly formatted clips:

```bash
rec -r 16000 -c 1 -b 16 data/validation/positive/auvin-01.wav trim 0 3
rec -r 16000 -c 1 -b 16 data/validation/negative/background-01.wav trim 0 10
python -m src.validate_clips
```

Record at least 20 positive clips (`Auvin` and `Hey Auvin`) and 20 negative/noise clips across the microphones and distances you intend to support. Synthetic validation is a pipeline check, not a substitute for this field test.

## Production gates

| Metric | Target |
|--------|--------|
| Recall | ≥ 97% |
| Real-speech FPR | ≤ 0.05% |
| Hard-negative FPR | ≤ 3% |

Latest full-data checkpoint (threshold `0.5`, held-out synthetic windows):

| Metric | Result |
|--------|--------|
| Recall | 99.01% |
| Real-speech FPR | 0.013% |
| Hard-negative FPR | 2.16% |
| Mean positive score | 0.969 |
| Mean negative score | 0.017 |

All configured training gates pass. The separate nine-clip synthetic inference smoke test also passes with 4/4 positive detections and 0/5 false positives in both Python and Node. Real-microphone acceptance remains environment-specific and must be run before deployment.

## Detection pipeline

16 kHz audio → 80 ms frames → Silero VAD → melspectrogram (+480 sample context) → speech embedding → Conv1D classifier → threshold + debounce + cooldown.

## Runtime adapters

- Node: import `createNodeRuntime` from `@auvin/wake-word/node`.
- Browser: import `createWebRuntime` from `@auvin/wake-word/web`; `models/auvin.onnx` is self-contained for a single browser fetch.
- React Native: install the optional `onnxruntime-react-native` peer and import `createNativeRuntime` from `@auvin/wake-word/native`.

All adapters are passed to `createWakeWordDetector(runtime, options)`. The default live threshold is `0.6`, with two consecutive hits and a 1.5-second cooldown; tune these only against recordings from the target environment.

## License

Training code and `@auvin/wake-word` are project-local. Backbone models follow openWakeWord / Silero licenses.
