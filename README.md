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

### 3. Synthesize training data (~43k clips, ~5 GB)

```bash
python -m src.synthesize --download-voices
# Smoke test first:
python -m src.synthesize --smoke
python -m src.featurize
```

### 4. Featurize, train, export

```bash
python -m src.featurize --workers 8
python -m src.train 2>&1 | tee logs/train.log
python -m src.export_onnx
```

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

## Production gates

| Metric | Target |
|--------|--------|
| Recall | ≥ 97% |
| Real-speech FPR | ≤ 0.05% |
| Hard-negative FPR | ≤ 3% |

## Detection pipeline

16 kHz audio → 80 ms frames → Silero VAD → melspectrogram (+480 sample context) → speech embedding → Conv1D classifier → threshold + debounce + cooldown.

## License

Training code and `@auvin/wake-word` are project-local. Backbone models follow openWakeWord / Silero licenses.
