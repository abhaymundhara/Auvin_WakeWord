export const SAMPLE_RATE = 16000;
export const FRAME_SAMPLES = 1280;
export const MEL_CONTEXT_SAMPLES = 480;
export const MEL_FRAMES_PER_STEP = 8;
export const MEL_BUFFER_FRAMES = 76;
export const EMBEDDING_DIM = 96;
export const CLASSIFIER_FRAMES = 16;
export const WARMUP_FRAMES = 26;
export const MIN_CLIP_SAMPLES = Math.floor(2.4 * SAMPLE_RATE);

export const MEL_CALIBRATION_SCALE = 10.0;
export const MEL_CALIBRATION_OFFSET = 2.0;

export const VAD_H_DIM = 64;
export const VAD_C_DIM = 64;
export const VAD_FRAME_SAMPLES = 512;

export const DEFAULT_THRESHOLD = 0.6;
export const DEFAULT_DEBOUNCE_HITS = 2;
export const DEFAULT_COOLDOWN_MS = 1500;
