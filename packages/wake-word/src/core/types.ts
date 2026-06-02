export type TensorMap = Record<string, Float32Array | BigInt64Array>;

export interface InferenceSessionLike {
  run(feeds: TensorMap, outputNames?: string[]): Promise<TensorMap>;
}

export interface InferenceRuntime {
  createSession(modelPath: string): Promise<InferenceSessionLike>;
}

export interface WakeWordDetectorOptions {
  backboneDir: string;
  classifierPath: string;
  threshold?: number;
  debounceHits?: number;
  cooldownMs?: number;
  onDetection?: (event: DetectionEvent) => void;
}

export interface DetectionEvent {
  score: number;
  timestampMs: number;
  timeToWakeMs: number;
}

export interface WakeWordDetector {
  load(): Promise<void>;
  processFrame(frame: Float32Array): Promise<number | null>;
  reset(): void;
  dispose(): void;
}

export interface FrameProcessorResult {
  score: number | null;
  detected: boolean;
}
