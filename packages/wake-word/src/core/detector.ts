import {
  DEFAULT_COOLDOWN_MS,
  DEFAULT_DEBOUNCE_HITS,
  DEFAULT_THRESHOLD,
} from "./constants.js";
import type { DetectionEvent, WakeWordDetectorOptions } from "./types.js";
import { FRAME_SAMPLES, StreamingPipeline } from "./pipeline.js";
import type { InferenceRuntime } from "./types.js";

export class WakeWordDetectorController {
  private pipeline: StreamingPipeline | null = null;
  private runtime: InferenceRuntime;
  private opts: Required<Pick<WakeWordDetectorOptions, "threshold" | "debounceHits" | "cooldownMs">> &
    WakeWordDetectorOptions;
  private consecutiveHits = 0;
  private cooldownUntil = 0;

  constructor(runtime: InferenceRuntime, options: WakeWordDetectorOptions) {
    this.runtime = runtime;
    this.opts = {
      threshold: DEFAULT_THRESHOLD,
      debounceHits: DEFAULT_DEBOUNCE_HITS,
      cooldownMs: DEFAULT_COOLDOWN_MS,
      ...options,
    };
  }

  async load(): Promise<void> {
    const classifier = await this.runtime.createSession(this.opts.classifierPath);
    this.pipeline = new StreamingPipeline(this.opts.backboneDir, classifier);
    await this.pipeline.init((path) => this.runtime.createSession(path));
  }

  reset(): void {
    this.pipeline?.reset();
    this.consecutiveHits = 0;
    this.cooldownUntil = 0;
  }

  dispose(): void {
    this.pipeline = null;
  }

  async processFrame(frame: Float32Array): Promise<number | null> {
    if (!this.pipeline) throw new Error("Detector not loaded");
    if (frame.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${frame.length}`);
    }
    const now = Date.now();
    if (now < this.cooldownUntil) {
      return null;
    }
    const score = await this.pipeline.processFrame(frame);
    if (score === null || score < this.opts.threshold!) {
      this.consecutiveHits = 0;
      return score;
    }
    this.consecutiveHits += 1;
    if (this.consecutiveHits >= this.opts.debounceHits!) {
      this.consecutiveHits = 0;
      this.cooldownUntil = now + this.opts.cooldownMs!;
      const event: DetectionEvent = {
        score,
        timestampMs: now,
        timeToWakeMs: this.pipeline.getTimeToWakeMs(),
      };
      this.opts.onDetection?.(event);
    }
    return score;
  }
}

export function createWakeWordDetector(
  runtime: InferenceRuntime,
  options: WakeWordDetectorOptions,
): WakeWordDetectorController {
  return new WakeWordDetectorController(runtime, options);
}
