import { describe, expect, it } from "vitest";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createWakeWordDetector, createNodeRuntime } from "../src/index.js";
import { FRAME_SAMPLES, SAMPLE_RATE } from "../src/core/constants.js";
import { StreamingPipeline } from "../src/core/pipeline.js";
import type { InferenceSessionLike } from "../src/core/types.js";

const root = path.resolve(fileURLToPath(new URL("../../..", import.meta.url)));
const backboneDir = path.join(root, "models/backbone");
const classifierPath = path.join(root, "models/auvin.onnx");

function silenceFrame(): Float32Array {
  return new Float32Array(FRAME_SAMPLES);
}

describe("wake-word pipeline", () => {
  it("loads backbone models and scores silence low", async () => {
    const runtime = createNodeRuntime();
    const classifier = await runtime.createSession(classifierPath);
    const pipeline = new StreamingPipeline(backboneDir, classifier);
    await pipeline.init((p) => runtime.createSession(p));

    for (let i = 0; i < 40; i += 1) {
      await pipeline.processFrame(silenceFrame());
    }
    const score = await pipeline.processFrame(silenceFrame());
    expect(score === null || score < 0.6).toBe(true);
  }, 120000);

  it("runs per-frame inference under 20ms after warmup", async () => {
    const runtime = createNodeRuntime();
    const classifier = await runtime.createSession(classifierPath);
    const pipeline = new StreamingPipeline(backboneDir, classifier);
    await pipeline.init((p) => runtime.createSession(p));
    for (let i = 0; i < 35; i += 1) {
      await pipeline.processFrame(silenceFrame());
    }
    const start = performance.now();
    await pipeline.processFrame(silenceFrame());
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(20);
  }, 120000);

  it("warms feature history during low-VAD leading audio", async () => {
    let speech = false;
    let classifierRuns = 0;
    const classifier: InferenceSessionLike = {
      async run() {
        classifierRuns += 1;
        return { output: Float32Array.from([0.9]) };
      },
    };
    const pipeline = new StreamingPipeline("/models", classifier);
    await pipeline.init(async (modelPath) => ({
      async run() {
        if (modelPath.endsWith("silero_vad.onnx")) {
          return {
            output: Float32Array.from([speech ? 1 : 0]),
            hn: new Float32Array(128),
            cn: new Float32Array(128),
          };
        }
        if (modelPath.endsWith("melspectrogram.onnx")) {
          return { output: new Float32Array(8 * 32) };
        }
        return { output: new Float32Array(96) };
      },
    }));

    for (let i = 0; i < 16; i += 1) {
      expect(await pipeline.processFrame(silenceFrame())).toBeNull();
    }
    expect(classifierRuns).toBe(0);

    speech = true;
    expect(await pipeline.processFrame(silenceFrame())).toBeCloseTo(0.9);
    expect(classifierRuns).toBe(1);
  });

  it("debounces detections", async () => {
    let detections = 0;
    const runtime = createNodeRuntime();
    const detector = createWakeWordDetector(runtime, {
      backboneDir,
      classifierPath,
      threshold: 0.0,
      debounceHits: 2,
      cooldownMs: 5000,
      onDetection: () => {
        detections += 1;
      },
    });
    await detector.load();
    for (let i = 0; i < 50; i += 1) {
      await detector.processFrame(silenceFrame());
    }
    expect(detections).toBeLessThanOrEqual(1);
  }, 120000);
});

describe("constants", () => {
  it("uses 16kHz framing", () => {
    expect(SAMPLE_RATE).toBe(16000);
    expect(FRAME_SAMPLES).toBe(1280);
  });
});
