#!/usr/bin/env node
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createNodeRuntime, createWakeWordDetector } from "../index.js";
import { FRAME_SAMPLES } from "../core/constants.js";

const root = path.resolve(fileURLToPath(new URL("../../../..", import.meta.url)));
const backboneDir = path.join(root, "models/backbone");
const classifierPath = path.join(root, "models/auvin.onnx");

async function main() {
  const runtime = createNodeRuntime();
  let detections = 0;
  const detector = createWakeWordDetector(runtime, {
    backboneDir,
    classifierPath,
    onDetection: (event) => {
      detections += 1;
      console.log(`DETECTED score=${event.score.toFixed(3)} ttw=${event.timeToWakeMs}ms`);
    },
  });
  await detector.load();
  console.log("Loaded. Processing silence frames (use validate_clips.py for real audio)...");

  for (let i = 0; i < 50; i += 1) {
    await detector.processFrame(new Float32Array(FRAME_SAMPLES));
  }
  console.log(`Done. detections=${detections}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
