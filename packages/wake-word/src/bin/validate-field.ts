#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createNodeRuntime, createWakeWordDetector } from "../index.js";
import { FRAME_SAMPLES, SAMPLE_RATE } from "../core/constants.js";
import { decodeMonoWav } from "../node/wav.js";

const root = path.resolve(fileURLToPath(new URL("../../../..", import.meta.url)));

interface ClipResult {
  file: string;
  label: "positive" | "negative";
  detections: number;
  maxScore: number;
  scoredFrames: number;
}

function numericArg(name: string, fallback: number): number {
  const index = process.argv.indexOf(name);
  return index >= 0 ? Number(process.argv[index + 1]) : fallback;
}

function stringArg(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1]! : fallback;
}

function wavFiles(
  directory: string,
  prefix: string,
  holdoutModulo: number,
): string[] {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory)
    .filter((name) => name.startsWith(prefix) && name.endsWith(".wav"))
    .filter((name) => {
      if (holdoutModulo <= 0) return true;
      const match = name.match(/-(\d+)\.wav$/);
      return match !== null && Number(match[1]) % holdoutModulo === 0;
    })
    .sort()
    .map((name) => path.join(directory, name));
}

async function validateClip(
  file: string,
  label: "positive" | "negative",
  threshold: number,
  debounceHits: number,
): Promise<ClipResult> {
  const decoded = decodeMonoWav(fs.readFileSync(file));
  if (decoded.sampleRate !== SAMPLE_RATE) {
    throw new Error(`${file}: expected ${SAMPLE_RATE} Hz, got ${decoded.sampleRate}`);
  }

  let detections = 0;
  let maxScore = 0;
  let scoredFrames = 0;
  const detector = createWakeWordDetector(createNodeRuntime(), {
    backboneDir: path.join(root, "models/backbone"),
    classifierPath: path.join(root, "models/auvin.onnx"),
    threshold,
    debounceHits,
    cooldownMs: 1500,
    onDetection: () => { detections += 1; },
  });
  await detector.load();
  for (
    let offset = 0;
    offset + FRAME_SAMPLES <= decoded.samples.length;
    offset += FRAME_SAMPLES
  ) {
    const score = await detector.processFrame(
      decoded.samples.slice(offset, offset + FRAME_SAMPLES),
    );
    if (score !== null) {
      scoredFrames += 1;
      maxScore = Math.max(maxScore, score);
    }
  }
  detector.dispose();
  return {
    file: path.relative(root, file),
    label,
    detections,
    maxScore,
    scoredFrames,
  };
}

async function main(): Promise<void> {
  const clipsDir = path.resolve(stringArg(
    "--clips-dir",
    path.join(root, "data/validation"),
  ));
  const threshold = numericArg("--threshold", 0.6);
  const debounceHits = numericArg("--debounce-hits", 2);
  const minimumRecall = numericArg("--min-recall", 0.9);
  const maximumFpr = numericArg("--max-fpr", 0.05);
  const holdoutModulo = numericArg("--holdout-modulo", 0);

  const results: ClipResult[] = [];
  for (const file of wavFiles(
    path.join(clipsDir, "positive"),
    "auvin-",
    holdoutModulo,
  )) {
    results.push(await validateClip(file, "positive", threshold, debounceHits));
  }
  for (const file of wavFiles(
    path.join(clipsDir, "negative"),
    "bg-",
    holdoutModulo,
  )) {
    results.push(await validateClip(file, "negative", threshold, debounceHits));
  }
  const positives = results.filter((result) => result.label === "positive");
  const negatives = results.filter((result) => result.label === "negative");
  if (!positives.length || !negatives.length) {
    throw new Error(`Expected positive and negative WAVs under ${clipsDir}`);
  }

  const recall = positives.filter((result) => result.detections > 0).length
    / positives.length;
  const falsePositiveRate = negatives
    .filter((result) => result.detections > 0).length / negatives.length;
  const passed = recall >= minimumRecall && falsePositiveRate <= maximumFpr;
  console.log(JSON.stringify({
    clips: results.length,
    threshold,
    debounceHits,
    holdoutModulo,
    recall,
    falsePositiveRate,
    gates: { minimumRecall, maximumFpr },
    passed,
    results,
  }, null, 2));
  if (!passed) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
