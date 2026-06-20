import type { InferenceSessionLike } from "./types.js";
import {
  CLASSIFIER_FRAMES,
  EMBEDDING_DIM,
  FRAME_SAMPLES,
  MEL_BUFFER_FRAMES,
  MEL_CALIBRATION_OFFSET,
  MEL_CALIBRATION_SCALE,
  MEL_CONTEXT_SAMPLES,
  MEL_FRAMES_PER_STEP,
  SAMPLE_RATE,
  VAD_C_DIM,
  VAD_FRAME_SAMPLES,
  VAD_H_DIM,
} from "./constants.js";

function calibrateMel(spec: Float32Array): Float32Array {
  const out = new Float32Array(spec.length);
  for (let i = 0; i < spec.length; i += 1) {
    out[i] = spec[i]! / MEL_CALIBRATION_SCALE + MEL_CALIBRATION_OFFSET;
  }
  return out;
}

export class StreamingPipeline {
  private melSession: InferenceSessionLike | null = null;
  private embSession: InferenceSessionLike | null = null;
  private vadSession: InferenceSessionLike | null = null;

  private pcmBuffer: number[] = [];
  private melBuffer: Float32Array[] = this.createInitialMelBuffer();
  private embeddingBuffer: Float32Array[] = [];
  private vadH = new Float32Array(2 * 1 * VAD_H_DIM);
  private vadC = new Float32Array(2 * 1 * VAD_C_DIM);
  private frameCount = 0;
  private speechStartedAt: number | null = null;

  constructor(
    private readonly backboneDir: string,
    private readonly classifierSession: InferenceSessionLike,
  ) {}

  private createInitialMelBuffer(): Float32Array[] {
    return Array.from(
      { length: MEL_BUFFER_FRAMES },
      () => new Float32Array(32).fill(1),
    );
  }

  async init(
    createSession: (path: string) => Promise<InferenceSessionLike>,
  ): Promise<void> {
    this.melSession = await createSession(`${this.backboneDir}/melspectrogram.onnx`);
    this.embSession = await createSession(`${this.backboneDir}/embedding_model.onnx`);
    this.vadSession = await createSession(`${this.backboneDir}/silero_vad.onnx`);
  }

  reset(): void {
    this.pcmBuffer = [];
    this.melBuffer = this.createInitialMelBuffer();
    this.embeddingBuffer = [];
    this.vadH.fill(0);
    this.vadC.fill(0);
    this.frameCount = 0;
    this.speechStartedAt = null;
  }

  private async runVad(frame: Float32Array): Promise<number> {
    if (!this.vadSession) return 1;
    const chunk = frame.length >= VAD_FRAME_SAMPLES
      ? frame.subarray(0, VAD_FRAME_SAMPLES)
      : frame;
    const feeds: Record<string, Float32Array | BigInt64Array> = {
      input: chunk,
      h: this.vadH,
      c: this.vadC,
      sr: BigInt64Array.from([BigInt(SAMPLE_RATE)]),
    };
    const out = await this.vadSession.run(feeds);
    const prob = out.output ?? out.out ?? Object.values(out)[0];
    if (out.hn) this.vadH = new Float32Array(out.hn as Float32Array);
    if (out.cn) this.vadC = new Float32Array(out.cn as Float32Array);
    if (out.h) this.vadH = new Float32Array(out.h as Float32Array);
    if (out.c) this.vadC = new Float32Array(out.c as Float32Array);
    return prob ? Number(prob[0]) : 1;
  }

  private async runMel(context: Float32Array): Promise<Float32Array> {
    if (!this.melSession) throw new Error("Mel session not loaded");
    const out = await this.melSession.run({ input: context });
    const raw = (out.output ?? Object.values(out)[0]) as Float32Array;
    return calibrateMel(raw);
  }

  private async runEmbedding(melWindow: Float32Array[]): Promise<Float32Array> {
    if (!this.embSession) throw new Error("Embedding session not loaded");
    const input4d = new Float32Array(MEL_BUFFER_FRAMES * 32);
    melWindow.forEach((row, idx) => input4d.set(row, idx * 32));
    const out = await this.embSession.run({ input_1: input4d });
    const emb = (out.output ?? out.output_1 ?? Object.values(out)[0]) as Float32Array;
    return emb.slice(0, EMBEDDING_DIM);
  }

  private pushMel(mel: Float32Array): void {
    const frames = mel.length / 32;
    for (let i = 0; i < frames; i += 1) {
      this.melBuffer.push(mel.subarray(i * 32, (i + 1) * 32));
    }
    if (this.melBuffer.length > 970) {
      this.melBuffer = this.melBuffer.slice(-970);
    }
  }

  private async processBufferedFrame(): Promise<void> {
    const needed = FRAME_SAMPLES + MEL_CONTEXT_SAMPLES;
    let context: Float32Array;
    if (this.pcmBuffer.length < needed) {
      const pad = new Float32Array(needed - this.pcmBuffer.length);
      context = new Float32Array(needed);
      context.set(pad, 0);
      context.set(Float32Array.from(this.pcmBuffer), pad.length);
    } else {
      context = Float32Array.from(this.pcmBuffer.slice(-needed));
    }
    const mel = await this.runMel(context);
    if (mel.length / 32 !== MEL_FRAMES_PER_STEP) {
      throw new Error(`Expected ${MEL_FRAMES_PER_STEP} mel frames, got ${mel.length / 32}`);
    }
    this.pushMel(mel);
    if (this.melBuffer.length >= MEL_BUFFER_FRAMES) {
      const window = this.melBuffer.slice(-MEL_BUFFER_FRAMES);
      const emb = await this.runEmbedding(window);
      this.embeddingBuffer.push(emb);
      if (this.embeddingBuffer.length > 120) {
        this.embeddingBuffer = this.embeddingBuffer.slice(-120);
      }
    }
  }

  async processFrame(frame: Float32Array): Promise<number | null> {
    const speechProb = await this.runVad(frame);
    if (this.speechStartedAt === null) {
      if (speechProb >= 0.1) {
        this.speechStartedAt = Date.now();
      }
    }
    this.pcmBuffer.push(...frame);
    if (this.pcmBuffer.length > SAMPLE_RATE * 10) {
      this.pcmBuffer = this.pcmBuffer.slice(-SAMPLE_RATE * 10);
    }
    await this.processBufferedFrame();
    this.frameCount += 1;
    if (speechProb < 0.1) {
      return null;
    }
    if (this.embeddingBuffer.length < CLASSIFIER_FRAMES) {
      return null;
    }
    const seq = this.embeddingBuffer.slice(-CLASSIFIER_FRAMES);
    const input = new Float32Array(CLASSIFIER_FRAMES * EMBEDDING_DIM);
    seq.forEach((emb, idx) => input.set(emb, idx * EMBEDDING_DIM));
    const out = await this.classifierSession.run({ input });
    const scoreTensor = (out.output ?? Object.values(out)[0]) as Float32Array;
    return scoreTensor[0] ?? null;
  }

  getTimeToWakeMs(): number {
    if (this.speechStartedAt === null) return 0;
    return Date.now() - this.speechStartedAt;
  }
}

export { FRAME_SAMPLES, SAMPLE_RATE };
