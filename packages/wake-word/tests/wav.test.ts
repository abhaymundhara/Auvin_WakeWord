import { describe, expect, it } from "vitest";
import { decodeMonoWav } from "../src/node/wav.js";

function pcm16Wav(samples: number[], sampleRate = 16000): Uint8Array {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    [...value].forEach((character, index) => {
      view.setUint8(offset + index, character.charCodeAt(0));
    });
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, dataBytes, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, sample, true));
  return new Uint8Array(buffer);
}

describe("WAV decoder", () => {
  it("decodes mono PCM16 samples", () => {
    const decoded = decodeMonoWav(pcm16Wav([-32768, 0, 16384]));
    expect(decoded.sampleRate).toBe(16000);
    expect([...decoded.samples]).toEqual([-1, 0, 0.5]);
  });
});
