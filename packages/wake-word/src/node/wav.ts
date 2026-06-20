export interface DecodedWav {
  samples: Float32Array;
  sampleRate: number;
}

function chunkId(data: Uint8Array, offset: number): string {
  return String.fromCharCode(...data.subarray(offset, offset + 4));
}

export function decodeMonoWav(data: Uint8Array): DecodedWav {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (chunkId(data, 0) !== "RIFF" || chunkId(data, 8) !== "WAVE") {
    throw new Error("Expected a RIFF/WAVE file");
  }

  let audioFormat = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let pcmOffset = -1;
  let pcmBytes = 0;

  let offset = 12;
  while (offset + 8 <= data.byteLength) {
    const id = chunkId(data, offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      audioFormat = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bitsPerSample = view.getUint16(body + 14, true);
    } else if (id === "data") {
      pcmOffset = body;
      pcmBytes = size;
    }
    offset = body + size + (size % 2);
  }

  if (channels !== 1) throw new Error(`Expected mono WAV, got ${channels} channels`);
  if (pcmOffset < 0) throw new Error("WAV has no data chunk");

  const bytesPerSample = bitsPerSample / 8;
  const sampleCount = Math.floor(pcmBytes / bytesPerSample);
  const samples = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) {
    const position = pcmOffset + i * bytesPerSample;
    if (audioFormat === 1 && bitsPerSample === 16) {
      samples[i] = view.getInt16(position, true) / 32768;
    } else if (audioFormat === 3 && bitsPerSample === 32) {
      samples[i] = view.getFloat32(position, true);
    } else {
      throw new Error(
        `Unsupported WAV format=${audioFormat} bits=${bitsPerSample}`,
      );
    }
  }
  return { samples, sampleRate };
}
