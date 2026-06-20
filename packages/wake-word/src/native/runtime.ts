import type {
  InferenceRuntime,
  InferenceSessionLike,
  TensorMap,
} from "../core/types.js";

interface NativeTensorLike {
  data: Float32Array | BigInt64Array;
}

interface NativeSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, NativeTensorLike>>;
}

export interface NativeOrtModule {
  Tensor: new (
    type: "float32" | "int64",
    data: Float32Array | BigInt64Array,
    dims: number[],
  ) => unknown;
  InferenceSession: {
    create(modelPath: string): Promise<NativeSessionLike>;
  };
}

function tensorDimensions(name: string, data: Float32Array | BigInt64Array): number[] {
  if (name === "input" && data instanceof Float32Array) {
    if (data.length === 1280 || data.length === 1760) return [1, data.length];
    if (data.length === 512) return [1, 512];
    if (data.length === 1536) return [1, 16, 96];
  }
  if (name === "input_1") return [1, 76, 32, 1];
  if (name === "h" || name === "c") return [2, 1, 64];
  if (name === "sr") return [1];
  return [data.length];
}

async function loadNativeOrt(): Promise<NativeOrtModule> {
  return import("onnxruntime-react-native") as Promise<NativeOrtModule>;
}

export function createNativeRuntime(
  injectedOrt?: NativeOrtModule,
): InferenceRuntime {
  return {
    async createSession(modelPath: string): Promise<InferenceSessionLike> {
      const ort = injectedOrt ?? await loadNativeOrt();
      const session = await ort.InferenceSession.create(modelPath);
      return {
        async run(feeds: TensorMap): Promise<TensorMap> {
          const nativeFeeds: Record<string, unknown> = {};
          for (const [name, data] of Object.entries(feeds)) {
            const type = data instanceof BigInt64Array ? "int64" : "float32";
            nativeFeeds[name] = new ort.Tensor(
              type,
              data,
              tensorDimensions(name, data),
            );
          }
          const result = await session.run(nativeFeeds);
          const mapped: TensorMap = {};
          for (const [name, tensor] of Object.entries(result)) {
            mapped[name] = tensor.data;
          }
          return mapped;
        },
      };
    },
  };
}
