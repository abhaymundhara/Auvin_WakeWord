import type { InferenceRuntime, InferenceSessionLike, TensorMap } from "../core/types.js";

export function createWebRuntime(): InferenceRuntime {
  return {
    async createSession(modelPath: string): Promise<InferenceSessionLike> {
      const ort = await import("onnxruntime-web");
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
      });
      return {
        async run(feeds: TensorMap): Promise<TensorMap> {
          const ortFeeds: Record<string, import("onnxruntime-web").Tensor> = {};
          for (const [name, data] of Object.entries(feeds)) {
            let dims: number[];
            if (name === "input" && data instanceof Float32Array) {
              if (data.length === 1280 || data.length === 1760) dims = [1, data.length];
              else if (data.length === 512) dims = [1, 512];
              else if (data.length === 1536) dims = [1, 16, 96];
              else dims = [data.length];
            } else if (name === "input_1") dims = [1, 76, 32, 1];
            else if (name === "h" || name === "c") dims = [2, 1, 64];
            else if (name === "sr") dims = [1];
            else dims = [data.length];
            if (data instanceof BigInt64Array) {
              ortFeeds[name] = new ort.Tensor("int64", data, dims);
            } else {
              ortFeeds[name] = new ort.Tensor("float32", data as Float32Array, dims);
            }
          }
          const result = await session.run(ortFeeds);
          const mapped: TensorMap = {};
          for (const [key, tensor] of Object.entries(result)) {
            mapped[key] = tensor.data as Float32Array;
          }
          return mapped;
        },
      };
    },
  };
}
