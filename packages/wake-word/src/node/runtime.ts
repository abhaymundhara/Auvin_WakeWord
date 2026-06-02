import type { InferenceRuntime, InferenceSessionLike, TensorMap } from "../core/types.js";
import { Tensor } from "onnxruntime-node";

function wrapSession(session: import("onnxruntime-node").InferenceSession): InferenceSessionLike {
  return {
    async run(feeds: TensorMap): Promise<TensorMap> {
      const ortFeeds: Record<string, import("onnxruntime-node").Tensor> = {};
      for (const [name, data] of Object.entries(feeds)) {
        let dims: number[];
        if (name === "input" && data instanceof Float32Array) {
          if (data.length === 1280 || data.length === 1760) {
            dims = [1, data.length];
          } else if (data.length === 512) {
            dims = [1, 512];
          } else if (data.length === 1536) {
            dims = [1, 16, 96];
          } else {
            dims = [data.length];
          }
          ortFeeds[name] = new Tensor("float32", data, dims);
        } else if (name === "input_1" && data instanceof Float32Array) {
          ortFeeds[name] = new Tensor("float32", data, [1, 76, 32, 1]);
        } else if ((name === "h" || name === "c") && data instanceof Float32Array) {
          ortFeeds[name] = new Tensor("float32", data, [2, 1, 64]);
        } else if (name === "sr" && data instanceof BigInt64Array) {
          ortFeeds[name] = new Tensor("int64", data, [1]);
        } else if (data instanceof Float32Array) {
          ortFeeds[name] = new Tensor("float32", data, [data.length]);
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
}

export function createNodeRuntime(): InferenceRuntime {
  return {
    async createSession(modelPath: string): Promise<InferenceSessionLike> {
      const ort = await import("onnxruntime-node");
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["cpu"],
      });
      return wrapSession(session);
    },
  };
}
