import { describe, expect, it } from "vitest";
import {
  createNativeRuntime,
  type NativeOrtModule,
} from "../src/native/runtime.js";

describe("React Native runtime", () => {
  it("uses native ORT tensors with model-specific dimensions", async () => {
    const dimensions = new Map<string, number[]>();
    class FakeTensor {
      constructor(
        public readonly type: "float32" | "int64",
        public readonly data: Float32Array | BigInt64Array,
        public readonly dims: number[],
      ) {}
    }
    const ort: NativeOrtModule = {
      Tensor: FakeTensor,
      InferenceSession: {
        async create() {
          return {
            async run(feeds) {
              for (const [name, tensor] of Object.entries(feeds)) {
                dimensions.set(name, (tensor as FakeTensor).dims);
              }
              return { output: { data: Float32Array.from([0.75]) } };
            },
          };
        },
      },
    };

    const session = await createNativeRuntime(ort).createSession("model.onnx");
    const output = await session.run({
      input: new Float32Array(1536),
      h: new Float32Array(128),
      sr: BigInt64Array.from([16000n]),
    });

    expect(dimensions.get("input")).toEqual([1, 16, 96]);
    expect(dimensions.get("h")).toEqual([2, 1, 64]);
    expect(dimensions.get("sr")).toEqual([1]);
    expect(output.output?.[0]).toBeCloseTo(0.75);
  });
});
