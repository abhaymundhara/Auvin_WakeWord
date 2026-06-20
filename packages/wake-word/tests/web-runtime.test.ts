import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createWebRuntime } from "../src/web/runtime.js";

const root = path.resolve(fileURLToPath(new URL("../../..", import.meta.url)));

describe("web runtime", () => {
  it("loads and runs the exported classifier with WASM", async () => {
    const runtime = createWebRuntime();
    const session = await runtime.createSession(path.join(root, "models/auvin.onnx"));
    const output = await session.run({ input: new Float32Array(16 * 96) });

    expect(output.output).toBeInstanceOf(Float32Array);
    expect(output.output?.[0]).toBeGreaterThanOrEqual(0);
    expect(output.output?.[0]).toBeLessThanOrEqual(1);
  }, 120000);
});
