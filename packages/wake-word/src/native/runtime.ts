import type { InferenceRuntime } from "../core/types.js";
import { createWebRuntime } from "../web/runtime.js";

/** React Native adapter — uses web ORT until native bindings are wired. */
export function createNativeRuntime(): InferenceRuntime {
  return createWebRuntime();
}
