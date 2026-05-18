/**
 * Model persistence utilities (pickle-like serialization).
 * Mirrors sklearn.utils._joblib and joblib.dump/load patterns.
 *
 * Provides JSON-based model serialization for tsikit-learn estimators.
 */

import { BaseEstimator } from "../base.js";

export interface SerializedModel {
  /** Class name for reconstruction */
  className: string;
  /** Module path for reconstruction */
  modulePath?: string;
  /** All parameter values from get_params() */
  params: Record<string, unknown>;
  /** Fitted attributes (coef_, intercept_, etc.) */
  fittedAttributes: Record<string, unknown>;
  /** Schema version */
  version: number;
}

/** Check if a value is a typed array */
function isTypedArray(
  v: unknown,
): v is Float64Array | Float32Array | Int32Array | Uint8Array {
  return (
    v instanceof Float64Array ||
    v instanceof Float32Array ||
    v instanceof Int32Array ||
    v instanceof Uint8Array
  );
}

/** Serialize a value to a JSON-safe representation */
function serializeValue(v: unknown): unknown {
  if (v === null || v === undefined) return v;
  if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") return v;
  if (isTypedArray(v)) {
    return {
      __typedArray: true,
      type: v.constructor.name,
      data: Array.from(v),
    };
  }
  if (Array.isArray(v)) {
    return v.map(serializeValue);
  }
  if (v instanceof BaseEstimator) {
    return dumpEstimator(v);
  }
  if (typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      out[k] = serializeValue(val);
    }
    return out;
  }
  return v;
}

/** Deserialize a value from JSON-safe representation */
function deserializeValue(v: unknown): unknown {
  if (v === null || v === undefined) return v;
  if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") return v;
  if (Array.isArray(v)) {
    return v.map(deserializeValue);
  }
  if (typeof v === "object") {
    const obj = v as Record<string, unknown>;
    if (obj["__typedArray"] === true) {
      const type = obj["type"] as string;
      const data = obj["data"] as number[];
      switch (type) {
        case "Float64Array": return new Float64Array(data);
        case "Float32Array": return new Float32Array(data);
        case "Int32Array": return new Int32Array(data);
        case "Uint8Array": return new Uint8Array(data);
        default: return new Float64Array(data);
      }
    }
    const out: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(obj)) {
      out[k] = deserializeValue(val);
    }
    return out;
  }
  return v;
}

/**
 * Serialize a fitted estimator to a JSON-compatible object.
 * Call this on a fitted estimator to capture its state.
 */
export function dumpEstimator(estimator: BaseEstimator): SerializedModel {
  const params = estimator.get_params(false);
  const fittedAttributes: Record<string, unknown> = {};

  // Collect fitted attributes (those ending with _)
  for (const key of Object.keys(estimator as unknown as Record<string, unknown>)) {
    if (key.endsWith("_") && !key.endsWith("__")) {
      fittedAttributes[key] = serializeValue(
        (estimator as unknown as Record<string, unknown>)[key],
      );
    }
  }

  const serializedParams: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    serializedParams[k] = serializeValue(v);
  }

  return {
    className: estimator.constructor.name,
    params: serializedParams,
    fittedAttributes,
    version: 1,
  };
}

/**
 * Serialize a fitted estimator to a JSON string.
 */
export function dumpJSON(estimator: BaseEstimator): string {
  return JSON.stringify(dumpEstimator(estimator), null, 2);
}

/**
 * Load an estimator from a serialized model object.
 * The caller must provide the estimator class (constructor) to instantiate.
 */
export function loadEstimator<T extends BaseEstimator>(
  Constructor: new (params?: Record<string, unknown>) => T,
  serialized: SerializedModel,
): T {
  const deserializedParams: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(serialized.params)) {
    deserializedParams[k] = deserializeValue(v);
  }

  const estimator = new Constructor(deserializedParams);

  // Restore fitted attributes
  for (const [key, val] of Object.entries(serialized.fittedAttributes)) {
    (estimator as unknown as Record<string, unknown>)[key] = deserializeValue(val);
  }

  return estimator;
}

/**
 * Load an estimator from a JSON string.
 * The caller must provide the estimator class (constructor) to instantiate.
 */
export function loadJSON<T extends BaseEstimator>(
  Constructor: new (params?: Record<string, unknown>) => T,
  json: string,
): T {
  const serialized = JSON.parse(json) as SerializedModel;
  return loadEstimator(Constructor, serialized);
}

/**
 * Memory cache for estimator results (memoization).
 * Mirrors joblib.Memory for caching expensive computations.
 */
export class Memory {
  private cache = new Map<string, unknown>();
  private location: string;
  private verbose: number;

  constructor(options: { location?: string; verbose?: number } = {}) {
    this.location = options.location ?? "cache";
    this.verbose = options.verbose ?? 0;
  }

  /**
   * Cache a function's results. Returns a wrapped version of the function
   * that caches by serialized arguments.
   */
  cache_fn<TArgs extends unknown[], TResult>(
    fn: (...args: TArgs) => TResult,
    _options: { ignore?: string[] } = {},
  ): (...args: TArgs) => TResult {
    const self = this;
    return function (...args: TArgs): TResult {
      const key = JSON.stringify(args, (_k, v: unknown) => {
        if (isTypedArray(v)) return Array.from(v);
        return v;
      });
      if (self.cache.has(key)) {
        if (self.verbose > 0) console.log(`[Memory] Cache hit for ${fn.name}`);
        return self.cache.get(key) as TResult;
      }
      const result = fn(...args);
      self.cache.set(key, result);
      return result;
    };
  }

  /** Clear the cache */
  clear(): void {
    this.cache.clear();
  }

  /** Number of cached items */
  get size(): number {
    return this.cache.size;
  }

  toString(): string {
    return `Memory(location=${this.location}, items=${this.cache.size})`;
  }
}

/**
 * Parallel computation utilities compatible with joblib.Parallel interface.
 * In this TypeScript implementation, these run sequentially in the main thread.
 */
export interface DelayedResult<T> {
  fn: () => T;
}

/**
 * Wrap a function call for deferred execution (alias for joblib.delayed).
 * Named `deferTask` to avoid conflict with utils/parallel.delayed.
 */
export function deferTask<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => TResult,
): (...args: TArgs) => DelayedResult<TResult> {
  return (...args: TArgs) => ({
    fn: () => fn(...args),
  });
}

/**
 * Run deferred tasks in sequence (simulating joblib.Parallel).
 * In a browser/Node environment, true parallelism requires Workers.
 */
export function runTasks<T>(
  tasks: DelayedResult<T>[],
  _options: { nJobs?: number; verbose?: number } = {},
): T[] {
  return tasks.map(t => t.fn());
}
