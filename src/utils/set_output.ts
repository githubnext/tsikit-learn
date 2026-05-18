/**
 * set_output API — ported from sklearn.utils._set_output
 * Controls the output container type for transformers.
 */

export type OutputType = "default" | "pandas" | "polars";

let _globalOutputType: OutputType = "default";

export interface SetOutputConfig {
  transform?: OutputType;
}

/**
 * Set global default output type for all transformers.
 * This mirrors sklearn's set_config(transform_output=...).
 */
export function setConfig(config: SetOutputConfig): void {
  if (config.transform !== undefined) {
    _globalOutputType = config.transform;
  }
}

/**
 * Get the current global configuration.
 */
export function getConfig(): Required<SetOutputConfig> {
  return { transform: _globalOutputType };
}

/**
 * Context manager-like utility for temporarily changing configuration.
 * Returns an object with a done() method to restore the previous config.
 */
export function configContext(config: SetOutputConfig): { done: () => void } {
  const previous = getConfig();
  setConfig(config);
  return {
    done() {
      setConfig(previous);
    },
  };
}

/**
 * Mixin that adds set_output support to a transformer class.
 * Call augmentWithSetOutput(instance) to add set_output to any transformer.
 */
export interface SetOutputMixin {
  setOutput(config: SetOutputConfig): this;
  getOutputType(): OutputType;
}

/** The output type set on a specific transformer instance */
const instanceOutputTypes = new WeakMap<object, OutputType>();

/**
 * Add set_output capability to a transformer instance.
 */
export function augmentWithSetOutput<T extends object>(instance: T): T & SetOutputMixin {
  const augmented = instance as T & SetOutputMixin;
  augmented.setOutput = function (config: SetOutputConfig) {
    if (config.transform !== undefined) {
      instanceOutputTypes.set(this as object, config.transform);
    }
    return this;
  };
  augmented.getOutputType = function () {
    return instanceOutputTypes.get(this as object) ?? _globalOutputType;
  };
  return augmented;
}

/**
 * Wraps a 2D Float64Array output into the appropriate container
 * based on the transformer's configured output type.
 *
 * Currently supports only "default" (returns Float64Array[]).
 * "pandas" and "polars" return the same structure (future: could wrap in tsb DataFrame).
 */
export function wrapOutput(
  data: Float64Array[],
  _outputType: OutputType,
  _featureNamesOut?: string[],
): Float64Array[] {
  // In a browser/Bun environment without pandas/polars bindings,
  // always return raw arrays regardless of the output type setting.
  return data;
}

/**
 * A simple record for storing named configuration entries,
 * used by Memory and pipeline caching mechanisms.
 */
export interface ConfigEntry {
  key: string;
  value: unknown;
}

/**
 * Global configuration store (analogous to sklearn's _global_config dict).
 */
export class GlobalConfig {
  private readonly store = new Map<string, unknown>();

  set(key: string, value: unknown): void {
    this.store.set(key, value);
  }

  get<T>(key: string, defaultValue: T): T {
    return this.store.has(key) ? (this.store.get(key) as T) : defaultValue;
  }

  getAll(): Record<string, unknown> {
    return Object.fromEntries(this.store);
  }

  reset(): void {
    this.store.clear();
  }
}

export const globalConfig = new GlobalConfig();
