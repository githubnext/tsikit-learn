/**
 * Deprecation helpers — analogous to sklearn.utils.deprecation.
 * Provides utilities for marking deprecated functions, classes, and attributes.
 */

/** Options for the deprecated() decorator / wrapper. */
export interface DeprecationOptions {
  /** Replacement symbol or instruction shown in the warning. */
  alternative?: string;
  /** sklearn version in which this was deprecated. */
  since?: string;
  /** sklearn version in which this will be removed. */
  removeIn?: string;
  /** Extra detail appended to the warning message. */
  extra?: string;
}

/** Severity of the deprecation warning. */
export type DeprecationSeverity = "warn" | "error";

/** A structured deprecation warning record. */
export interface DeprecationWarning {
  symbol: string;
  message: string;
  options: DeprecationOptions;
  timestamp: number;
}

// Module-level registry of emitted warnings (de-duplicated by symbol + caller).
const _emittedWarnings = new Set<string>();
const _warningHistory: DeprecationWarning[] = [];

/**
 * Emits a deprecation warning for `symbol`, once per unique call site.
 *
 * @param symbol   Name of the deprecated symbol (function/class/attribute).
 * @param options  Additional context for the warning message.
 * @param severity If "error", throws instead of warning.
 */
export function warn(
  symbol: string,
  options: DeprecationOptions = {},
  severity: DeprecationSeverity = "warn",
): void {
  const key = `${symbol}|${options.since ?? ""}|${options.removeIn ?? ""}`;
  if (_emittedWarnings.has(key)) return;
  _emittedWarnings.add(key);

  const msg = buildMessage(symbol, options);
  const record: DeprecationWarning = {
    symbol,
    message: msg,
    options,
    timestamp: Date.now(),
  };
  _warningHistory.push(record);

  if (severity === "error") throw new Error(msg);
  if (typeof console !== "undefined")
    console.warn(`[DeprecationWarning] ${msg}`);
}

/** Builds the human-readable deprecation message for a symbol. */
export function buildMessage(
  symbol: string,
  options: DeprecationOptions = {},
): string {
  let msg = `\`${symbol}\` is deprecated`;
  if (options.since) msg += ` since version ${options.since}`;
  if (options.removeIn) msg += ` and will be removed in ${options.removeIn}`;
  msg += ".";
  if (options.alternative) msg += ` Use \`${options.alternative}\` instead.`;
  if (options.extra) msg += ` ${options.extra}`;
  return msg;
}

/** Returns a copy of the full deprecation warning history. */
export function getWarningHistory(): DeprecationWarning[] {
  return [..._warningHistory];
}

/** Clears the set of emitted warnings (useful in tests). */
export function clearWarnings(): void {
  _emittedWarnings.clear();
  _warningHistory.length = 0;
}

/**
 * Wraps a function to emit a deprecation warning on each call.
 *
 * @example
 * ```ts
 * const oldFoo = deprecated(newFoo, "oldFoo", { alternative: "newFoo", since: "1.4" });
 * ```
 */
export function deprecated<T extends (...args: unknown[]) => unknown>(
  fn: T,
  symbol: string,
  options: DeprecationOptions = {},
): T {
  return ((...args: Parameters<T>): ReturnType<T> => {
    warn(symbol, options);
    return fn(...args) as ReturnType<T>;
  }) as T;
}

/**
 * Class decorator (TypeScript 5 style) that emits a deprecation warning
 * whenever the class is instantiated.
 */
export function deprecatedClass(options: DeprecationOptions = {}) {
  // biome-ignore lint/suspicious/noExplicitAny: mixin class requires any[] per TypeScript spec
  return <T extends new (...args: any[]) => object>(
    Base: T,
    ctx?: { name?: string },
  ): T => {
    const name = ctx?.name ?? Base.name;
    return class extends Base {
      // biome-ignore lint/suspicious/noExplicitAny: mixin class requires any[] per TypeScript spec
      constructor(...args: any[]) {
        super(...args);
        warn(name, options);
      }
    } as T;
  };
}

/**
 * Wraps an object property getter to emit a deprecation warning on first access.
 *
 * @example
 * ```ts
 * class MyEstimator {
 *   get oldAttr() { return deprecatedGetter(this, "oldAttr", this.newAttr, { since: "1.4" }); }
 * }
 * ```
 */
export function deprecatedGetter<V>(
  _instance: object,
  attrName: string,
  value: V,
  options: DeprecationOptions = {},
): V {
  warn(attrName, options);
  return value;
}
