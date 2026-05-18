/**
 * Compatibility and distribution utilities — ported from sklearn.utils.fixes
 * and sklearn.utils._param_validation
 */

/**
 * A loguniform distribution (log-uniform random variable) for hyperparameter
 * search, similar to scipy.stats.loguniform.
 */
export class loguniform {
  private readonly logLow: number;
  private readonly logHigh: number;

  constructor(
    private readonly low: number,
    private readonly high: number,
  ) {
    if (low <= 0 || high <= 0) {
      throw new RangeError("loguniform bounds must be positive");
    }
    if (low >= high) {
      throw new RangeError("low must be less than high");
    }
    this.logLow = Math.log(low);
    this.logHigh = Math.log(high);
  }

  /** Draw a single sample from the log-uniform distribution */
  rvs(randomState?: number): number {
    // Simple LCG if randomState provided, else Math.random
    let u: number;
    if (randomState !== undefined) {
      // LCG with modulus 2^31 - 1
      const a = 1664525;
      const c = 1013904223;
      const m = 2 ** 31;
      u = ((a * randomState + c) % m) / m;
    } else {
      u = Math.random();
    }
    return Math.exp(this.logLow + u * (this.logHigh - this.logLow));
  }

  /** Draw n samples */
  sample(n: number, randomState?: number): Float64Array {
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = this.rvs(randomState !== undefined ? randomState + i : undefined);
    }
    return result;
  }
}

/**
 * A uniform integer distribution for hyperparameter search.
 * Samples integers uniformly from [low, high).
 */
export class randint {
  constructor(
    private readonly low: number,
    private readonly high: number,
  ) {
    if (!Number.isInteger(low) || !Number.isInteger(high)) {
      throw new TypeError("randint bounds must be integers");
    }
    if (low >= high) {
      throw new RangeError("low must be less than high");
    }
  }

  /** Draw a single integer sample */
  rvs(randomState?: number): number {
    let u: number;
    if (randomState !== undefined) {
      const a = 1664525;
      const c = 1013904223;
      const m = 2 ** 31;
      u = ((a * randomState + c) % m) / m;
    } else {
      u = Math.random();
    }
    return this.low + Math.floor(u * (this.high - this.low));
  }

  /** Draw n integer samples */
  sample(n: number, randomState?: number): Int32Array {
    const result = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = this.rvs(randomState !== undefined ? randomState + i : undefined);
    }
    return result;
  }
}

/** Version tuple for comparing sklearn-style version strings */
export interface VersionTuple {
  major: number;
  minor: number;
  patch: number;
}

/** Parse a semantic version string into a VersionTuple */
export function parseVersion(version: string): VersionTuple {
  const parts = version.split(".").map(Number);
  return {
    major: parts[0] ?? 0,
    minor: parts[1] ?? 0,
    patch: parts[2] ?? 0,
  };
}

/** Compare two version tuples: returns negative, 0, or positive */
export function compareVersions(a: VersionTuple, b: VersionTuple): number {
  if (a.major !== b.major) return a.major - b.major;
  if (a.minor !== b.minor) return a.minor - b.minor;
  return a.patch - b.patch;
}

/** Check if version a is at least version b */
export function versionAtLeast(a: VersionTuple, b: VersionTuple): boolean {
  return compareVersions(a, b) >= 0;
}

/** Current tsikit-learn version */
export const TSIKIT_LEARN_VERSION: VersionTuple = { major: 0, minor: 1, patch: 0 };

/**
 * Threadpoolctl-like context for controlling parallel workers.
 * In browser/Bun environments, threading is limited so this is a no-op.
 */
export function threadpoolLimits(n: number): { restore: () => void } {
  void n;
  return { restore: () => undefined };
}

/**
 * Whether we are running in a 32-bit environment.
 * TypeScript/JS always uses 64-bit floats, so this is always false.
 */
export const IS_32BIT = false;
