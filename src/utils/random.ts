/**
 * Random state utilities.
 * Mirrors sklearn.utils.check_random_state and related helpers.
 */

/** A minimal seeded pseudo-random number generator (Mulberry32). */
export class RandomState {
  private seed: number;

  constructor(seed = 0) {
    this.seed = seed >>> 0;
  }

  /** Uniform float in [0, 1). */
  random(): number {
    this.seed += 0x6d2b79f5;
    let s = this.seed;
    s = Math.imul(s ^ (s >>> 15), s | 1);
    s ^= s + Math.imul(s ^ (s >>> 7), s | 61);
    return ((s ^ (s >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform integer in [low, high). */
  randint(low: number, high: number): number {
    return low + Math.floor(this.random() * (high - low));
  }

  /** Sample `size` uniform floats in [low, high). */
  uniform(low: number, high: number, size: number): Float64Array {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) out[i]! = low + this.random() * (high - low);
    return out;
  }

  /** Sample from a standard normal distribution (Box-Muller). */
  randn(size: number): Float64Array {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.max(this.random(), 1e-14);
      const u2 = this.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      out[i]! = r * Math.cos(theta);
      if (i + 1 < size) out[i + 1]! = r * Math.sin(theta);
    }
    return out;
  }

  /** Shuffle an array in-place (Fisher-Yates). */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randint(0, i + 1);
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
    return arr;
  }

  /** Shuffle a typed array in-place (Fisher-Yates). */
  shuffleTyped(arr: Float64Array | Int32Array): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randint(0, i + 1);
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
  }

  /** Draw `nSamples` indices in [0, nTotal) with replacement. */
  choice(nTotal: number, nSamples: number, replace = true): Int32Array {
    if (replace) {
      return Int32Array.from({ length: nSamples }, () =>
        this.randint(0, nTotal),
      );
    }
    // Without replacement: partial Fisher-Yates
    const pool = Int32Array.from({ length: nTotal }, (_, i) => i);
    for (let i = 0; i < nSamples; i++) {
      const j = i + this.randint(0, nTotal - i);
      const tmp = pool[i]!;
      pool[i] = pool[j]!;
      pool[j] = tmp;
    }
    return pool.slice(0, nSamples);
  }
}

/**
 * Coerce a seed value into a `RandomState`.
 *
 * - `null` / `undefined` → uses `Math.random()` seed (non-reproducible).
 * - `number` → reproducible `RandomState(seed)`.
 * - `RandomState` → returned as-is.
 */
export function checkRandomState(
  seed?: number | RandomState | null,
): RandomState {
  if (seed == null) return new RandomState(Math.floor(Math.random() * 2 ** 31));
  if (typeof seed === "number") return new RandomState(seed);
  return seed;
}

/**
 * Resample arrays with or without replacement.
 *
 * @param arrays - One or more arrays (all same length) to resample in tandem.
 * @param nSamples - Number of samples to draw (defaults to len of first array).
 * @param replace - Whether to sample with replacement (default: true).
 * @param randomState - Seed or RandomState.
 * @returns Resampled arrays in the same order.
 */
export function resampleArrays(
  arrays: (Float64Array | Int32Array | unknown[])[],
  nSamples?: number,
  replace = true,
  randomState?: number | RandomState | null,
): (Float64Array | Int32Array | unknown[])[] {
  const rng = checkRandomState(randomState);
  const n = arrays[0]?.length ?? 0;
  const k = nSamples ?? n;
  const idx = rng.choice(n, k, replace);

  return arrays.map((arr) => {
    if (arr instanceof Float64Array) {
      return Float64Array.from(idx, (i) => arr[i]! ?? 0);
    }
    if (arr instanceof Int32Array) {
      return Int32Array.from(idx, (i) => arr[i]! ?? 0);
    }
    return Array.from(idx, (i) => (arr as unknown[])[i]);
  });
}

/**
 * Compute a stable hash of a string (djb2 variant). Useful for hashing
 * feature names or other string identifiers to a numeric bucket.
 */
export function hashString(s: string, nBuckets = 2 ** 20): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(h, 33) ^ s.charCodeAt(i)) >>> 0;
  }
  return h % nBuckets;
}
