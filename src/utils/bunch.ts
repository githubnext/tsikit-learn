/**
 * Bunch: a simple container for datasets (like sklearn.utils.Bunch).
 * Also: check_array, column_or_1d and other utility functions.
 */

export interface BunchData {
  [key: string]: unknown;
}

/**
 * Container object exposing keys as attributes.
 * Mirrors sklearn.utils.Bunch.
 */
export class Bunch {
  [key: string]: unknown;

  constructor(data: BunchData) {
    for (const [k, v] of Object.entries(data)) {
      this[k] = v;
    }
  }

  keys(): string[] {
    return Object.keys(this).filter((k) => typeof this[k] !== "function");
  }

  values(): unknown[] {
    return this.keys().map((k) => this[k]);
  }

  entries(): Array<[string, unknown]> {
    return this.keys().map((k) => [k, this[k]] as [string, unknown]);
  }
}

/**
 * Check that X is a 2D array of Float64Arrays.
 * Throws if input is malformed. Mirrors sklearn.utils.check_array (simplified).
 */
export function checkArray2D(
  X: unknown,
  options: { ensureMinSamples?: number; ensureMinFeatures?: number } = {},
): Float64Array[] {
  if (!Array.isArray(X)) throw new Error("Input must be an array.");
  if (X.length === 0) return [];

  const minSamples = options.ensureMinSamples ?? 1;
  const minFeatures = options.ensureMinFeatures ?? 1;

  if (X.length < minSamples)
    throw new Error(`Input must have at least ${minSamples} samples.`);

  const p = (X[0] as Float64Array | number[]).length ?? 0;
  if (p < minFeatures)
    throw new Error(`Input must have at least ${minFeatures} features.`);

  return X.map((row, i) => {
    if (row instanceof Float64Array) return row;
    if (Array.isArray(row)) return new Float64Array(row as number[]);
    throw new Error(`Row ${i} is not a Float64Array or number array.`);
  });
}

/**
 * Raise if array has more than one non-singleton dimension.
 * Mirrors sklearn.utils.validation.column_or_1d.
 */
export function columnOr1d(y: unknown): Float64Array {
  if (y instanceof Float64Array) return y;
  if (y instanceof Int32Array) return new Float64Array(y);
  if (Array.isArray(y)) return new Float64Array(y as number[]);
  throw new Error("y must be a Float64Array, Int32Array, or number array.");
}

/**
 * Return indices that would sort an array. Mirrors numpy.argsort.
 */
export function argsort(
  arr: Float64Array | number[],
  reverse = false,
): Int32Array {
  const idx = Array.from({ length: arr.length }, (_, i) => i);
  const a = Array.from(arr);
  if (reverse) idx.sort((i, j) => (a[j] ?? 0) - (a[i] ?? 0));
  else idx.sort((i, j) => (a[i] ?? 0) - (a[j] ?? 0));
  return new Int32Array(idx);
}

/**
 * Shuffle an array in-place using Fisher-Yates. Returns the same array.
 */
export function shuffle<T>(arr: T[], randomState?: number): T[] {
  let seed = (randomState ?? 0) + 1;
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i]!;
    arr[i] = arr[j]!;
    arr[j] = tmp;
  }
  return arr;
}

/**
 * Resample arrays (with optional replacement). Mirrors sklearn.utils.resample.
 */
export function resample<T>(
  arr: T[],
  options: { nSamples?: number; replace?: boolean; randomState?: number } = {},
): T[] {
  const n = arr.length;
  const nSamples = options.nSamples ?? n;
  const replace = options.replace ?? true;

  let seed = (options.randomState ?? 0) + 1;
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };

  if (replace) {
    return Array.from({ length: nSamples }, () => arr[Math.floor(rng() * n)]!);
  }

  // Without replacement: sample nSamples from arr
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i]!;
    indices[i] = indices[j]!;
    indices[j] = tmp;
  }
  return indices.slice(0, nSamples).map((i) => arr[i]!);
}

/** Compute unique values and counts. Mirrors numpy.unique with return_counts. */
export function unique(arr: Int32Array | number[]): {
  values: Int32Array;
  counts: Int32Array;
} {
  const counts = new Map<number, number>();
  for (const v of arr) counts.set(v, (counts.get(v) ?? 0) + 1);
  const sortedValues = Array.from(counts.keys()).sort((a, b) => a - b);
  return {
    values: new Int32Array(sortedValues),
    counts: new Int32Array(sortedValues.map((v) => counts.get(v) ?? 0)),
  };
}
