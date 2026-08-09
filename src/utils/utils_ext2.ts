/**
 * Additional utility functions: resample, shuffle, safe_sqr, logsumexp.
 * Mirrors sklearn.utils extras.
 */

export function resample<T>(
  arrays: T[][],
  options: { nSamples?: number; replace?: boolean; randomState?: number } = {},
): T[][] {
  if (arrays.length === 0) return [];
  const n = arrays[0]?.length ?? 0;
  const nSamples = options.nSamples ?? n;
  const replace = options.replace ?? true;

  let rng = options.randomState ?? 0;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };

  let indices: number[];
  if (replace) {
    indices = Array.from({ length: nSamples }, () => Math.floor(nextRand() * n));
  } else {
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < Math.min(nSamples, n); i++) {
      const j = Math.floor(nextRand() * (n - i)) + i;
      const tmp = pool[i] ?? 0;
      pool[i] = pool[j] ?? 0;
      pool[j] = tmp;
    }
    indices = pool.slice(0, Math.min(nSamples, n));
  }

  return arrays.map((arr) => indices.map((i) => arr[i] as T));
}

export function shuffleArray<T>(arr: T[], randomState = 0): T[] {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(nextRand() * (i + 1));
    const tmp = out[i];
    out[i] = out[j] as T;
    out[j] = tmp as T;
  }
  return out;
}

export function safeSqr(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] ?? 0) ** 2;
  return out;
}

export function logsumexp(x: Float64Array): number {
  let maxVal = -Number.MAX_VALUE;
  for (const v of x) if (v > maxVal) maxVal = v;
  let s = 0;
  for (const v of x) s += Math.exp(v - maxVal);
  return maxVal + Math.log(s);
}

export function softmax(x: Float64Array): Float64Array {
  const logZ = logsumexp(x);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.exp((x[i] ?? 0) - logZ);
  return out;
}

export function columnNorms(X: Float64Array[], ord: 1 | 2 = 2): Float64Array {
  const nFeatures = X[0]?.length ?? 0;
  const norms = new Float64Array(nFeatures);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) {
      if (ord === 1) norms[j] = (norms[j] ?? 0) + Math.abs(row[j] ?? 0);
      else norms[j] = (norms[j] ?? 0) + (row[j] ?? 0) ** 2;
    }
  }
  if (ord === 2) {
    for (let j = 0; j < nFeatures; j++) norms[j] = Math.sqrt(norms[j] ?? 0);
  }
  return norms;
}

export function rowNorms(X: Float64Array[], squared = false): Float64Array {
  const norms = new Float64Array(X.length);
  for (let i = 0; i < X.length; i++) {
    let s = 0;
    for (const v of X[i] ?? []) s += v ** 2;
    norms[i] = squared ? s : Math.sqrt(s);
  }
  return norms;
}

export function weightedMode(
  values: Int32Array,
  weights: Float64Array,
): { mode: number; score: number } {
  const scores = new Map<number, number>();
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? 0;
    scores.set(v, (scores.get(v) ?? 0) + (weights[i] ?? 0));
  }
  let bestVal = 0;
  let bestScore = -1;
  for (const [v, s] of scores) {
    if (s > bestScore) {
      bestScore = s;
      bestVal = v;
    }
  }
  return { mode: bestVal, score: bestScore };
}
