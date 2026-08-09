/**
 * Extended statistical utilities: descriptive stats, hypothesis tests,
 * distribution functions, and statistical distances.
 */

/** Descriptive statistics for a numeric array. */
export interface DescriptiveStats {
  mean: number;
  median: number;
  std: number;
  variance: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  iqr: number;
  skewness: number;
  kurtosis: number;
}

export function describeArray(x: Float64Array): DescriptiveStats {
  const n = x.length;
  if (n === 0) return { mean: 0, median: 0, std: 0, variance: 0, min: 0, max: 0, q1: 0, q3: 0, iqr: 0, skewness: 0, kurtosis: 0 };

  const sorted = new Float64Array(x).sort();
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);

  const median = n % 2 === 0 ? ((sorted[n / 2 - 1] ?? 0) + (sorted[n / 2] ?? 0)) / 2 : (sorted[Math.floor(n / 2)] ?? 0);
  const q1 = sorted[Math.floor(n * 0.25)] ?? 0;
  const q3 = sorted[Math.floor(n * 0.75)] ?? 0;

  let m3 = 0, m4 = 0;
  for (const v of x) {
    m3 += ((v - mean) / (std + 1e-10)) ** 3;
    m4 += ((v - mean) / (std + 1e-10)) ** 4;
  }
  const skewness = m3 / n;
  const kurtosis = m4 / n - 3;

  return { mean, median, std, variance, min: sorted[0] ?? 0, max: sorted[n - 1] ?? 0, q1, q3, iqr: q3 - q1, skewness, kurtosis };
}

/** Pearson correlation coefficient. */
export function pearsonCorrelation(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let cov = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) {
    cov += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my);
    sx += ((x[i] ?? 0) - mx) ** 2;
    sy += ((y[i] ?? 0) - my) ** 2;
  }
  return cov / (Math.sqrt(sx * sy) + 1e-10);
}

/** Spearman rank correlation coefficient. */
export function spearmanCorrelation(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  const rankX = ranks(x);
  const rankY = ranks(y);
  return pearsonCorrelation(rankX, rankY);
}

function ranks(x: Float64Array): Float64Array {
  const sorted = Array.from(x).map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const r = new Float64Array(x.length);
  for (let rank = 0; rank < sorted.length; rank++) r[sorted[rank]!.i] = rank + 1;
  return r;
}

/** Kolmogorov-Smirnov statistic (two-sample). */
export function ksStatistic(x: Float64Array, y: Float64Array): number {
  const allVals = [...Array.from(x), ...Array.from(y)].sort((a, b) => a - b);
  let maxDiff = 0;
  const nx = x.length, ny = y.length;
  for (const v of allVals) {
    const cdfX = Array.from(x).filter((xi) => xi <= v).length / nx;
    const cdfY = Array.from(y).filter((yi) => yi <= v).length / ny;
    maxDiff = Math.max(maxDiff, Math.abs(cdfX - cdfY));
  }
  return maxDiff;
}

/** Wasserstein distance (Earth Mover's Distance) between two 1D distributions. */
export function wassersteinDistance(x: Float64Array, y: Float64Array): number {
  const sx = new Float64Array(x).sort();
  const sy = new Float64Array(y).sort();
  const n = Math.max(sx.length, sy.length);

  // Interpolate to same length
  const interpX = interpolate(sx, n);
  const interpY = interpolate(sy, n);

  let dist = 0;
  for (let i = 0; i < n; i++) dist += Math.abs((interpX[i] ?? 0) - (interpY[i] ?? 0));
  return dist / n;
}

function interpolate(arr: Float64Array, n: number): Float64Array {
  if (arr.length === n) return arr;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (arr.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, arr.length - 1);
    result[i] = (arr[lo] ?? 0) + (t - lo) * ((arr[hi] ?? 0) - (arr[lo] ?? 0));
  }
  return result;
}

/** Kullback-Leibler divergence (assumes probabilities sum to 1). */
export function klDivergence(p: Float64Array, q: Float64Array): number {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i] ?? 0;
    const qi = Math.max(q[i] ?? 1e-10, 1e-10);
    if (pi > 0) kl += pi * Math.log(pi / qi);
  }
  return kl;
}

/** Jensen-Shannon divergence (symmetric). */
export function jsDivergence(p: Float64Array, q: Float64Array): number {
  const m = p.map((pi, i) => (pi + (q[i] ?? 0)) / 2);
  return 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);
}

/** Cramér's V — measure of association between two categorical variables. */
export function cramersV(x: Int32Array, y: Int32Array): number {
  const n = x.length;
  const xVals = [...new Set(Array.from(x))];
  const yVals = [...new Set(Array.from(y))];

  let chi2 = 0;
  for (const xv of xVals) {
    for (const yv of yVals) {
      const observed = Array.from(x).filter((xi, i) => xi === xv && (y[i] ?? -1) === yv).length;
      const expected = Array.from(x).filter((xi) => xi === xv).length *
        Array.from(y).filter((yi) => yi === yv).length / n;
      if (expected > 0) chi2 += (observed - expected) ** 2 / expected;
    }
  }
  const minDim = Math.min(xVals.length - 1, yVals.length - 1);
  return Math.sqrt(chi2 / (n * (minDim + 1e-10)));
}
