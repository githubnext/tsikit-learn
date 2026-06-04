/**
 * Utils extensions: statistical tests, hypothesis testing, bootstrap utilities
 */

export interface BootstrapResult {
  statistic: number;
  ciLow: number;
  ciHigh: number;
  bootstrapDistribution: Float64Array;
}

export function bootstrapCI(
  data: Float64Array,
  statFn: (x: Float64Array) => number,
  options: { nResamples?: number; ci?: number; seed?: number } = {}
): BootstrapResult {
  const nResamples = options.nResamples ?? 9999;
  const ci = options.ci ?? 0.95;
  const n = data.length;
  const dist = new Float64Array(nResamples);
  let rng = options.seed ?? 42;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

  for (let b = 0; b < nResamples; b++) {
    const sample = new Float64Array(n);
    for (let i = 0; i < n; i++) sample[i] = data[Math.floor(rand() * n)] ?? 0;
    dist[b] = statFn(sample);
  }
  dist.sort();

  const alpha = (1 - ci) / 2;
  const lo = Math.floor(alpha * nResamples);
  const hi = Math.floor((1 - alpha) * nResamples);
  return {
    statistic: statFn(data),
    ciLow: dist[lo] ?? 0,
    ciHigh: dist[hi] ?? 0,
    bootstrapDistribution: dist,
  };
}

export function permutationTest(
  x: Float64Array,
  y: Float64Array,
  statFn: (a: Float64Array, b: Float64Array) => number,
  options: { nPermutations?: number; alternative?: 'two-sided' | 'greater' | 'less'; seed?: number } = {}
): { statistic: number; pValue: number; nullDistribution: Float64Array } {
  const nPermutations = options.nPermutations ?? 9999;
  const alternative = options.alternative ?? 'two-sided';
  const observed = statFn(x, y);
  const combined = new Float64Array([...x, ...y]);
  const n = x.length;
  const null_ = new Float64Array(nPermutations);
  let rng = options.seed ?? 42;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

  for (let b = 0; b < nPermutations; b++) {
    // Fisher-Yates shuffle
    const perm = combined.slice();
    for (let i = perm.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = perm[i]!; perm[i] = perm[j]!; perm[j] = tmp;
    }
    null_[b] = statFn(perm.slice(0, n), perm.slice(n));
  }

  let count = 0;
  for (let b = 0; b < nPermutations; b++) {
    const v = null_[b] ?? 0;
    if (alternative === 'two-sided' && Math.abs(v) >= Math.abs(observed)) count++;
    else if (alternative === 'greater' && v >= observed) count++;
    else if (alternative === 'less' && v <= observed) count++;
  }
  return { statistic: observed, pValue: (count + 1) / (nPermutations + 1), nullDistribution: null_ };
}

export function mannWhitneyU(x: Float64Array, y: Float64Array): { u: number; pValue: number } {
  const nx = x.length, ny = y.length;
  let u = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const xi = x[i] ?? 0, yj = y[j] ?? 0;
      if (xi > yj) u += 1;
      else if (xi === yj) u += 0.5;
    }
  }
  // Normal approximation
  const mu = nx * ny / 2;
  const sigma = Math.sqrt(nx * ny * (nx + ny + 1) / 12);
  const z = (u - mu) / sigma;
  const pValue = 2 * (1 - normalCDF(Math.abs(z)));
  return { u, pValue };
}

export function wilcoxonSignedRankTest(x: Float64Array, y: Float64Array): { statistic: number; pValue: number } {
  const n = x.length;
  const diffs: { abs: number; sign: number }[] = [];
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - (y[i] ?? 0);
    if (d !== 0) diffs.push({ abs: Math.abs(d), sign: Math.sign(d) });
  }
  diffs.sort((a, b) => a.abs - b.abs);
  let wPlus = 0, wMinus = 0;
  for (let i = 0; i < diffs.length; i++) {
    const rank = i + 1;
    if ((diffs[i]?.sign ?? 0) > 0) wPlus += rank;
    else wMinus += rank;
  }
  const T = Math.min(wPlus, wMinus);
  const m = diffs.length;
  const mu = m * (m + 1) / 4;
  const sigma = Math.sqrt(m * (m + 1) * (2 * m + 1) / 24);
  const z = (T - mu) / sigma;
  return { statistic: T, pValue: 2 * (1 - normalCDF(Math.abs(z))) };
}

export function kruskalWallis(groups: Float64Array[]): { hStatistic: number; pValue: number } {
  const n = groups.reduce((s, g) => s + g.length, 0);
  const all: { value: number; group: number }[] = [];
  for (let g = 0; g < groups.length; g++) {
    for (const v of groups[g]!) all.push({ value: v, group: g });
  }
  all.sort((a, b) => a.value - b.value);
  const ranks = new Float64Array(n);
  for (let i = 0; i < n; i++) ranks[i] = i + 1;

  let H = 0;
  for (let g = 0; g < groups.length; g++) {
    const ng = groups[g]!.length;
    let rankSum = 0;
    for (let i = 0; i < n; i++) if ((all[i]?.group ?? -1) === g) rankSum += ranks[i] ?? 0;
    H += rankSum ** 2 / ng;
  }
  H = (12 / (n * (n + 1))) * H - 3 * (n + 1);
  const df = groups.length - 1;
  const pValue = 1 - chi2CDF(H, df);
  return { hStatistic: H, pValue };
}

export function friedmanTest(data: Float64Array[]): { statistic: number; pValue: number } {
  const k = data[0]?.length ?? 0;
  const n = data.length;
  const rankMatrix = data.map(row => {
    const sorted = row.slice().sort((a, b) => a - b);
    return new Float64Array(row.map(v => sorted.indexOf(v) + 1));
  });
  let statistic = 0;
  for (let j = 0; j < k; j++) {
    const colSum = rankMatrix.reduce((s, row) => s + (row[j] ?? 0), 0);
    statistic += colSum ** 2;
  }
  statistic = (12 / (n * k * (k + 1))) * statistic - 3 * n * (k + 1);
  const pValue = 1 - chi2CDF(statistic, k - 1);
  return { statistic, pValue };
}

export function leveneTest(groups: Float64Array[]): { statistic: number; pValue: number } {
  const k = groups.length;
  const N = groups.reduce((s, g) => s + g.length, 0);
  const means = groups.map(g => g.reduce((s, v) => s + v, 0) / g.length);
  const W = groups.map((g, i) => g.map(v => Math.abs(v - (means[i] ?? 0))));
  const grandMean = W.flat().reduce((s, v) => s + v, 0) / N;
  const Wmeans = W.map(wi => wi.reduce((s, v) => s + v, 0) / wi.length);
  let numerator = 0;
  for (let i = 0; i < k; i++) numerator += W[i]!.length * ((Wmeans[i]! - grandMean) ** 2);
  numerator *= (N - k) / (k - 1);
  let denominator = 0;
  for (let i = 0; i < k; i++) {
    for (const v of W[i]!) denominator += (v - (Wmeans[i] ?? 0)) ** 2;
  }
  const F = numerator / denominator;
  return { statistic: F, pValue: 1 - fCDF(F, k - 1, N - k) };
}

export function normalCDF(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989422820 * Math.exp(-(z * z) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))));
  return z > 0 ? 1 - p : p;
}

export function chi2CDF(x: number, df: number): number {
  if (x <= 0) return 0;
  return gammaIncomplete(df / 2, x / 2);
}

export function fCDF(x: number, df1: number, df2: number): number {
  if (x <= 0) return 0;
  const z = (df1 * x) / (df1 * x + df2);
  return betaIncomplete(df1 / 2, df2 / 2, z);
}

function gammaIncomplete(a: number, x: number): number {
  if (x < 0) return 0;
  let sum = 1 / a;
  let term = 1 / a;
  for (let n = 1; n < 100; n++) {
    term *= x / (a + n);
    sum += term;
    if (term < 1e-10) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

function betaIncomplete(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta) / a;
  let sum = 1, term = 1;
  for (let n = 1; n < 200; n++) {
    term *= (a + n - 1) * x / n;
    sum += term / (a + n);
    if (Math.abs(term) < 1e-10) break;
  }
  return front * sum;
}

function lgamma(x: number): number {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  let xx = x - 1;
  let y = c[0]!;
  for (let i = 1; i < g + 2; i++) y += (c[i] ?? 0) / (xx + i);
  const t = xx + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(y);
}

export function cohenD(x: Float64Array, y: Float64Array): number {
  const nx = x.length, ny = y.length;
  const mx = x.reduce((s, v) => s + v, 0) / nx;
  const my = y.reduce((s, v) => s + v, 0) / ny;
  const vx = x.reduce((s, v) => s + (v - mx) ** 2, 0) / (nx - 1);
  const vy = y.reduce((s, v) => s + (v - my) ** 2, 0) / (ny - 1);
  const sp = Math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2));
  return (mx - my) / sp;
}

export function etaSquared(groups: Float64Array[]): number {
  const allValues = groups.flat();
  const grandMean = allValues.reduce((s, v) => s + v, 0) / allValues.length;
  const ssBetween = groups.reduce((s, g) => {
    const gm = g.reduce((ss, v) => ss + v, 0) / g.length;
    return s + g.length * (gm - grandMean) ** 2;
  }, 0);
  const ssTotal = allValues.reduce((s, v) => s + (v - grandMean) ** 2, 0);
  return ssBetween / ssTotal;
}
