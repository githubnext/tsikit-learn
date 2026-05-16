/**
 * Statistical utility functions.
 */

export function mean(x: Float64Array): number {
  if (x.length === 0) return Number.NaN;
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0;
  return s / x.length;
}

export function variance(x: Float64Array, ddof = 0): number {
  if (x.length === 0) return Number.NaN;
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += ((x[i] ?? 0) - m) ** 2;
  return s / (x.length - ddof);
}

export function std(x: Float64Array, ddof = 0): number {
  return Math.sqrt(variance(x, ddof));
}

export function covariance(x: Float64Array, y: Float64Array): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return Number.NaN;
  const mx = mean(x);
  const my = mean(y);
  let s = 0;
  for (let i = 0; i < n; i++) s += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my);
  return s / n;
}

export function pearsonR(x: Float64Array, y: Float64Array): number {
  const sx = std(x);
  const sy = std(y);
  if (sx === 0 || sy === 0) return Number.NaN;
  return covariance(x, y) / (sx * sy);
}

function rankArray(x: Float64Array): Float64Array {
  const idx = Array.from({ length: x.length }, (_, i) => i);
  idx.sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0));
  const ranks = new Float64Array(x.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j < idx.length && (x[idx[j] ?? 0] ?? 0) === (x[idx[i] ?? 0] ?? 0))
      j++;
    const r = (i + j - 1) / 2 + 1;
    for (let k = i; k < j; k++) ranks[idx[k] ?? 0] = r;
    i = j;
  }
  return ranks;
}

export function spearmanR(x: Float64Array, y: Float64Array): number {
  return pearsonR(rankArray(x), rankArray(y));
}

/** Welch's t-test — returns statistic and approximate p-value via t-distribution CDF. */
export function tTest(
  a: Float64Array,
  b: Float64Array,
): { statistic: number; pValue: number } {
  const na = a.length;
  const nb = b.length;
  const ma = mean(a);
  const mb = mean(b);
  const va = variance(a, 1) / na;
  const vb = variance(b, 1) / nb;
  const se = Math.sqrt(va + vb);
  if (se === 0) return { statistic: 0, pValue: 1 };
  const t = (ma - mb) / se;
  const df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1));
  const p = 2 * (1 - tCdf(Math.abs(t), df));
  return { statistic: t, pValue: p };
}

/** One-way ANOVA F-test. */
export function fOneWay(...groups: Float64Array[]): {
  statistic: number;
  pValue: number;
} {
  const k = groups.length;
  const allN = groups.reduce((s, g) => s + g.length, 0);
  const grandMean = mean(
    Float64Array.from(groups.flatMap((g) => Array.from(g))),
  );
  let ssBetween = 0;
  let ssWithin = 0;
  for (const g of groups) {
    const gm = mean(g);
    ssBetween += g.length * (gm - grandMean) ** 2;
    for (let i = 0; i < g.length; i++) ssWithin += ((g[i] ?? 0) - gm) ** 2;
  }
  const dfBetween = k - 1;
  const dfWithin = allN - k;
  if (dfBetween <= 0 || dfWithin <= 0 || ssWithin === 0)
    return { statistic: Number.NaN, pValue: Number.NaN };
  const F = ssBetween / dfBetween / (ssWithin / dfWithin);
  const p = 1 - fCdf(F, dfBetween, dfWithin);
  return { statistic: F, pValue: p };
}

// ── Approximation helpers ────────────────────────────────────────────────────

/** Regularised incomplete beta function via continued-fraction (Lentz). */
function betaInc(a: number, b: number, x: number): number {
  if (x < 0 || x > 1) return Number.NaN;
  if (x === 0) return 0;
  if (x === 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  // Use symmetry relation for better convergence
  if (x > (a + 1) / (a + b + 2)) return 1 - betaInc(b, a, 1 - x);
  // Lentz continued fraction
  let f = 1;
  let C = f;
  let D = 0;
  for (let m = 0; m <= 200; m++) {
    for (let s = 0; s <= 1; s++) {
      let d: number;
      if (s === 0) {
        if (m === 0) {
          d = 1;
        } else {
          d = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
        }
      } else {
        d = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
      }
      D = 1 + d * D;
      if (Math.abs(D) < 1e-30) D = 1e-30;
      C = 1 + d / C;
      if (Math.abs(C) < 1e-30) C = 1e-30;
      D = 1 / D;
      const delta = C * D;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-10) break;
    }
  }
  return front * (f - 1);
}

function lgamma(x: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  const xm1 = x - 1;
  let a = c[0] ?? 0;
  const t = xm1 + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += (c[i] ?? 0) / (xm1 + i);
  return (
    0.5 * Math.log(2 * Math.PI) + (xm1 + 0.5) * Math.log(t) - t + Math.log(a)
  );
}

/** t-distribution CDF approximation. */
function tCdf(t: number, df: number): number {
  const x = df / (df + t * t);
  return 1 - 0.5 * betaInc(df / 2, 0.5, x);
}

/** F-distribution CDF approximation. */
function fCdf(f: number, d1: number, d2: number): number {
  const x = (d1 * f) / (d1 * f + d2);
  return betaInc(d1 / 2, d2 / 2, x);
}
