/**
 * Utils extensions: statistical tests, effect size measures, bootstrap utilities.
 * Mirrors sklearn.utils additional methods.
 */

/** Compute Cohen's d effect size between two groups. */
export function cohensD(group1: Float64Array, group2: Float64Array): number {
  const n1 = group1.length;
  const n2 = group2.length;
  const mean1 = group1.reduce((s, v) => s + v, 0) / (n1 || 1);
  const mean2 = group2.reduce((s, v) => s + v, 0) / (n2 || 1);
  const var1 = group1.reduce((s, v) => s + (v - mean1) ** 2, 0) / (n1 - 1 || 1);
  const var2 = group2.reduce((s, v) => s + (v - mean2) ** 2, 0) / (n2 - 1 || 1);
  const pooledStd = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2 || 1));
  return (mean1 - mean2) / (pooledStd || 1);
}

/** One-sample t-test: test whether mean of sample differs from a hypothesized value. */
export function tTest1Sample(x: Float64Array, popmean: number): { statistic: number; df: number } {
  const n = x.length;
  const mean = x.reduce((s, v) => s + v, 0) / (n || 1);
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1 || 1);
  const se = Math.sqrt(variance / (n || 1));
  return { statistic: (mean - popmean) / (se || 1), df: n - 1 };
}

/** Independent samples t-test. */
export function tTest2Sample(x: Float64Array, y: Float64Array, equalVar = true): { statistic: number; df: number } {
  const nx = x.length;
  const ny = y.length;
  const mx = x.reduce((s, v) => s + v, 0) / (nx || 1);
  const my = y.reduce((s, v) => s + v, 0) / (ny || 1);
  const vx = x.reduce((s, v) => s + (v - mx) ** 2, 0) / (nx - 1 || 1);
  const vy = y.reduce((s, v) => s + (v - my) ** 2, 0) / (ny - 1 || 1);

  if (equalVar) {
    const sp = Math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 || 1));
    const se = sp * Math.sqrt(1 / nx + 1 / ny);
    return { statistic: (mx - my) / (se || 1), df: nx + ny - 2 };
  }
  // Welch's t-test
  const se = Math.sqrt(vx / nx + vy / ny);
  const df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1 || 1) + (vy / ny) ** 2 / (ny - 1 || 1));
  return { statistic: (mx - my) / (se || 1), df };
}

/** Mann-Whitney U test (non-parametric). */
export function mannWhitneyU(x: Float64Array, y: Float64Array): { U: number; z: number } {
  const nx = x.length;
  const ny = y.length;
  let U1 = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if ((x[i] ?? 0) > (y[j] ?? 0)) U1++;
      else if ((x[i] ?? 0) === (y[j] ?? 0)) U1 += 0.5;
    }
  }
  const U2 = nx * ny - U1;
  const U = Math.min(U1, U2);
  const mu = nx * ny / 2;
  const sigma = Math.sqrt(nx * ny * (nx + ny + 1) / 12);
  return { U, z: (U - mu) / (sigma || 1) };
}

/** Bootstrap confidence interval for a statistic. */
export function bootstrapCI(
  x: Float64Array,
  statFn: (sample: Float64Array) => number,
  n_bootstrap = 1000,
  alpha = 0.05,
): { lower: number; upper: number; estimate: number } {
  const n = x.length;
  const bootstrapStats: number[] = [];
  for (let b = 0; b < n_bootstrap; b++) {
    const sample = new Float64Array(n).map(() => x[Math.floor(Math.random() * n)] ?? 0);
    bootstrapStats.push(statFn(sample));
  }
  bootstrapStats.sort((a, b) => a - b);
  const lower = bootstrapStats[Math.floor(alpha / 2 * n_bootstrap)] ?? 0;
  const upper = bootstrapStats[Math.floor((1 - alpha / 2) * n_bootstrap)] ?? 0;
  return { lower, upper, estimate: statFn(x) };
}

/** Compute Spearman rank correlation coefficient. */
export function spearmanRho(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  const rankX = ranks(x);
  const rankY = ranks(y);
  let d2sum = 0;
  for (let i = 0; i < n; i++) d2sum += ((rankX[i] ?? 0) - (rankY[i] ?? 0)) ** 2;
  return 1 - 6 * d2sum / (n * (n * n - 1) || 1);
}

function ranks(x: Float64Array): Float64Array {
  const n = x.length;
  const sorted = Array.from(x).map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[sorted[i]!.i] = i + 1;
  return r;
}

/** Mutual information estimation using k-nearest neighbor method. */
export function mutualInfoEstimate(X: Float64Array[], y: Int32Array, k = 3): Float64Array {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const scores = new Float64Array(d);

  for (let f = 0; f < d; f++) {
    const vals = new Float64Array(X.map(row => row[f] ?? 0));
    let mi = 0;

    const classes = [...new Set(Array.from(y))];
    for (const cls of classes) {
      const clsIdx = Array.from({ length: n }, (_, i) => i).filter(i => y[i] === cls);
      const nc = clsIdx.length;
      for (const i of clsIdx) {
        // k-nearest in class
        const classVals = clsIdx.map(j => Math.abs((vals[j] ?? 0) - (vals[i] ?? 0))).sort((a, b) => a - b);
        const eps_c = classVals[k] ?? 1;
        // k-nearest overall
        const allVals = Array.from(vals).map(v => Math.abs(v - (vals[i] ?? 0))).sort((a, b) => a - b);
        const eps = allVals[k] ?? 1;
        const m = allVals.filter(v => v <= eps_c).length;
        mi += (Math.log(n) + Math.log(nc) - Math.log(m || 1) - Math.log(k)) / n;
      }
    }
    scores[f] = Math.max(0, mi);
  }
  return scores;
}
