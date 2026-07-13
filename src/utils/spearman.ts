/**
 * Spearman correlation and rank-based statistics.
 * Port of sklearn.utils.stats (Spearman) and scipy.stats.spearmanr
 */

/**
 * Compute ranks of an array (average rank for ties).
 */
export function rankData(x: Float64Array): Float64Array {
  const n = x.length;
  const indexed = Array.from(x, (v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n - 1 && indexed[j + 1]!.v === indexed[j]!.v) j++;
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k]!.i] = rank;
    i = j + 1;
  }
  return ranks;
}

/**
 * Compute Spearman rank correlation coefficient.
 * Port of scipy.stats.spearmanr
 */
export function spearmanr(
  x: Float64Array,
  y: Float64Array,
): { correlation: number; pvalue: number } {
  if (x.length !== y.length) throw new Error("x and y must have same length");
  const n = x.length;
  const rx = rankData(x);
  const ry = rankData(y);

  // Pearson correlation of ranks
  let meanRx = 0;
  let meanRy = 0;
  for (let i = 0; i < n; i++) {
    meanRx += rx[i]! / n;
    meanRy += ry[i]! / n;
  }

  let cov = 0;
  let varRx = 0;
  let varRy = 0;
  for (let i = 0; i < n; i++) {
    cov += (rx[i]! - meanRx) * (ry[i]! - meanRy);
    varRx += (rx[i]! - meanRx) ** 2;
    varRy += (ry[i]! - meanRy) ** 2;
  }

  const corr = cov / (Math.sqrt(varRx * varRy) + 1e-15);

  // t-statistic for significance
  const t = corr * Math.sqrt((n - 2) / (1 - corr ** 2 + 1e-15));
  // Approximate p-value using normal approximation
  const pvalue = 2 * (1 - normalCDF(Math.abs(t)));

  return { correlation: corr, pvalue };
}

function normalCDF(z: number): number {
  // Abramowitz & Stegun approximation
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * Math.abs(z));
  const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  const erf = 1 - poly * Math.exp(-z * z);
  return 0.5 * (1 + (z >= 0 ? erf : -erf));
}

/**
 * Compute Kendall's tau correlation.
 */
export function kendalltau(
  x: Float64Array,
  y: Float64Array,
): { correlation: number; pvalue: number } {
  const n = x.length;
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = (x[i]! - x[j]!) * (y[i]! - y[j]!);
      if (dx > 0) concordant++;
      else if (dx < 0) discordant++;
    }
  }
  const nPairs = (n * (n - 1)) / 2;
  const tau = (concordant - discordant) / nPairs;
  // Approximate p-value
  const z = tau / Math.sqrt((2 * (2 * n + 5)) / (9 * nPairs));
  const pvalue = 2 * (1 - normalCDF(Math.abs(z)));
  return { correlation: tau, pvalue };
}

/**
 * Compute Pearson correlation.
 */
export function pearsonr(
  x: Float64Array,
  y: Float64Array,
): { correlation: number; pvalue: number } {
  const n = x.length;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += x[i]! / n;
    meanY += y[i]! / n;
  }

  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    cov += (x[i]! - meanX) * (y[i]! - meanY);
    varX += (x[i]! - meanX) ** 2;
    varY += (y[i]! - meanY) ** 2;
  }

  const corr = cov / (Math.sqrt(varX * varY) + 1e-15);
  const t = corr * Math.sqrt((n - 2) / (1 - corr ** 2 + 1e-15));
  const pvalue = 2 * (1 - normalCDF(Math.abs(t)));
  return { correlation: corr, pvalue };
}

/**
 * Spearman correlation matrix for multiple variables.
 */
export function spearmanMatrix(X: Float64Array[]): Float64Array[] {
  const n = X.length;
  const matrix: Float64Array[] = Array.from(
    { length: n },
    () => new Float64Array(n),
  );
  for (let i = 0; i < n; i++) {
    matrix[i]![i] = 1.0;
    for (let j = i + 1; j < n; j++) {
      const { correlation } = spearmanr(X[i]!, X[j]!);
      matrix[i]![j] = correlation;
      matrix[j]![i] = correlation;
    }
  }
  return matrix;
}
