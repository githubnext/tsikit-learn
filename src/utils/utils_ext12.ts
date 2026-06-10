/**
 * Covariance estimation utilities and statistical helpers.
 */

export function empiricalCovariance(X: Float64Array[], assumeCentered = false): Float64Array[] {
  const n = X.length, p = X[0]?.length ?? 0;
  const mean = assumeCentered
    ? new Float64Array(p)
    : new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[j] ?? 0), 0) / n);
  const cov = Array.from({ length: p }, (_, i) =>
    new Float64Array(p).map((_, j) => X.reduce((s, row) => s + ((row[i] ?? 0) - mean[i]!) * ((row[j] ?? 0) - mean[j]!), 0) / (n - (assumeCentered ? 0 : 1)))
  );
  return cov;
}

export function shrunkCovariance(X: Float64Array[], shrinkage = 0.1): Float64Array[] {
  const cov = empiricalCovariance(X);
  const p = cov.length;
  const trace = cov.reduce((s, row, i) => s + (row[i] ?? 0), 0);
  const mu = trace / p;
  return cov.map((row, i) => new Float64Array(row.map((v, j) => (1 - shrinkage) * v + (i === j ? shrinkage * mu : 0))));
}

export function ledoitWolf(X: Float64Array[]): { covariance: Float64Array[]; shrinkage: number } {
  const n = X.length, p = X[0]?.length ?? 0;
  const S = empiricalCovariance(X);
  const trace_S = S.reduce((s, row, i) => s + (row[i] ?? 0), 0);
  const trace_S2 = S.reduce((s, row) => s + row.reduce((ss, v) => ss + v * v, 0), 0);
  // Ledoit-Wolf analytical formula
  const mu = trace_S / p;
  const delta = S.reduce((s, row, i) => s + row.reduce((ss, v, j) => ss + (v - (i === j ? mu : 0)) ** 2, 0), 0) / p;
  const beta2 = Array.from({ length: n }, (_, i) => {
    const xi = X[i]!;
    return S.reduce((s, row, r) => {
      const cross = xi.reduce((ss, v, k) => ss + v * (xi[k] ?? 0) * (row[k] ?? 0), 0);
      return s + (cross - (S[r]![r] ?? 0)) ** 2;
    }, 0);
  }).reduce((s, v) => s + v, 0) / (n ** 2 * p);
  const shrinkage = Math.min(1, Math.max(0, beta2 / delta));
  return { covariance: shrunkCovariance(X, shrinkage), shrinkage };
}

export function oracleApproximatingShrinkage(X: Float64Array[]): { covariance: Float64Array[]; shrinkage: number } {
  const n = X.length, p = X[0]?.length ?? 0;
  const S = empiricalCovariance(X);
  const traceS = S.reduce((s, row, i) => s + (row[i] ?? 0), 0);
  const traceS2 = S.reduce((s, row) => s + row.reduce((ss, v) => ss + v * v, 0), 0);
  // OAS formula
  const rho = Math.min(1, Math.max(0, ((1 - 2 / p) * traceS2 + traceS ** 2) / ((n + 1 - 2 / p) * (traceS2 - traceS ** 2 / p))));
  return { covariance: shrunkCovariance(X, rho), shrinkage: rho };
}

export function mahalanobisDistance(x: Float64Array, mean: Float64Array, precisionMatrix: Float64Array[]): number {
  const diff = new Float64Array(x.map((v, i) => v - (mean[i] ?? 0)));
  const tmp = new Float64Array(diff.length).map((_, i) => precisionMatrix[i]!.reduce((s, v, j) => s + v * (diff[j] ?? 0), 0));
  return Math.sqrt(Math.max(diff.reduce((s, v, i) => s + v * (tmp[i] ?? 0), 0), 0));
}

export function pearsonCorrelation(X: Float64Array[]): Float64Array[] {
  const p = X[0]?.length ?? 0;
  const cov = empiricalCovariance(X);
  return Array.from({ length: p }, (_, i) =>
    new Float64Array(p).map((_, j) => {
      const stdI = Math.sqrt(cov[i]![i] ?? 0), stdJ = Math.sqrt(cov[j]![j] ?? 0);
      return stdI > 0 && stdJ > 0 ? (cov[i]![j] ?? 0) / (stdI * stdJ) : 0;
    })
  );
}
