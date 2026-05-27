/**
 * Extended covariance estimation: Oracle Approximating Shrinkage (OAS),
 * Ledoit-Wolf analytical estimator, and covariance comparison utilities.
 */

/** Ledoit-Wolf analytical shrinkage coefficient. */
export function ledoitWolfShrinkage(X: Float64Array[]): number {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n <= 1 || p === 0) return 0;

  // Sample covariance
  const mean = new Float64Array(p);
  for (const xi of X) {
    for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / n;

  const S = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of X) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        S[j]![k] = (S[j]![k] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) * ((xi[k] ?? 0) - (mean[k] ?? 0));
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) S[j]![k] = (S[j]![k] ?? 0) / n;
  }

  let trS = 0, trS2 = 0, trS_sq = 0;
  for (let j = 0; j < p; j++) trS += S[j]![j] ?? 0;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) trS2 += (S[j]![k] ?? 0) ** 2;
  }
  trS_sq = trS ** 2;

  // LW formula: delta = (((n-2)/n * trS2 + trS_sq) / ((n+2) * (trS2 - trS_sq/p)))
  const num = ((n - 2) / n) * trS2 + trS_sq;
  const den = (n + 2) * (trS2 - trS_sq / p);
  return den === 0 ? 1 : Math.min(1, Math.max(0, num / den));
}

/** OAS shrinkage estimator. */
export function oasShrinkage(X: Float64Array[]): number {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n <= 1 || p === 0) return 0;

  const mean = new Float64Array(p);
  for (const xi of X) {
    for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / n;

  const S = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of X) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        S[j]![k] = (S[j]![k] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) * ((xi[k] ?? 0) - (mean[k] ?? 0));
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) S[j]![k] = (S[j]![k] ?? 0) / n;
  }

  let trS = 0, trS2 = 0;
  for (let j = 0; j < p; j++) trS += S[j]![j] ?? 0;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) trS2 += (S[j]![k] ?? 0) ** 2;
  }

  const rho = (1 - 2 / p) * trS2 + trS ** 2;
  const gamma = (n + 1 - 2 / p) * (trS2 - trS ** 2 / p);
  return gamma === 0 ? 1 : Math.min(1, Math.max(0, rho / ((n + 1 - 2 / p) * gamma)));
}

/** Shrink sample covariance toward identity: Sigma = (1-alpha)*S + alpha*mu*I */
export function shrunkCovariance(
  X: Float64Array[],
  shrinkage: number,
): Float64Array[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;

  const mean = new Float64Array(p);
  for (const xi of X) {
    for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / n;

  const S = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of X) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        S[j]![k] = (S[j]![k] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) * ((xi[k] ?? 0) - (mean[k] ?? 0));
      }
    }
  }

  let trace = 0;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) S[j]![k] = (S[j]![k] ?? 0) / n;
    trace += S[j]![j] ?? 0;
  }
  const mu = trace / p;

  return S.map((row, j) =>
    row.map((v, k) => (1 - shrinkage) * v + (j === k ? shrinkage * mu : 0))
  );
}

/** Frobenius distance between two covariance matrices. */
export function covarianceFrobeniusDistance(A: Float64Array[], B: Float64Array[]): number {
  let dist = 0;
  for (let i = 0; i < A.length; i++) {
    const ai = A[i];
    const bi = B[i];
    if (ai === undefined || bi === undefined) continue;
    for (let j = 0; j < ai.length; j++) dist += ((ai[j] ?? 0) - (bi[j] ?? 0)) ** 2;
  }
  return Math.sqrt(dist);
}

/** Compute log-determinant of a symmetric positive definite matrix (via Cholesky). */
export function logDetCovariance(S: Float64Array[]): number {
  const p = S.length;
  // Cholesky decomposition L such that S = L L^T
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      if (i === j) {
        const val = (S[i]![i] ?? 0) - sum;
        L[i]![i] = val > 0 ? Math.sqrt(val) : 1e-10;
      } else {
        L[i]![j] = ((S[i]![j] ?? 0) - sum) / (L[j]![j] ?? 1e-10);
      }
    }
  }
  let logDet = 0;
  for (let i = 0; i < p; i++) logDet += Math.log(Math.max(L[i]![i] ?? 1e-10, 1e-10));
  return 2 * logDet;
}
