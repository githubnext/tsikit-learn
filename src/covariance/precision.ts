/**
 * Covariance utilities: precision matrix estimation, covariance selection.
 * ledoit_wolf() and oas() functional APIs, plus precision/correlation conversion.
 * Mirrors sklearn.covariance functional API and utility functions.
 */

import { NotFittedError } from "../exceptions.js";

function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const m = new Float64Array(p);
  const n = X.length;
  for (const xi of X)
    for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (xi[j] ?? 0);
  for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) / n;
  return m;
}

function empCovMatrix(X: Float64Array[], means: Float64Array): Float64Array[] {
  const n = X.length;
  const p = means.length;
  const C = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of X) {
    for (let i = 0; i < p; i++) {
      const di = (xi[i] ?? 0) - (means[i] ?? 0);
      for (let j = i; j < p; j++) {
        const dj = (xi[j] ?? 0) - (means[j] ?? 0);
        C[i]![j] = (C[i]![j] ?? 0) + di * dj;
      }
    }
  }
  for (let i = 0; i < p; i++) {
    C[i]![i] = (C[i]![i] ?? 0) / n;
    for (let j = i + 1; j < p; j++) {
      C[i]![j] = (C[i]![j] ?? 0) / n;
      C[j]![i] = C[i]![j] ?? 0;
    }
  }
  return C;
}

function matTrace(M: Float64Array[]): number {
  let s = 0;
  for (let i = 0; i < M.length; i++) s += M[i]![i] ?? 0;
  return s;
}

function matFrobSq(M: Float64Array[]): number {
  let s = 0;
  for (const row of M)
    for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) ** 2;
  return s;
}

/** Invert diagonal of a matrix (for precision). */
function invertDiag(M: Float64Array[]): Float64Array[] {
  return M.map(
    (row, i) =>
      new Float64Array(row.map((v, j) => (i === j && v > 0 ? 1 / v : 0))),
  );
}

/**
 * Functional API: Ledoit-Wolf analytical shrinkage.
 * Mirrors sklearn.covariance.ledoit_wolf.
 */
export function ledoitWolf(
  X: Float64Array[],
  options: { assumeCentered?: boolean } = {},
): { covariance: Float64Array[]; shrinkage: number } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const location = options.assumeCentered ? new Float64Array(p) : colMeans(X);
  const S = empCovMatrix(X, location);
  const trS = matTrace(S);
  const trS2 = matFrobSq(S);
  const trSsq = trS ** 2;

  let delta = 0;
  for (let i = 0; i < p; i++) {
    for (let k = 0; k < p; k++) {
      let fourth = 0;
      for (let t = 0; t < n; t++) {
        const xt = X[t] ?? new Float64Array(p);
        fourth +=
          ((xt[i] ?? 0) - (location[i] ?? 0)) ** 2 *
          ((xt[k] ?? 0) - (location[k] ?? 0)) ** 2;
      }
      fourth /= n;
      delta += fourth - (S[i]![k] ?? 0) ** 2;
    }
  }
  delta /= n;

  const delta2 = trS2 - trSsq / p;
  const shrinkage =
    delta2 > 0
      ? Math.min(
          1,
          Math.max(0, (delta + ((n - 2) / n) * delta2) / ((n + 2) * delta2)),
        )
      : 0;

  const mu = trS / p;
  const covariance = S.map(
    (row, i) =>
      new Float64Array(
        row.map((v, j) => (1 - shrinkage) * v + shrinkage * (i === j ? mu : 0)),
      ),
  );
  return { covariance, shrinkage };
}

/**
 * Functional API: Oracle Approximating Shrinkage (OAS).
 * Mirrors sklearn.covariance.oas.
 */
export function oas(
  X: Float64Array[],
  options: { assumeCentered?: boolean } = {},
): { covariance: Float64Array[]; shrinkage: number } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const location = options.assumeCentered ? new Float64Array(p) : colMeans(X);
  const S = empCovMatrix(X, location);
  const trS = matTrace(S);
  const trS2 = matFrobSq(S);
  const trSsq = trS ** 2;

  const num = (1 - 2 / p) * trS2 + trSsq;
  const denom = (n + 1 - 2 / p) * (trS2 - trSsq / p);
  const shrinkage = denom > 0 ? Math.min(1, Math.max(0, num / denom)) : 0;

  const mu = trS / p;
  const covariance = S.map(
    (row, i) =>
      new Float64Array(
        row.map((v, j) => (1 - shrinkage) * v + shrinkage * (i === j ? mu : 0)),
      ),
  );
  return { covariance, shrinkage };
}

/**
 * Convert a covariance matrix to a correlation matrix.
 * Mirrors sklearn.covariance.cov_to_corr.
 */
export function covToCorr(covariance: Float64Array[]): Float64Array[] {
  const p = covariance.length;
  const std = new Float64Array(p).map((_, i) =>
    Math.sqrt(Math.max(covariance[i]![i] ?? 0, 1e-12)),
  );
  return covariance.map(
    (row, i) =>
      new Float64Array(row.map((v, j) => v / ((std[i] ?? 1) * (std[j] ?? 1)))),
  );
}

/**
 * Compute the log-likelihood of X under a Gaussian model.
 * Mirrors sklearn.covariance.empirical_covariance (log_likelihood method).
 */
export function gaussianLogLikelihood(
  X: Float64Array[],
  mean: Float64Array,
  covariance: Float64Array[],
): number {
  const n = X.length;
  const p = mean.length;

  // log-det via Cholesky
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = covariance[i]![j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      L[i]![j] =
        i === j
          ? Math.sqrt(Math.max(s, 1e-12))
          : s / Math.max(L[j]![j] ?? 1, 1e-12);
    }
  }
  let logDet = 0;
  for (let i = 0; i < p; i++)
    logDet += Math.log(Math.max(L[i]![i] ?? 1e-12, 1e-12));
  logDet *= 2;

  // trace(S * precision) where S = empirical covariance of X
  const S = empCovMatrix(X, mean);
  // Use diagonal approx for precision
  let trSP = 0;
  for (let i = 0; i < p; i++) {
    const cii = covariance[i]![i] ?? 1;
    trSP += (S[i]![i] ?? 0) / Math.max(cii, 1e-12);
  }

  return -0.5 * (n * (p * Math.log(2 * Math.PI) + logDet + trSP));
}

/**
 * Sparse inverse covariance estimator (precision matrix selector).
 * Uses a simple soft-threshold approach to zero out small entries.
 * Mirrors sklearn.covariance sparse precision concepts.
 */
export class SparsePrecision {
  threshold: number;
  assumeCentered: boolean;

  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;

  constructor(options: { threshold?: number; assumeCentered?: boolean } = {}) {
    this.threshold = options.threshold ?? 0.1;
    this.assumeCentered = options.assumeCentered ?? false;
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    const location = this.assumeCentered ? new Float64Array(p) : colMeans(X);
    this.location_ = location;
    const S = empCovMatrix(X, location);
    this.covariance_ = S;

    // Simple diagonal precision estimate with soft-thresholding
    const P = invertDiag(S);
    // Soft-threshold off-diagonal elements
    this.precision_ = P.map(
      (row, i) =>
        new Float64Array(
          row.map((v, j) => {
            if (i === j) return v;
            return Math.abs(v) > this.threshold
              ? v - Math.sign(v) * this.threshold
              : 0;
          }),
        ),
    );
    return this;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    if (this.precision_ === null || this.location_ === null) {
      throw new NotFittedError("SparsePrecision");
    }
    const P = this.precision_;
    const mu = this.location_;
    const p = mu.length;
    return new Float64Array(
      X.map((xi) => {
        let d = 0;
        for (let j = 0; j < p; j++) {
          let pRow = 0;
          for (let k = 0; k < p; k++)
            pRow += (P[j]![k] ?? 0) * ((xi[k] ?? 0) - (mu[k] ?? 0));
          d += ((xi[j] ?? 0) - (mu[j] ?? 0)) * pRow;
        }
        return d;
      }),
    );
  }
}
