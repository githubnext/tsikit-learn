/**
 * Fast coordinate descent solver for ElasticNet/Lasso.
 * Port of sklearn.linear_model._cd_fast
 */

export interface CDResult {
  weights: Float64Array;
  gap: number;
  tol: number;
  nIter: number;
}

/** Compute the soft thresholding operator */
export function softThreshold(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

/**
 * Enet coordinate descent solver.
 * Minimizes: (1/(2*n_samples)) * ||y - Xw||^2 + alpha*l1_ratio*||w||_1
 *            + (alpha*(1-l1_ratio)/2) * ||w||^2
 */
export function enEtCoordDescent(
  X: Float64Array[],
  y: Float64Array,
  alpha: number,
  rho: number,
  maxIter: number,
  tol: number,
  positive: boolean,
): CDResult {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const weights = new Float64Array(d);
  let gap = Number.POSITIVE_INFINITY;
  let nIter = 0;

  // Precompute column norms
  const colNorms = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < n; i++) colNorms[j] += (X[i]?.[j] ?? 0) ** 2;
    colNorms[j]! /= n;
  }

  // Compute residual
  const residual = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    residual[i] = y[i] ?? 0;
    for (let j = 0; j < d; j++)
      residual[i]! -= (weights[j] ?? 0) * (X[i]?.[j] ?? 0);
  }

  for (nIter = 0; nIter < maxIter; nIter++) {
    let maxChange = 0;
    for (let j = 0; j < d; j++) {
      const colNorm = colNorms[j]!;
      if (colNorm < 1e-16) continue;
      const wOld = weights[j]!;

      // Compute rho_j = (X_j^T * residual)/n + w_j * colNorm
      let rhoJ = wOld * colNorm;
      for (let i = 0; i < n; i++)
        rhoJ += ((X[i]?.[j] ?? 0) * (residual[i] ?? 0)) / n;

      // Soft threshold
      let wNew: number;
      if (positive) {
        wNew = Math.max(
          0,
          softThreshold(rhoJ, alpha * rho) / (colNorm + alpha * (1 - rho)),
        );
      } else {
        wNew = softThreshold(rhoJ, alpha * rho) / (colNorm + alpha * (1 - rho));
      }
      weights[j] = wNew;

      // Update residual
      const delta = wNew - wOld;
      if (Math.abs(delta) > 1e-16) {
        for (let i = 0; i < n; i++) residual[i]! -= delta * (X[i]?.[j] ?? 0);
        maxChange = Math.max(maxChange, Math.abs(delta) * Math.sqrt(colNorm));
      }
    }

    if (maxChange < tol) {
      gap = maxChange;
      break;
    }
  }

  // Compute final gap
  let residualNorm = 0;
  for (let i = 0; i < n; i++) residualNorm += (residual[i] ?? 0) ** 2;
  gap = residualNorm / (2 * n);
  for (let j = 0; j < d; j++) gap += alpha * rho * Math.abs(weights[j]!);
  for (let j = 0; j < d; j++)
    gap += ((alpha * (1 - rho)) / 2) * weights[j]! ** 2;

  return { weights, gap, tol, nIter };
}

/**
 * Sparse enet coordinate descent with sample weights.
 */
export function sparseCdFast(
  X: Float64Array[],
  y: Float64Array,
  sampleWeight: Float64Array,
  alpha: number,
  rho: number,
  maxIter: number,
  tol: number,
  positive: boolean,
): CDResult {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const weights = new Float64Array(d);
  let nIter = 0;

  const residual = new Float64Array(n);
  for (let i = 0; i < n; i++)
    residual[i] = (y[i] ?? 0) * (sampleWeight[i] ?? 1);

  const colNorms = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < n; i++)
      colNorms[j] += (sampleWeight[i] ?? 1) * (X[i]?.[j] ?? 0) ** 2;
  }

  for (nIter = 0; nIter < maxIter; nIter++) {
    let maxChange = 0;
    for (let j = 0; j < d; j++) {
      const colNorm = colNorms[j]!;
      if (colNorm < 1e-16) continue;
      const wOld = weights[j]!;
      let rhoJ = wOld * colNorm;
      for (let i = 0; i < n; i++) rhoJ += (X[i]?.[j] ?? 0) * (residual[i] ?? 0);
      let wNew: number;
      if (positive) {
        wNew = Math.max(
          0,
          softThreshold(rhoJ, alpha * rho) / (colNorm + alpha * (1 - rho)),
        );
      } else {
        wNew = softThreshold(rhoJ, alpha * rho) / (colNorm + alpha * (1 - rho));
      }
      weights[j] = wNew;
      const delta = wNew - wOld;
      if (Math.abs(delta) > 1e-16) {
        for (let i = 0; i < n; i++)
          residual[i]! -= delta * (sampleWeight[i] ?? 1) * (X[i]?.[j] ?? 0);
        maxChange = Math.max(maxChange, Math.abs(delta));
      }
    }
    if (maxChange < tol) break;
  }

  return { weights, gap: 0, tol, nIter };
}
