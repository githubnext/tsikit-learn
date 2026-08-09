/**
 * Minibatch coordinate descent utilities for linear models.
 * Mirrors sklearn.linear_model._cd_fast internals:
 * scalar soft-thresholding, coordinate descent update, ISTA solver,
 * ElasticNet alpha path computation.
 * Note: enetPath (vector form) is in lasso_path.ts,
 *       softThreshold (vector form) is in stochastic_gradient.ts.
 */

/**
 * Scalar soft-thresholding: sign(x) * max(|x| - threshold, 0).
 * Core operation in LASSO/ElasticNet coordinate descent.
 * Mirrors sklearn.linear_model._cd_fast.soft_thresholding (scalar form).
 */
export function softThresholdScalar(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

/**
 * One coordinate descent update for ElasticNet.
 * Updates coef[j] in place.
 * rho_j = <X_j, r + X_j * coef_j> / n (partial correlation)
 */
export function coordinateDescentUpdate(
  j: number,
  X: Float64Array[],
  residual: Float64Array,
  coef: Float64Array,
  alpha: number,
  l1Ratio: number,
  norm2: Float64Array,
): number {
  const n = X.length;
  const normj = norm2[j] ?? 1;
  if (normj === 0) return 0;

  const oldCoefJ = coef[j] ?? 0;
  let rho = 0;
  for (let i = 0; i < n; i++) {
    rho += (residual[i] ?? 0) * (X[i]![j] ?? 0);
  }
  rho = rho / n + oldCoefJ * normj;

  const l1Penalty = alpha * l1Ratio;
  const l2Penalty = alpha * (1 - l1Ratio);

  const newCoefJ = softThresholdScalar(rho, l1Penalty) / (normj + l2Penalty);
  coef[j] = newCoefJ;

  return newCoefJ - oldCoefJ;
}

/**
 * Compute feature-wise squared norms for X (columns).
 */
export function computeFeatureNorms(X: Float64Array[]): Float64Array {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const norms = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      norms[j]! += (X[i]![j] ?? 0) ** 2 / n;
    }
  }
  return norms;
}

export interface CdEnetPathResult {
  alphas: Float64Array;
  coefs: Float64Array[];
  dualGaps: Float64Array;
  nIters: Int32Array;
}

/**
 * Compute the ElasticNet regularization path via coordinate descent.
 * Mirrors sklearn.linear_model._cd_fast enet path internals.
 * (Distinct from enetPath in lasso_path.ts which uses a different interface.)
 */
export function cdEnetPath(
  X: Float64Array[],
  y: Float64Array,
  options: {
    l1Ratio?: number;
    eps?: number;
    nAlphas?: number;
    alphas?: Float64Array;
    maxIter?: number;
    tol?: number;
  } = {},
): CdEnetPathResult {
  const l1Ratio = options.l1Ratio ?? 0.5;
  const eps = options.eps ?? 1e-3;
  const nAlphas = options.nAlphas ?? 100;
  const maxIter = options.maxIter ?? 1000;
  const tol = options.tol ?? 1e-4;

  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;

  let alphaMax = 0;
  for (let j = 0; j < p; j++) {
    let rho = 0;
    for (let i = 0; i < n; i++) rho += (X[i]![j] ?? 0) * (y[i] ?? 0);
    alphaMax = Math.max(alphaMax, Math.abs(rho / n));
  }
  if (l1Ratio > 0) alphaMax /= l1Ratio;

  const alphas =
    options.alphas ??
    new Float64Array(
      Array.from(
        { length: nAlphas },
        (_, k) => alphaMax * eps ** (k / (nAlphas - 1)),
      ),
    );

  const coefs: Float64Array[] = [];
  const dualGaps = new Float64Array(alphas.length);
  const nIters = new Int32Array(alphas.length);
  const norm2 = computeFeatureNorms(X);

  const coef = new Float64Array(p);
  const residual = Float64Array.from(y);

  for (let aIdx = 0; aIdx < alphas.length; aIdx++) {
    const alpha = alphas[aIdx] ?? 0;

    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        const delta = coordinateDescentUpdate(
          j,
          X,
          residual,
          coef,
          alpha,
          l1Ratio,
          norm2,
        );
        if (Math.abs(delta) > maxChange) maxChange = Math.abs(delta);
        if (delta !== 0) {
          for (let i = 0; i < n; i++) {
            residual[i]! -= delta * (X[i]![j] ?? 0);
          }
        }
      }
      nIters[aIdx] = iter + 1;
      if (maxChange < tol) break;
    }

    let dualGap = 0;
    for (let i = 0; i < n; i++) dualGap += (residual[i] ?? 0) ** 2;
    dualGaps[aIdx] = dualGap / (2 * n);

    coefs.push(Float64Array.from(coef));
  }

  return { alphas, coefs, dualGaps, nIters };
}

/**
 * ISTA (Iterative Shrinkage Thresholding Algorithm) solver for Lasso.
 * Mirrors sklearn's proximal gradient approach for sparse signal recovery.
 */
export function istaLasso(
  X: Float64Array[],
  y: Float64Array,
  alpha: number,
  options: { maxIter?: number; tol?: number } = {},
): Float64Array {
  const maxIter = options.maxIter ?? 1000;
  const tol = options.tol ?? 1e-4;
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;

  // Estimate Lipschitz constant L via power iteration
  let v = new Float64Array(p).fill(1);
  for (let it = 0; it < 20; it++) {
    const Xv = new Float64Array(n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < p; j++) Xv[i]! += (X[i]![j] ?? 0) * (v[j] ?? 0);
    const Av = new Float64Array(p);
    for (let j = 0; j < p; j++)
      for (let i = 0; i < n; i++) Av[j]! += (X[i]![j] ?? 0) * (Xv[i] ?? 0);
    let norm = 0;
    for (const x of Av) norm += x * x;
    norm = Math.sqrt(norm) || 1;
    v = Av.map((x) => x / norm);
  }
  let L = 0;
  const Xv2 = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < p; j++) Xv2[i]! += (X[i]![j] ?? 0) * (v[j] ?? 0);
  for (const x of Xv2) L += x * x;
  L = L / n || 1;
  const lr = 1 / L;

  const coef = new Float64Array(p);

  for (let iter = 0; iter < maxIter; iter++) {
    const Xw = new Float64Array(n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < p; j++) Xw[i]! += (X[i]![j] ?? 0) * (coef[j] ?? 0);

    const grad = new Float64Array(p);
    for (let j = 0; j < p; j++)
      for (let i = 0; i < n; i++)
        grad[j]! += (X[i]![j] ?? 0) * ((Xw[i] ?? 0) - (y[i] ?? 0));
    for (let j = 0; j < p; j++) grad[j]! /= n;

    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      const z = (coef[j] ?? 0) - lr * (grad[j] ?? 0);
      const newJ = softThresholdScalar(z, lr * alpha);
      maxChange = Math.max(maxChange, Math.abs(newJ - (coef[j] ?? 0)));
      coef[j] = newJ;
    }

    if (maxChange < tol) break;
  }
  return coef;
}
