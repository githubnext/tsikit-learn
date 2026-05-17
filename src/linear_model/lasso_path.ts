/**
 * Coordinate Descent path algorithms for Lasso/ElasticNet.
 * Mirrors sklearn.linear_model: lasso_path, enet_path, LassoPath.
 */

import { NotFittedError } from "../exceptions.js";
import { checkXy } from "../utils/validation.js";

export interface PathResult {
  alphas: Float64Array;
  coefs: Float64Array[];
  dualGaps: Float64Array;
  nIters: number[];
}

/**
 * Coordinate descent soft-threshold step.
 */
function softThreshold(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

/**
 * Compute Lasso path via coordinate descent (warm-start over alpha grid).
 *
 * @param X - Training data [n x p].
 * @param y - Target vector [n].
 * @param alphas - Decreasing sequence of alpha values.
 * @param eps - Ratio of smallest to largest alpha (used to generate alphas if not provided).
 * @param nAlphas - Number of alphas on the path.
 * @param maxIter - Max iterations per alpha.
 * @param tol - Convergence tolerance.
 * @param l1Ratio - ElasticNet mixing: 1.0 = Lasso, 0 = Ridge.
 */
export function lassoPath(
  X: Float64Array[],
  y: Float64Array,
  alphas?: Float64Array,
  eps = 1e-3,
  nAlphas = 100,
  maxIter = 1000,
  tol = 1e-4,
  l1Ratio = 1.0,
): PathResult {
  checkXy(X, y);
  const n = X.length;
  const p = X[0]!.length;

  // Precompute column norms squared
  const colNormSq = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) colNormSq[j]! += (X[i]![j]! ?? 0) ** 2;
  }

  // Center X and y (no intercept in path algorithm)
  const xMean = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) xMean[j]! += X[i]![j]! ?? 0;
  for (let j = 0; j < p; j++) xMean[j]! /= n;
  const Xc = X.map((row) => Float64Array.from(row, (v, j) => v - (xMean[j]! ?? 0)));

  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += y[i]! ?? 0;
  yMean /= n;
  const yc = Float64Array.from(y, (v) => v - yMean);

  // Auto-generate alphas if not provided
  if (!alphas) {
    let alphaMax = 0;
    for (let j = 0; j < p; j++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += (Xc[i]![j]! ?? 0) * (yc[i]! ?? 0);
      alphaMax = Math.max(alphaMax, Math.abs(dot) / n);
    }
    alphaMax /= l1Ratio;
    const alphaMin = alphaMax * eps;
    const logMax = Math.log(alphaMax);
    const logMin = Math.log(alphaMin);
    alphas = Float64Array.from(
      { length: nAlphas },
      (_, i) => Math.exp(logMax + (i / (nAlphas - 1)) * (logMin - logMax)),
    );
  }

  const coefs: Float64Array[] = [];
  const dualGaps = new Float64Array(alphas.length);
  const nIters: number[] = [];

  // Warm start
  let coef = new Float64Array(p);

  for (let ai = 0; ai < alphas.length; ai++) {
    const alpha = alphas[ai]! ?? 1e-3;
    const l1Pen = alpha * l1Ratio;
    const l2Pen = alpha * (1 - l1Ratio);

    let iter = 0;
    for (; iter < maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        const norm2 = (colNormSq[j]! ?? 1) / n;
        if (norm2 < 1e-14) continue;

        // Compute partial residual wrt j
        let rj = 0;
        for (let i = 0; i < n; i++) {
          let pred = 0;
          for (let k = 0; k < p; k++) if (k !== j) pred += (Xc[i]![k]! ?? 0) * (coef[k]! ?? 0);
          rj += (Xc[i]![j]! ?? 0) * ((yc[i]! ?? 0) - pred);
        }
        rj /= n;

        const oldCoef = coef[j]! ?? 0;
        const newCoef = softThreshold(rj, l1Pen) / (norm2 + l2Pen);
        coef[j]! = newCoef;
        maxChange = Math.max(maxChange, Math.abs(newCoef - oldCoef));
      }
      if (maxChange < tol) break;
    }
    nIters.push(iter);

    // Dual gap
    let yPred = Float64Array.from({ length: n }, () => 0);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < p; j++) s += (Xc[i]![j]! ?? 0) * (coef[j]! ?? 0);
      yPred[i]! = s;
    }
    let rNorm = 0;
    for (let i = 0; i < n; i++) rNorm += ((yc[i]! ?? 0) - (yPred[i]! ?? 0)) ** 2;
    dualGaps[ai]! = rNorm / n;

    coefs.push(new Float64Array(coef));
  }

  return { alphas, coefs, dualGaps, nIters };
}

/**
 * ElasticNet path — alias for lassoPath with l1Ratio < 1.
 */
export function enetPath(
  X: Float64Array[],
  y: Float64Array,
  l1Ratio = 0.5,
  alphas?: Float64Array,
  eps = 1e-3,
  nAlphas = 100,
  maxIter = 1000,
  tol = 1e-4,
): PathResult {
  return lassoPath(X, y, alphas, eps, nAlphas, maxIter, tol, l1Ratio);
}

export interface LassoPathOptions {
  eps?: number;
  nAlphas?: number;
  alphas?: Float64Array;
  maxIter?: number;
  tol?: number;
  l1Ratio?: number;
  fit_intercept?: boolean;
}

/**
 * Lasso path estimator — wraps `lassoPath` as an sklearn-style class.
 */
export class LassoPath {
  eps: number;
  nAlphas: number;
  alphas_param: Float64Array | undefined;
  maxIter: number;
  tol: number;
  l1Ratio: number;
  fit_intercept: boolean;

  alphas_: Float64Array | null = null;
  coefs_: Float64Array[] | null = null;
  dualGaps_: Float64Array | null = null;
  nIters_: number[] | null = null;

  constructor(options: LassoPathOptions = {}) {
    this.eps = options.eps ?? 1e-3;
    this.nAlphas = options.nAlphas ?? 100;
    this.alphas_param = options.alphas;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
    this.l1Ratio = options.l1Ratio ?? 1.0;
    this.fit_intercept = options.fit_intercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const result = lassoPath(
      X,
      y,
      this.alphas_param,
      this.eps,
      this.nAlphas,
      this.maxIter,
      this.tol,
      this.l1Ratio,
    );
    this.alphas_ = result.alphas;
    this.coefs_ = result.coefs;
    this.dualGaps_ = result.dualGaps;
    this.nIters_ = result.nIters;
    return this;
  }

  predict(X: Float64Array[], alphaIdx?: number): Float64Array {
    if (!this.coefs_ || !this.alphas_) throw new NotFittedError("LassoPath is not fitted");
    const idx = alphaIdx ?? this.coefs_.length - 1;
    const coef = this.coefs_[idx]!;
    return Float64Array.from(X, (row) => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += (row[j]! ?? 0) * (coef[j]! ?? 0);
      return s;
    });
  }
}
