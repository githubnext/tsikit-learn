/**
 * TheilSenRegressor and RANSACRegressor.
 * Mirrors sklearn.linear_model.TheilSenRegressor and RANSACRegressor.
 */

import { NotFittedError } from "../exceptions.js";

export interface TheilSenRegressorOptions {
  fitIntercept?: boolean;
  maxSubpopulation?: number;
  nSubsamples?: number | null;
  maxIter?: number;
  tol?: number;
  randomState?: number;
}

/**
 * TheilSenRegressor — median-of-slopes robust linear regression.
 */
export class TheilSenRegressor {
  fitIntercept: boolean;
  maxSubpopulation: number;
  nSubsamples: number | null;
  maxIter: number;
  tol: number;
  randomState: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  breakdown_: number = 0;
  nIter_: number = 0;
  nSubsamples_: number = 0;

  constructor(options: TheilSenRegressorOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxSubpopulation = options.maxSubpopulation ?? 10000;
    this.nSubsamples = options.nSubsamples ?? null;
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-3;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const nSub = this.nSubsamples ?? Math.min(n, p + 1, this.maxSubpopulation);
    this.nSubsamples_ = nSub;

    // Simple implementation: take nSub pairs and compute median slopes
    const coef = new Float64Array(p);
    const slopes: Float64Array[] = [];

    // Use a simple LCG for reproducible subsampling
    let rng = this.randomState;
    const nextRng = () => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return rng / 4294967296;
    };

    const nPairs = Math.min((nSub * (nSub - 1)) / 2, this.maxSubpopulation);
    const indices: number[] = Array.from({ length: n }, (_, i) => i);

    for (let t = 0; t < nPairs; t++) {
      const i1 = Math.floor(nextRng() * n);
      let i2 = Math.floor(nextRng() * n);
      while (i2 === i1) i2 = Math.floor(nextRng() * n);

      const xi = X[indices[i1]!]!;
      const xj = X[indices[i2]!]!;
      const yi = y[indices[i1]!] ?? 0;
      const yj = y[indices[i2]!] ?? 0;

      const slope = new Float64Array(p);
      let denom = 0;
      for (let j = 0; j < p; j++) {
        const dx = (xj[j] ?? 0) - (xi[j] ?? 0);
        denom += dx * dx;
      }
      if (denom < 1e-12) continue;
      const dy = yj - yi;
      for (let j = 0; j < p; j++) {
        slope[j]! = (dy * ((xj[j] ?? 0) - (xi[j] ?? 0))) / denom;
      }
      slopes.push(slope);
    }

    // Median of slopes
    if (slopes.length === 0) {
      this.coef_ = new Float64Array(p);
      this.intercept_ = 0;
      return this;
    }

    for (let j = 0; j < p; j++) {
      const vals = slopes.map((s) => s[j] ?? 0).sort((a, b) => a - b);
      const mid = Math.floor(vals.length / 2);
      coef[j]! =
        vals.length % 2 === 0
          ? ((vals[mid - 1] ?? 0) + (vals[mid] ?? 0)) / 2
          : (vals[mid] ?? 0);
    }

    this.coef_ = coef;

    if (this.fitIntercept) {
      // Median of residuals
      const residuals: number[] = [];
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = 0; j < p; j++) dot += (coef[j] ?? 0) * (X[i]![j] ?? 0);
        residuals.push((y[i] ?? 0) - dot);
      }
      residuals.sort((a, b) => a - b);
      const mid = Math.floor(residuals.length / 2);
      this.intercept_ =
        residuals.length % 2 === 0
          ? ((residuals[mid - 1] ?? 0) + (residuals[mid] ?? 0)) / 2
          : (residuals[mid] ?? 0);
    }

    this.breakdown_ = 0.5;
    this.nIter_ = slopes.length;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_)
      throw new NotFittedError("TheilSenRegressor is not fitted");
    const n = X.length;
    const p = this.coef_.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let pred = this.intercept_;
      const xi = X[i]!;
      for (let j = 0; j < p; j++) pred += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      out[i]! = pred;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}

export interface RANSACRegressorOptions {
  minSamples?: number | null;
  residualThreshold?: number | null;
  maxTrials?: number;
  maxSkips?: number;
  stopNInliers?: number;
  stopScore?: number;
  stopProbability?: number;
  randomState?: number;
}

/**
 * RANSACRegressor — Random Sample Consensus robust regression.
 */
export class RANSACRegressor {
  minSamples: number | null;
  residualThreshold: number | null;
  maxTrials: number;
  maxSkips: number;
  stopNInliers: number;
  stopScore: number;
  stopProbability: number;
  randomState: number;

  estimator_coef_: Float64Array | null = null;
  estimator_intercept_: number = 0;
  inlierMask_: Uint8Array | null = null;
  nTrials_: number = 0;
  nSkips_: number = 0;

  constructor(options: RANSACRegressorOptions = {}) {
    this.minSamples = options.minSamples ?? null;
    this.residualThreshold = options.residualThreshold ?? null;
    this.maxTrials = options.maxTrials ?? 100;
    this.maxSkips = options.maxSkips ?? Number.MAX_SAFE_INTEGER;
    this.stopNInliers = options.stopNInliers ?? Number.MAX_SAFE_INTEGER;
    this.stopScore = options.stopScore ?? Number.POSITIVE_INFINITY;
    this.stopProbability = options.stopProbability ?? 0.99;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const minSamp = this.minSamples ?? Math.max(2, p + 1);

    // Estimate residual threshold from MAD if not provided
    const residThresh = this.residualThreshold ?? this._mad(y) * 1.4826;

    let rng = this.randomState;
    const nextRng = () => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return rng / 4294967296;
    };

    let bestScore = -1;
    let bestInliers: Uint8Array = new Uint8Array(n);
    let bestCoef: Float64Array = new Float64Array(p);
    let bestIntercept = 0;

    for (let trial = 0; trial < this.maxTrials; trial++) {
      // Random sample
      const sample: number[] = [];
      const pool = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < minSamp; i++) {
        const idx = Math.floor(nextRng() * (pool.length - i));
        const tmp = pool[pool.length - 1 - i]!;
        pool[pool.length - 1 - i]! = pool[idx];
        pool[idx]! = tmp;
        sample.push(pool[pool.length - 1 - i]!);
      }

      const Xs = sample.map((i) => X[i]!);
      const ys = new Float64Array(sample.map((i) => y[i] ?? 0));

      // Fit OLS on sample
      const { coef, intercept } = this._ols(Xs, ys, p);

      // Count inliers
      const inliers = new Uint8Array(n);
      let nInliers = 0;
      for (let i = 0; i < n; i++) {
        let pred = intercept;
        const xi = X[i]!;
        for (let j = 0; j < p; j++) pred += (coef[j] ?? 0) * (xi[j] ?? 0);
        if (Math.abs((y[i] ?? 0) - pred) <= residThresh) {
          inliers[i]! = 1;
          nInliers++;
        }
      }

      if (nInliers > bestScore) {
        bestScore = nInliers;
        bestInliers = inliers;
        bestCoef = coef;
        bestIntercept = intercept;
      }

      this.nTrials_ = trial + 1;
      if (nInliers >= this.stopNInliers) break;
    }

    // Refit on all inliers
    const inlierX = X.filter((_, i) => bestInliers[i] === 1);
    const inlierY = new Float64Array(
      Array.from({ length: n }, (_, i) => i)
        .filter((i) => bestInliers[i] === 1)
        .map((i) => y[i] ?? 0),
    );

    if (inlierX.length > p) {
      const { coef, intercept } = this._ols(inlierX, inlierY, p);
      this.estimator_coef_ = coef;
      this.estimator_intercept_ = intercept;
    } else {
      this.estimator_coef_ = bestCoef;
      this.estimator_intercept_ = bestIntercept;
    }

    this.inlierMask_ = bestInliers;
    return this;
  }

  private _mad(y: Float64Array): number {
    const sorted = Array.from(y).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median =
      sorted.length % 2 === 0
        ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
        : (sorted[mid] ?? 0);
    const devs = sorted.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
    return devs.length % 2 === 0
      ? ((devs[mid - 1] ?? 0) + (devs[mid] ?? 0)) / 2
      : (devs[mid] ?? 0);
  }

  private _ols(
    X: Float64Array[],
    y: Float64Array,
    p: number,
  ): { coef: Float64Array; intercept: number } {
    const n = X.length;
    let yMean = 0;
    const xMean = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      yMean += y[i] ?? 0;
      for (let j = 0; j < p; j++) xMean[j]! += X[i]![j] ?? 0;
    }
    yMean /= n;
    for (let j = 0; j < p; j++) xMean[j]! /= n;

    const XtX = new Float64Array(p * p);
    const Xty = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const xi = X[i]!;
      const yi = (y[i] ?? 0) - yMean;
      for (let j = 0; j < p; j++) {
        const xij = (xi[j] ?? 0) - (xMean[j] ?? 0);
        Xty[j]! += xij * yi;
        for (let k = 0; k < p; k++)
          XtX[j * p + k]! += xij * ((xi[k] ?? 0) - (xMean[k] ?? 0));
      }
    }
    for (let j = 0; j < p; j++) XtX[j * p + j]! += 1e-10;

    const coef = this._solveLinear(XtX, Xty, p);
    let intercept = yMean;
    for (let j = 0; j < p; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
    return { coef, intercept };
  }

  private _solveLinear(
    A: Float64Array,
    b: Float64Array,
    n: number,
  ): Float64Array {
    const M = new Float64Array(n * (n + 1));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i * (n + 1) + j]! = A[i * n + j] ?? 0;
      M[i * (n + 1) + n]! = b[i] ?? 0;
    }
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (
          Math.abs(M[row * (n + 1) + col] ?? 0) >
          Math.abs(M[maxRow * (n + 1) + col] ?? 0)
        )
          maxRow = row;
      }
      for (let k = col; k <= n; k++) {
        const tmp = M[col * (n + 1) + k] ?? 0;
        M[col * (n + 1) + k]! = M[maxRow * (n + 1) + k] ?? 0;
        M[maxRow * (n + 1) + k]! = tmp;
      }
      const pivot = M[col * (n + 1) + col] ?? 0;
      if (Math.abs(pivot) < 1e-12) continue;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = (M[row * (n + 1) + col] ?? 0) / pivot;
        for (let k = col; k <= n; k++)
          M[row * (n + 1) + k]! -= factor * (M[col * (n + 1) + k] ?? 0);
      }
    }
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const pivot = M[i * (n + 1) + i] ?? 0;
      if (Math.abs(pivot) > 1e-12) x[i]! = (M[i * (n + 1) + n] ?? 0) / pivot;
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimator_coef_)
      throw new NotFittedError("RANSACRegressor is not fitted");
    const n = X.length;
    const p = this.estimator_coef_.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let pred = this.estimator_intercept_;
      const xi = X[i]!;
      for (let j = 0; j < p; j++)
        pred += (this.estimator_coef_[j] ?? 0) * (xi[j] ?? 0);
      out[i]! = pred;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}
