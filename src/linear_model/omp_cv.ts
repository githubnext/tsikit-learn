/**
 * OrthogonalMatchingPursuitCV — cross-validated OMP.
 * Mirrors sklearn.linear_model.OrthogonalMatchingPursuitCV.
 */

import { BaseEstimator } from "../base.js";
import { NotFittedError } from "../exceptions.js";

export interface OMPCVParams {
  copy?: boolean;
  fitIntercept?: boolean;
  normalize?: boolean;
  maxIter?: number | null;
  cv?: number;
  nJobs?: number | null;
  verbose?: boolean;
}

/** Dot product. */
function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

/** OMP: Orthogonal matching pursuit — returns coef for a given n_nonzero. */
function ompFit(
  X: Float64Array[],
  y: Float64Array,
  nNonzero: number,
  fitIntercept: boolean,
): { coef: Float64Array; intercept: number } {
  const n = X.length;
  const p = X[0]?.length ?? 0;

  let Xc = X;
  let yc = y;
  const xMean = new Float64Array(p);
  let yMean = 0;

  if (fitIntercept) {
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) xMean[j]! += X[i]?.[j] ?? 0;
      xMean[j]! /= n;
    }
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;

    Xc = X.map((row) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = (row[j] ?? 0) - (xMean[j] ?? 0);
      return r;
    });
    yc = new Float64Array(n);
    for (let i = 0; i < n; i++) yc[i] = (y[i] ?? 0) - yMean;
  }

  const residual = new Float64Array(yc);
  const coef = new Float64Array(p);
  const activeSet: number[] = [];
  const activeX: Float64Array[] = [];

  for (let iter = 0; iter < Math.min(nNonzero, p); iter++) {
    // Find atom most correlated with residual
    let bestCorr = -1;
    let bestIdx = 0;
    for (let j = 0; j < p; j++) {
      if (activeSet.includes(j)) continue;
      const xj = new Float64Array(n);
      for (let i = 0; i < n; i++) xj[i] = Xc[i]?.[j] ?? 0;
      const corr = Math.abs(dot(xj, residual));
      if (corr > bestCorr) {
        bestCorr = corr;
        bestIdx = j;
      }
    }
    activeSet.push(bestIdx);
    const xj = new Float64Array(n);
    for (let i = 0; i < n; i++) xj[i] = Xc[i]?.[bestIdx] ?? 0;
    activeX.push(xj);

    // Least squares on active set (normal equations)
    const k = activeX.length;
    const G = Array.from({ length: k }, () => new Float64Array(k));
    const h = new Float64Array(k);
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++)
        G[a]![b] = dot(
          activeX[a] ?? new Float64Array(n),
          activeX[b] ?? new Float64Array(n),
        );
      h[a] = dot(activeX[a] ?? new Float64Array(n), yc);
    }

    // Solve G*alpha = h via Gaussian elimination
    const A = G.map((row) => new Float64Array(row));
    const bv = new Float64Array(h);
    for (let col = 0; col < k; col++) {
      let pivot = col;
      for (let row = col + 1; row < k; row++) {
        if (Math.abs(A[row]![col] ?? 0) > Math.abs(A[pivot]![col] ?? 0))
          pivot = row;
      }
      const tmp = A[col];
      A[col] = A[pivot]!;
      A[pivot] = tmp!;
      const tb = bv[col] ?? 0;
      bv[col] = bv[pivot] ?? 0;
      bv[pivot] = tb;
      const diag = A[col]![col] ?? 0;
      if (Math.abs(diag) < 1e-14) continue;
      for (let row = 0; row < k; row++) {
        if (row === col) continue;
        const factor = (A[row]![col] ?? 0) / diag;
        for (let kk = 0; kk < k; kk++)
          A[row]![kk]! -= factor * (A[col]![kk] ?? 0);
        bv[row]! -= factor * (bv[col] ?? 0);
      }
    }
    const alpha = new Float64Array(k);
    for (let i = 0; i < k; i++)
      alpha[i] = (A[i]![i] ?? 0) !== 0 ? (bv[i] ?? 0) / (A[i]![i] ?? 1) : 0;

    // Update coef and residual
    coef.fill(0);
    for (let a = 0; a < k; a++) coef[activeSet[a] ?? 0] = alpha[a] ?? 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < p; j++) pred += (coef[j] ?? 0) * (Xc[i]?.[j] ?? 0);
      residual[i] = (yc[i] ?? 0) - pred;
    }
  }

  let intercept = 0;
  if (fitIntercept) {
    intercept = yMean;
    for (let j = 0; j < p; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
  }
  return { coef, intercept };
}

/**
 * OrthogonalMatchingPursuitCV — cross-validated OMP.
 *
 * Selects the number of non-zero coefficients via cross-validation.
 * Mirrors sklearn.linear_model.OrthogonalMatchingPursuitCV.
 */
export class OrthogonalMatchingPursuitCV extends BaseEstimator {
  readonly copy: boolean;
  readonly fitIntercept: boolean;
  readonly maxIter: number | null;
  readonly cv: number;

  coef_: Float64Array | null = null;
  intercept_: number | null = null;
  nNonzeroCoefs_: number | null = null;
  nFeaturesIn_: number | null = null;

  constructor(params: OMPCVParams = {}) {
    super();
    this.copy = params.copy ?? true;
    this.fitIntercept = params.fitIntercept ?? true;
    this.maxIter = params.maxIter ?? null;
    this.cv = params.cv ?? 5;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const maxN = Math.min(this.maxIter ?? p, p, n - 1);

    // CV fold split
    const foldSize = Math.floor(n / this.cv);
    let bestNnz = 1;
    let bestCvScore = Number.POSITIVE_INFINITY;

    for (let nnz = 1; nnz <= maxN; nnz++) {
      let mse = 0;
      let foldCount = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize;
        const end = fold === this.cv - 1 ? n : start + foldSize;
        const trainX: Float64Array[] = [];
        const trainY: number[] = [];
        const valX: Float64Array[] = [];
        const valY: number[] = [];
        for (let i = 0; i < n; i++) {
          if (i >= start && i < end) {
            valX.push(X[i] ?? new Float64Array(p));
            valY.push(y[i] ?? 0);
          } else {
            trainX.push(X[i] ?? new Float64Array(p));
            trainY.push(y[i] ?? 0);
          }
        }
        if (trainX.length < nnz + 1) continue;
        const { coef, intercept } = ompFit(
          trainX,
          new Float64Array(trainY),
          nnz,
          this.fitIntercept,
        );
        for (let i = 0; i < valX.length; i++) {
          let pred = intercept;
          const xi = valX[i] ?? new Float64Array(p);
          for (let j = 0; j < p; j++) pred += (coef[j] ?? 0) * (xi[j] ?? 0);
          mse += ((valY[i] ?? 0) - pred) ** 2;
        }
        foldCount += valX.length;
      }
      const avgMse = foldCount > 0 ? mse / foldCount : Number.POSITIVE_INFINITY;
      if (avgMse < bestCvScore) {
        bestCvScore = avgMse;
        bestNnz = nnz;
      }
    }

    this.nNonzeroCoefs_ = bestNnz;
    const { coef, intercept } = ompFit(X, y, bestNnz, this.fitIntercept);
    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null || this.intercept_ === null)
      throw new NotFittedError("OrthogonalMatchingPursuitCV");
    const coef = this.coef_;
    const intercept = this.intercept_;
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let s = intercept;
      const xi = X[i] ?? new Float64Array(0);
      for (let j = 0; j < coef.length; j++) s += (coef[j] ?? 0) * (xi[j] ?? 0);
      out[i] = s;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += y[i] ?? 0;
    mean /= n;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < n; i++) {
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
