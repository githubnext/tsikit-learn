/**
 * Multi-task linear models: MultiTaskLasso, MultiTaskElasticNet.
 * Mirrors sklearn.linear_model.MultiTaskLasso and MultiTaskElasticNet.
 */

import { NotFittedError } from "../exceptions.js";

function softThreshold(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

/** Block soft-threshold for a row of coefficients (L2,1 group norm penalty). */
function blockSoftThreshold(row: Float64Array, threshold: number): void {
  let norm = 0;
  for (let j = 0; j < row.length; j++) norm += (row[j] ?? 0) ** 2;
  norm = Math.sqrt(norm);
  if (norm <= threshold) {
    for (let j = 0; j < row.length; j++) row[j] = 0;
  } else {
    const scale = 1 - threshold / norm;
    for (let j = 0; j < row.length; j++) row[j] = (row[j] ?? 0) * scale;
  }
}

export interface MultiTaskOptions {
  alpha?: number;
  l1Ratio?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tol?: number;
}

/**
 * Multi-task Lasso with L2,1 norm penalty (joint feature selection across tasks).
 * Mirrors sklearn.linear_model.MultiTaskLasso.
 */
export class MultiTaskLasso {
  alpha: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  nIter_: number = 0;

  constructor(options: MultiTaskOptions = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const t = (Y[0] ?? new Float64Array(0)).length;

    let Xc = X;
    let Yc = Y;
    let xMeans = new Float64Array(p);
    let yMeans = new Float64Array(t);

    if (this.fitIntercept) {
      xMeans = new Float64Array(p);
      yMeans = new Float64Array(t);
      for (const xi of X) for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0);
      for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) / n;
      for (const yi of Y) for (let k = 0; k < t; k++) yMeans[k] = (yMeans[k] ?? 0) + (yi[k] ?? 0);
      for (let k = 0; k < t; k++) yMeans[k] = (yMeans[k] ?? 0) / n;
      Xc = X.map((xi) => { const r = new Float64Array(p); for (let j = 0; j < p; j++) r[j] = (xi[j] ?? 0) - (xMeans[j] ?? 0); return r; });
      Yc = Y.map((yi) => { const r = new Float64Array(t); for (let k = 0; k < t; k++) r[k] = (yi[k] ?? 0) - (yMeans[k] ?? 0); return r; });
    }

    // Initialize coefficients: p x t matrix stored as rows (p rows of length t)
    const coef: Float64Array[] = [];
    for (let j = 0; j < p; j++) coef.push(new Float64Array(t));

    // Precompute X'X diagonal and X'Y
    const xColNormSq = new Float64Array(p);
    for (const xi of Xc) for (let j = 0; j < p; j++) xColNormSq[j] = (xColNormSq[j] ?? 0) + (xi[j] ?? 0) ** 2;

    const xtY: Float64Array[] = [];
    for (let j = 0; j < p; j++) {
      const v = new Float64Array(t);
      for (let i = 0; i < n; i++) for (let k = 0; k < t; k++) v[k] = (v[k] ?? 0) + ((Xc[i] ?? new Float64Array(0))[j] ?? 0) * ((Yc[i] ?? new Float64Array(0))[k] ?? 0);
      xtY.push(v);
    }

    // Block coordinate descent
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const colNorm = xColNormSq[j] ?? 0;
        if (colNorm === 0) continue;

        // Compute residual correlation for feature j
        const rho = new Float64Array(t);
        for (let k = 0; k < t; k++) rho[k] = (xtY[j] ?? new Float64Array(0))[k] ?? 0;
        for (let j2 = 0; j2 < p; j2++) {
          if (j2 === j) continue;
          for (let i = 0; i < n; i++) {
            const xij2 = ((Xc[i] ?? new Float64Array(0))[j] ?? 0) * ((Xc[i] ?? new Float64Array(0))[j2] ?? 0);
            for (let k = 0; k < t; k++) rho[k] = (rho[k] ?? 0) - xij2 * ((coef[j2] ?? new Float64Array(0))[k] ?? 0);
          }
        }
        for (let k = 0; k < t; k++) rho[k] = (rho[k] ?? 0) / colNorm;

        const oldRow = new Float64Array(coef[j] ?? new Float64Array(t));
        blockSoftThreshold(rho, (this.alpha * n) / colNorm);
        const newRow = coef[j]!;
        for (let k = 0; k < t; k++) newRow[k] = rho[k] ?? 0;

        for (let k = 0; k < t; k++) {
          const d = Math.abs((newRow[k] ?? 0) - (oldRow[k] ?? 0));
          if (d > maxDelta) maxDelta = d;
        }
      }
      this.nIter_ = iter + 1;
      if (maxDelta < this.tol) break;
    }

    // coef_ stored as t x p (tasks x features), matching sklearn convention
    this.coef_ = [];
    for (let k = 0; k < t; k++) {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j] = (coef[j] ?? new Float64Array(0))[k] ?? 0;
      this.coef_.push(row);
    }

    if (this.fitIntercept) {
      this.intercept_ = new Float64Array(t);
      for (let k = 0; k < t; k++) {
        let s = yMeans[k] ?? 0;
        for (let j = 0; j < p; j++) s -= ((this.coef_[k] ?? new Float64Array(0))[j] ?? 0) * (xMeans[j] ?? 0);
        this.intercept_[k] = s;
      }
    } else {
      this.intercept_ = new Float64Array(t);
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.coef_) throw new NotFittedError("MultiTaskLasso is not fitted yet.");
    const t = this.coef_.length;
    return X.map((xi) => {
      const pred = new Float64Array(t);
      for (let k = 0; k < t; k++) {
        let s = this.intercept_![k] ?? 0;
        for (let j = 0; j < xi.length; j++) s += ((this.coef_![k] ?? new Float64Array(0))[j] ?? 0) * (xi[j] ?? 0);
        pred[k] = s;
      }
      return pred;
    });
  }
}

/**
 * Multi-task ElasticNet with L1/L2 mixed penalty and L2,1 group sparsity.
 * Mirrors sklearn.linear_model.MultiTaskElasticNet.
 */
export class MultiTaskElasticNet {
  alpha: number;
  l1Ratio: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  nIter_: number = 0;

  constructor(options: MultiTaskOptions = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.l1Ratio = options.l1Ratio ?? 0.5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const t = (Y[0] ?? new Float64Array(0)).length;
    const l1 = this.alpha * this.l1Ratio;
    const l2 = this.alpha * (1 - this.l1Ratio);

    let Xc = X;
    let Yc = Y;
    let xMeans = new Float64Array(p);
    let yMeans = new Float64Array(t);

    if (this.fitIntercept) {
      for (const xi of X) for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0);
      for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) / n;
      for (const yi of Y) for (let k = 0; k < t; k++) yMeans[k] = (yMeans[k] ?? 0) + (yi[k] ?? 0);
      for (let k = 0; k < t; k++) yMeans[k] = (yMeans[k] ?? 0) / n;
      Xc = X.map((xi) => { const r = new Float64Array(p); for (let j = 0; j < p; j++) r[j] = (xi[j] ?? 0) - (xMeans[j] ?? 0); return r; });
      Yc = Y.map((yi) => { const r = new Float64Array(t); for (let k = 0; k < t; k++) r[k] = (yi[k] ?? 0) - (yMeans[k] ?? 0); return r; });
    }

    const coef: Float64Array[] = [];
    for (let j = 0; j < p; j++) coef.push(new Float64Array(t));

    const xColNormSq = new Float64Array(p);
    for (const xi of Xc) for (let j = 0; j < p; j++) xColNormSq[j] = (xColNormSq[j] ?? 0) + (xi[j] ?? 0) ** 2;

    const xtY: Float64Array[] = [];
    for (let j = 0; j < p; j++) {
      const v = new Float64Array(t);
      for (let i = 0; i < n; i++) for (let k = 0; k < t; k++) v[k] = (v[k] ?? 0) + ((Xc[i] ?? new Float64Array(0))[j] ?? 0) * ((Yc[i] ?? new Float64Array(0))[k] ?? 0);
      xtY.push(v);
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const colNorm = (xColNormSq[j] ?? 0) + l2 * n;
        if (colNorm === 0) continue;

        const rho = new Float64Array(t);
        for (let k = 0; k < t; k++) rho[k] = (xtY[j] ?? new Float64Array(0))[k] ?? 0;
        for (let j2 = 0; j2 < p; j2++) {
          if (j2 === j) continue;
          for (let i = 0; i < n; i++) {
            const xij2 = ((Xc[i] ?? new Float64Array(0))[j] ?? 0) * ((Xc[i] ?? new Float64Array(0))[j2] ?? 0);
            for (let k = 0; k < t; k++) rho[k] = (rho[k] ?? 0) - xij2 * ((coef[j2] ?? new Float64Array(0))[k] ?? 0);
          }
        }
        for (let k = 0; k < t; k++) rho[k] = (rho[k] ?? 0) / colNorm;

        const oldRow = new Float64Array(coef[j] ?? new Float64Array(t));
        blockSoftThreshold(rho, (l1 * n) / colNorm);
        const newRow = coef[j]!;
        for (let k = 0; k < t; k++) newRow[k] = rho[k] ?? 0;

        for (let k = 0; k < t; k++) {
          const d = Math.abs((newRow[k] ?? 0) - (oldRow[k] ?? 0));
          if (d > maxDelta) maxDelta = d;
        }
      }
      this.nIter_ = iter + 1;
      if (maxDelta < this.tol) break;
    }

    this.coef_ = [];
    for (let k = 0; k < t; k++) {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j] = (coef[j] ?? new Float64Array(0))[k] ?? 0;
      this.coef_.push(row);
    }

    if (this.fitIntercept) {
      this.intercept_ = new Float64Array(t);
      for (let k = 0; k < t; k++) {
        let s = yMeans[k] ?? 0;
        for (let j = 0; j < p; j++) s -= ((this.coef_[k] ?? new Float64Array(0))[j] ?? 0) * (xMeans[j] ?? 0);
        this.intercept_[k] = s;
      }
    } else {
      this.intercept_ = new Float64Array(t);
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.coef_) throw new NotFittedError("MultiTaskElasticNet is not fitted yet.");
    const t = this.coef_.length;
    return X.map((xi) => {
      const pred = new Float64Array(t);
      for (let k = 0; k < t; k++) {
        let s = this.intercept_![k] ?? 0;
        for (let j = 0; j < xi.length; j++) s += ((this.coef_![k] ?? new Float64Array(0))[j] ?? 0) * (xi[j] ?? 0);
        pred[k] = s;
      }
      return pred;
    });
  }
}
