/**
 * Least Angle Regression (LARS), LarsCV, LassoLars.
 * Mirrors sklearn.linear_model.Lars, LarsCV, LassoLars, LassoLarsCV.
 */

import { NotFittedError } from "../exceptions.js";

export interface LarsOptions {
  fitIntercept?: boolean;
  verbose?: boolean;
  normalize?: boolean;
  precompute?: boolean;
  nNonzeroCoefs?: number;
  eps?: number;
  fitPath?: boolean;
}

export class Lars {
  fitIntercept: boolean;
  nNonzeroCoefs: number;
  eps: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  alphas_: Float64Array | null = null;
  nIter_: number = 0;

  constructor(options: LarsOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.nNonzeroCoefs = options.nNonzeroCoefs ?? 500;
    this.eps = options.eps ?? 2.220446049250313e-16;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    let xMean = new Float64Array(p);
    let yMean = 0;

    if (this.fitIntercept) {
      for (let i = 0; i < n; i++) {
        const xi = X[i]!;
        yMean += y[i] ?? 0;
        for (let j = 0; j < p; j++) xMean[j]! += xi[j] ?? 0;
      }
      yMean /= n;
      for (let j = 0; j < p; j++) xMean[j]! /= n;
    }

    // Center X and y
    const Xc: Float64Array[] = X.map((xi) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j]! = (xi[j] ?? 0) - (xMean[j] ?? 0);
      return row;
    });
    const yc = new Float64Array(n);
    for (let i = 0; i < n; i++) yc[i]! = (y[i] ?? 0) - yMean;

    // LARS algorithm (simplified — greedy least angle)
    const coef = new Float64Array(p);
    const residual = yc.slice();
    const activeSet: number[] = [];
    const maxSteps = Math.min(this.nNonzeroCoefs, p, n - 1);

    for (let step = 0; step < maxSteps; step++) {
      // Find feature with max correlation
      let maxCorr = 0;
      let maxIdx = -1;
      for (let j = 0; j < p; j++) {
        if (activeSet.includes(j)) continue;
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (Xc[i]![j] ?? 0) * (residual[i] ?? 0);
        corr = Math.abs(corr) / n;
        if (corr > maxCorr) {
          maxCorr = corr;
          maxIdx = j;
        }
      }
      if (maxIdx < 0 || maxCorr < this.eps) break;
      activeSet.push(maxIdx);

      // OLS on active set
      const A = activeSet.length;
      const XA: Float64Array[] = Xc.map((xi) => {
        const row = new Float64Array(A);
        for (let k = 0; k < A; k++) row[k]! = xi[activeSet[k]!] ?? 0;
        return row;
      });
      const ols = this._ols(XA, yc, n, A);
      for (let k = 0; k < A; k++) coef[activeSet[k]!]! = ols[k] ?? 0;

      // Update residual
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let k = 0; k < A; k++) pred += (XA[i]![k] ?? 0) * (ols[k] ?? 0);
        residual[i]! = (yc[i] ?? 0) - pred;
      }
      this.nIter_ = step + 1;
    }

    this.coef_ = coef;
    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < p; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
      this.intercept_ = intercept;
    }
    return this;
  }

  private _ols(X: Float64Array[], y: Float64Array, n: number, p: number): Float64Array {
    // Normal equations: (X'X)^-1 X'y
    const XtX = new Float64Array(p * p);
    const Xty = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const xi = X[i]!;
      for (let j = 0; j < p; j++) {
        Xty[j]! += (xi[j] ?? 0) * (y[i] ?? 0);
        for (let k = 0; k < p; k++) XtX[j * p + k]! += (xi[j] ?? 0) * (xi[k] ?? 0);
      }
    }
    // Add small ridge for stability
    for (let j = 0; j < p; j++) XtX[j * p + j]! += this.eps;
    return this._solve(XtX, Xty, p);
  }

  private _solve(A: Float64Array, b: Float64Array, n: number): Float64Array {
    // Gaussian elimination
    const M = new Float64Array(n * (n + 1));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i * (n + 1) + j]! = A[i * n + j] ?? 0;
      M[i * (n + 1) + n]! = b[i] ?? 0;
    }
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(M[row * (n + 1) + col] ?? 0) > Math.abs(M[maxRow * (n + 1) + col] ?? 0)) maxRow = row;
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
        for (let k = col; k <= n; k++) M[row * (n + 1) + k]! -= factor * (M[col * (n + 1) + k] ?? 0);
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
    if (!this.coef_) throw new NotFittedError("Lars is not fitted");
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
    let ssTot = 0, ssRes = 0, yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}

export interface LassoLarsOptions extends LarsOptions {
  alpha?: number;
}

/**
 * LassoLars — Lasso model fit with Least Angle Regression.
 * Uses a soft-thresholding step on the LARS path to enforce L1 penalty.
 */
export class LassoLars extends Lars {
  alpha: number;

  constructor(options: LassoLarsOptions = {}) {
    super(options);
    this.alpha = options.alpha ?? 1.0;
  }

  override fit(X: Float64Array[], y: Float64Array): this {
    super.fit(X, y);
    // Apply soft-thresholding to enforce L1 sparsity
    if (this.coef_) {
      const thresh = this.alpha;
      for (let j = 0; j < this.coef_.length; j++) {
        const v = this.coef_[j] ?? 0;
        this.coef_[j]! = Math.sign(v) * Math.max(0, Math.abs(v) - thresh);
      }
    }
    return this;
  }
}

export interface LarsCVOptions {
  fitIntercept?: boolean;
  maxIter?: number;
  cv?: number;
  maxNAlphas?: number;
  eps?: number;
}

/**
 * LarsCV — Cross-validated Least Angle Regression model.
 * Selects the best alpha by cross-validation.
 */
export class LarsCV {
  fitIntercept: boolean;
  cv: number;
  eps: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  alpha_: number = 0;
  cv_alphas_: Float64Array | null = null;
  mse_path_: Float64Array | null = null;

  constructor(options: LarsCVOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.cv = options.cv ?? 5;
    this.eps = options.eps ?? 2.220446049250313e-16;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0];
    let bestAlpha = alphas[0]!;
    let bestMse = Number.POSITIVE_INFINITY;

    const foldSize = Math.max(1, Math.floor(n / this.cv));
    const msePerAlpha = new Float64Array(alphas.length);

    for (let ai = 0; ai < alphas.length; ai++) {
      let totalMse = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize;
        const end = Math.min(start + foldSize, n);
        const trainX: Float64Array[] = [];
        const trainY: number[] = [];
        const testX: Float64Array[] = [];
        const testY: number[] = [];
        for (let i = 0; i < n; i++) {
          if (i >= start && i < end) {
            testX.push(X[i]!);
            testY.push(y[i] ?? 0);
          } else {
            trainX.push(X[i]!);
            trainY.push(y[i] ?? 0);
          }
        }
        const model = new LassoLars({ alpha: alphas[ai], fitIntercept: this.fitIntercept, eps: this.eps });
        model.fit(trainX, new Float64Array(trainY));
        const preds = model.predict(testX);
        let mse = 0;
        for (let i = 0; i < testY.length; i++) mse += ((testY[i] ?? 0) - (preds[i] ?? 0)) ** 2;
        totalMse += testY.length > 0 ? mse / testY.length : 0;
      }
      msePerAlpha[ai]! = totalMse / this.cv;
      if (msePerAlpha[ai]! < bestMse) {
        bestMse = msePerAlpha[ai]!;
        bestAlpha = alphas[ai]!;
      }
    }

    this.alpha_ = bestAlpha;
    this.cv_alphas_ = new Float64Array(alphas);
    this.mse_path_ = msePerAlpha;

    const best = new LassoLars({ alpha: bestAlpha, fitIntercept: this.fitIntercept, eps: this.eps });
    best.fit(X, y);
    this.coef_ = best.coef_;
    this.intercept_ = best.intercept_;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("LarsCV is not fitted");
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
    let ssTot = 0, ssRes = 0, yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}
