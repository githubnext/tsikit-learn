/**
 * Weighted Least Squares (WLS) and Generalized Least Squares (GLS)
 * Ported from sklearn/statsmodels linear regression utilities
 */

export interface WLSOptions {
  fitIntercept?: boolean;
  copyX?: boolean;
  positiveCoef?: boolean;
}

/**
 * Weighted Ordinary Least Squares regression.
 * Minimizes the weighted sum of squared residuals:
 *   argmin_w sum_i w_i * (y_i - X_i @ w)^2
 */
export class WeightedLeastSquares {
  fitIntercept: boolean;

  private coef_: Float64Array | null = null;
  private intercept_: number = 0;
  private nFeatures_: number = 0;

  constructor(options: WLSOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array, sampleWeight?: Float64Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const p = this.fitIntercept ? nFeatures + 1 : nFeatures;
    this.nFeatures_ = nFeatures;

    // Build augmented design matrix with weights
    // XtWX @ coef = XtWy  (normal equations)
    const XtWX = new Float64Array(p * p);
    const XtWy = new Float64Array(p);

    for (let i = 0; i < nSamples; i++) {
      const xi = X[i]!;
      const yi = y[i] ?? 0;
      const wi = sampleWeight ? (sampleWeight[i] ?? 1) : 1;

      // Augmented row: [x0, x1, ..., 1] if fitIntercept
      const row = new Float64Array(p);
      for (let j = 0; j < nFeatures; j++) row[j] = xi[j] ?? 0;
      if (this.fitIntercept) row[nFeatures] = 1;

      for (let j = 0; j < p; j++) {
        XtWy[j]! += wi * (row[j] ?? 0) * yi;
        for (let k = 0; k < p; k++) {
          XtWX[j * p + k]! += wi * (row[j] ?? 0) * (row[k] ?? 0);
        }
      }
    }

    // Solve XtWX @ coef = XtWy using Cholesky decomposition
    const coef = this._solveNormalEq(XtWX, XtWy, p);

    if (this.fitIntercept) {
      this.coef_ = coef.slice(0, nFeatures);
      this.intercept_ = coef[nFeatures] ?? 0;
    } else {
      this.coef_ = coef;
      this.intercept_ = 0;
    }
    return this;
  }

  private _solveNormalEq(
    A: Float64Array,
    b: Float64Array,
    p: number,
  ): Float64Array {
    // Cholesky-based solver (in-place)
    const L = new Float64Array(p * p);
    for (let j = 0; j < p; j++) {
      let s = A[j * p + j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[j * p + k] ?? 0) ** 2;
      if (s <= 0) {
        // Regularize with a small diagonal addition (ridge)
        s = 1e-10;
      }
      L[j * p + j] = Math.sqrt(s);
      for (let i = j + 1; i < p; i++) {
        let t = A[i * p + j] ?? 0;
        for (let k = 0; k < j; k++)
          t -= (L[i * p + k] ?? 0) * (L[j * p + k] ?? 0);
        L[i * p + j] = t / (L[j * p + j] ?? 1);
      }
    }

    // Forward substitution
    const x = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      let s = b[i] ?? 0;
      for (let k = 0; k < i; k++) s -= (L[i * p + k] ?? 0) * (x[k] ?? 0);
      x[i] = s / (L[i * p + i] ?? 1);
    }

    // Back substitution
    const coef = new Float64Array(p);
    for (let i = p - 1; i >= 0; i--) {
      let s = x[i] ?? 0;
      for (let k = i + 1; k < p; k++) s -= (L[k * p + i] ?? 0) * (coef[k] ?? 0);
      coef[i] = s / (L[i * p + i] ?? 1);
    }

    return coef;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const result = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let dot = this.intercept_;
      const xi = X[i]!;
      for (let j = 0; j < this.nFeatures_; j++) {
        dot += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      }
      result[i] = dot;
    }
    return result;
  }

  score(
    X: Float64Array[],
    y: Float64Array,
    sampleWeight?: Float64Array,
  ): number {
    const yPred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let wSum = 0;
    let wMean = 0;

    for (let i = 0; i < y.length; i++) {
      const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
      wMean += w * (y[i] ?? 0);
      wSum += w;
    }
    wMean /= wSum;

    for (let i = 0; i < y.length; i++) {
      const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
      const d = (y[i] ?? 0) - wMean;
      ssTot += w * d * d;
      const r = (y[i] ?? 0) - (yPred[i] ?? 0);
      ssRes += w * r * r;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }

  get coef(): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    return this.coef_;
  }

  get intercept(): number {
    return this.intercept_;
  }
}

export interface GLSOptions {
  fitIntercept?: boolean;
  maxIter?: number;
  tol?: number;
}

/**
 * Simplified Generalized Least Squares.
 * Assumes the covariance structure Omega is diagonal (heteroscedastic errors).
 * For general GLS, the user provides per-sample variances.
 */
export class GeneralizedLeastSquares {
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  private coef_: Float64Array | null = null;
  private intercept_: number = 0;
  private nFeatures_: number = 0;

  constructor(options: GLSOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-6;
  }

  /**
   * Fit using Feasible GLS (FGLS):
   * 1. OLS to get initial residuals
   * 2. Estimate variance from residuals
   * 3. Re-weight and fit WLS
   * Iterate until convergence.
   */
  fit(X: Float64Array[], y: Float64Array): this {
    const nSamples = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;

    // Initialize with uniform weights
    const weights = new Float64Array(nSamples).fill(1);
    let prevCoef: Float64Array | null = null;

    const wls = new WeightedLeastSquares({ fitIntercept: this.fitIntercept });

    for (let iter = 0; iter < this.maxIter; iter++) {
      wls.fit(X, y, weights);
      const yPred = wls.predict(X);

      // Estimate variance for each sample (using squared residuals)
      const variances = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        const r = (y[i] ?? 0) - (yPred[i] ?? 0);
        variances[i] = r * r + 1e-10;
      }

      // New weights = 1 / variance
      for (let i = 0; i < nSamples; i++) {
        weights[i] = 1 / (variances[i] ?? 1);
      }

      // Check convergence
      if (prevCoef !== null) {
        let maxDiff = 0;
        const coef = wls.coef;
        for (let j = 0; j < coef.length; j++) {
          maxDiff = Math.max(
            maxDiff,
            Math.abs((coef[j] ?? 0) - (prevCoef[j] ?? 0)),
          );
        }
        if (maxDiff < this.tol) break;
      }

      prevCoef = wls.coef.slice();
    }

    this.coef_ = wls.coef;
    this.intercept_ = wls.intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const result = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let dot = this.intercept_;
      const xi = X[i]!;
      for (let j = 0; j < this.nFeatures_; j++) {
        dot += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      }
      result[i] = dot;
    }
    return result;
  }

  get coef(): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    return this.coef_;
  }

  get intercept(): number {
    return this.intercept_;
  }
}

/**
 * Compute the Durbin-Watson statistic for residuals.
 * Tests for first-order serial correlation in regression residuals.
 * Values close to 2 indicate no autocorrelation.
 */
export function durbinWatson(residuals: Float64Array): number {
  let num = 0;
  let denom = 0;
  for (let i = 0; i < residuals.length; i++) {
    const r = residuals[i] ?? 0;
    denom += r * r;
    if (i > 0) {
      const diff = r - (residuals[i - 1] ?? 0);
      num += diff * diff;
    }
  }
  return denom === 0 ? 2 : num / denom;
}

/**
 * Compute the Breusch-Pagan test statistic for heteroscedasticity.
 * @param residuals Residuals from a regression
 * @param X Design matrix (used to test if residuals depend on X)
 * @returns Test statistic (chi-squared distributed under null)
 */
export function breuschPaganTest(
  residuals: Float64Array,
  X: Float64Array[],
): number {
  const n = residuals.length;
  // Squared residuals
  const sqResid = new Float64Array(n);
  let meanSqResid = 0;
  for (let i = 0; i < n; i++) {
    const r = residuals[i] ?? 0;
    sqResid[i] = r * r;
    meanSqResid += r * r;
  }
  meanSqResid /= n;

  // Regress squared residuals on X (using simple correlation)
  const nFeatures = X[0]?.length ?? 0;
  let r2 = 0;
  for (let j = 0; j < nFeatures; j++) {
    let covXY = 0;
    let varX = 0;
    let meanX = 0;
    for (let i = 0; i < n; i++) meanX += X[i]![j] ?? 0;
    meanX /= n;
    for (let i = 0; i < n; i++) {
      const xDev = (X[i]![j] ?? 0) - meanX;
      const yDev = (sqResid[i] ?? 0) - meanSqResid;
      covXY += xDev * yDev;
      varX += xDev * xDev;
    }
    if (varX > 0) {
      const corr = covXY / Math.sqrt(varX);
      r2 += (corr * corr) / (n * meanSqResid * meanSqResid || 1);
    }
  }

  // LM statistic = n * R²
  return n * r2;
}
