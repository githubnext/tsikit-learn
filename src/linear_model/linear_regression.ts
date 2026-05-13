/**
 * Linear Regression — Ordinary Least Squares.
 * Mirrors sklearn.linear_model.LinearRegression.
 *
 * Uses the normal equations: β = (X.T X)⁻¹ X.T y
 * Solved via Cholesky decomposition for numerical stability.
 */

import { BaseEstimator, RegressorMixin } from "../base.js";
import {
  addDiagonal,
  choleskyLinsolve,
  gramMatrix,
  safeDot,
  xtDotY,
} from "../utils/extmath.js";
import { checkArray, checkXy } from "../utils/validation.js";

export interface LinearRegressionParams {
  fit_intercept?: boolean;
  copy_X?: boolean;
  positive?: boolean;
}

/**
 * Ordinary least squares Linear Regression.
 *
 * Minimizes the residual sum of squares between observed and predicted values.
 * Equivalent to sklearn.linear_model.LinearRegression.
 *
 * @example
 * ```ts
 * import { LinearRegression } from 'tsikit-learn';
 *
 * const X = [new Float64Array([1]), new Float64Array([2]), new Float64Array([3])];
 * const y = new Float64Array([2, 4, 6]);
 *
 * const reg = new LinearRegression();
 * reg.fit(X, y);
 * console.log(reg.coef_);    // Float64Array [2]
 * console.log(reg.intercept_); // ~0
 * console.log(reg.predict([new Float64Array([4])])); // Float64Array [8]
 * ```
 */
export class LinearRegression extends BaseEstimator {
  fit_intercept: boolean;
  copy_X: boolean;
  positive: boolean;

  coef_?: Float64Array;
  intercept_?: number;
  n_features_in_?: number;
  rank_?: number;

  constructor(params: LinearRegressionParams = {}) {
    super();
    this.fit_intercept = params.fit_intercept ?? true;
    this.copy_X = params.copy_X ?? true;
    this.positive = params.positive ?? false;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    checkXy(X, y);
    checkArray(X);

    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    this.n_features_in_ = nFeatures;

    let XCenter = X;
    let yCenter = y;
    let xMean: Float64Array | undefined;
    let yMean = 0;

    if (this.fit_intercept) {
      // Center X and y
      xMean = new Float64Array(nFeatures);
      for (let i = 0; i < n; i++) {
        const row = X[i] ?? new Float64Array(nFeatures);
        for (let j = 0; j < nFeatures; j++) {
          xMean[j] = (xMean[j] ?? 0) + (row[j] ?? 0);
        }
      }
      for (let j = 0; j < nFeatures; j++) {
        xMean[j] = (xMean[j] ?? 0) / n;
      }
      yMean = 0;
      for (const v of y) yMean += v;
      yMean /= n;

      XCenter = X.map((row) => {
        const centered = new Float64Array(row);
        for (let j = 0; j < centered.length; j++) {
          centered[j] = (centered[j] ?? 0) - ((xMean as Float64Array)[j] ?? 0);
        }
        return centered;
      });
      yCenter = new Float64Array(y.length);
      for (let i = 0; i < y.length; i++) {
        yCenter[i] = (y[i] ?? 0) - yMean;
      }
    }

    // Solve normal equations: (X.T @ X) @ β = X.T @ y
    const XtX = gramMatrix(XCenter);
    const Xty = xtDotY(XCenter, yCenter);

    // Add tiny ridge to handle near-singular matrices
    addDiagonal(XtX, 1e-12);

    const coef = choleskyLinsolve(XtX, Xty);
    this.coef_ = coef;
    this.rank_ = nFeatures;

    if (this.fit_intercept && xMean !== undefined) {
      let intercept = yMean;
      for (let j = 0; j < nFeatures; j++) {
        intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
      }
      this.intercept_ = intercept;
    } else {
      this.intercept_ = 0;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    this._check_is_fitted(["coef_", "intercept_"]);
    const coef = this.coef_ as Float64Array;
    const intercept = this.intercept_ as number;
    const yPred = safeDot(X, coef);
    for (let i = 0; i < yPred.length; i++) {
      yPred[i] = (yPred[i] ?? 0) + intercept;
    }
    return yPred;
  }

  /** R² score on test data. */
  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = y[i] ?? 0;
      ssTot += (yi - yMean) ** 2;
      ssRes += (yi - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
