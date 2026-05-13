/**
 * Ridge Regression — L2-regularized Linear Regression.
 * Mirrors sklearn.linear_model.Ridge.
 *
 * Minimizes: ||y - Xw||² + alpha * ||w||²
 * Solved as: β = (X.T X + alpha * I)⁻¹ X.T y
 */

import { BaseEstimator } from "../base.js";
import {
  addDiagonal,
  choleskyLinsolve,
  gramMatrix,
  safeDot,
  xtDotY,
} from "../utils/extmath.js";
import { checkArray, checkXy } from "../utils/validation.js";

export interface RidgeParams {
  alpha?: number;
  fit_intercept?: boolean;
  copy_X?: boolean;
  max_iter?: number;
  tol?: number;
  solver?: "auto" | "cholesky";
}

/**
 * Linear least squares with L2 regularization.
 *
 * Equivalent to sklearn.linear_model.Ridge.
 *
 * @example
 * ```ts
 * import { Ridge } from 'tsikit-learn';
 *
 * const X = [new Float64Array([1, 0]), new Float64Array([0, 1]), new Float64Array([1, 1])];
 * const y = new Float64Array([1, 2, 3]);
 *
 * const reg = new Ridge({ alpha: 1.0 });
 * reg.fit(X, y);
 * console.log(reg.coef_);
 * ```
 */
export class Ridge extends BaseEstimator {
  alpha: number;
  fit_intercept: boolean;
  copy_X: boolean;
  max_iter: number;
  tol: number;
  solver: "auto" | "cholesky";

  coef_?: Float64Array;
  intercept_?: number;
  n_features_in_?: number;
  n_iter_?: number;

  constructor(params: RidgeParams = {}) {
    super();
    this.alpha = params.alpha ?? 1.0;
    this.fit_intercept = params.fit_intercept ?? true;
    this.copy_X = params.copy_X ?? true;
    this.max_iter = params.max_iter ?? 1000;
    this.tol = params.tol ?? 1e-4;
    this.solver = params.solver ?? "auto";
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

    // Solve (X.T @ X + alpha * I) @ β = X.T @ y
    const XtX = gramMatrix(XCenter);
    const Xty = xtDotY(XCenter, yCenter);

    // Add alpha * I (ridge regularization)
    addDiagonal(XtX, this.alpha);

    const coef = choleskyLinsolve(XtX, Xty);
    this.coef_ = coef;
    this.n_iter_ = 1;

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
