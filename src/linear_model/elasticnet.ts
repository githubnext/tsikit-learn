/**
 * ElasticNet regression combining L1 and L2 regularization.
 * Mirrors sklearn.linear_model.ElasticNet.
 */

import { NotFittedError } from "../exceptions.js";

function softThreshold(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

export class ElasticNet {
  alpha: number;
  l1Ratio: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(
    options: {
      alpha?: number;
      l1Ratio?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    this.alpha = options.alpha ?? 1.0;
    this.l1Ratio = options.l1Ratio ?? 0.5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    let yCenter = new Float64Array(n);
    let yMean = 0;
    if (this.fitIntercept) {
      for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
      yMean /= n;
      for (let i = 0; i < n; i++) yCenter[i] = (y[i] ?? 0) - yMean;
    } else {
      yCenter = y.slice();
    }

    const coef = new Float64Array(p);
    const alphaL1 = this.alpha * this.l1Ratio;
    const alphaL2 = this.alpha * (1 - this.l1Ratio);

    // Precompute column norms
    const colNorm2 = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (X[i]?.[j] ?? 0) ** 2;
      colNorm2[j] = s;
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        const oldCoefJ = coef[j] ?? 0;
        // Compute residual correlation
        let rj = 0;
        for (let i = 0; i < n; i++) {
          let pred = 0;
          for (let k = 0; k < p; k++) pred += (X[i]?.[k] ?? 0) * (coef[k] ?? 0);
          rj +=
            (X[i]?.[j] ?? 0) *
            (yCenter[i]! - pred + (X[i]?.[j] ?? 0) * oldCoefJ);
        }
        const denom = (colNorm2[j] ?? 0) + alphaL2 * n;
        coef[j] =
          denom !== 0 ? (softThreshold(rj / n, alphaL1) * n) / denom : 0;
        maxChange = Math.max(maxChange, Math.abs((coef[j] ?? 0) - oldCoefJ));
      }
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.coef_ = coef;
    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < p; j++) {
        let xMeanJ = 0;
        for (let i = 0; i < n; i++) xMeanJ += X[i]?.[j] ?? 0;
        xMeanJ /= n;
        intercept -= (coef[j] ?? 0) * xMeanJ;
      }
      this.intercept_ = intercept;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("ElasticNet is not fitted");
    const result = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let pred = this.intercept_;
      for (let j = 0; j < this.coef_.length; j++) {
        pred += (X[i]?.[j] ?? 0) * (this.coef_[j] ?? 0);
      }
      result[i] = pred;
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= y.length;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
