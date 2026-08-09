/**
 * Lasso and ElasticNet regression via coordinate descent.
 * Mirrors sklearn.linear_model.Lasso and ElasticNet.
 */

import { NotFittedError } from "../exceptions.js";

function softThreshold(x: number, threshold: number): number {
  if (x > threshold) return x - threshold;
  if (x < -threshold) return x + threshold;
  return 0;
}

export class Lasso {
  alpha: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;

  constructor(
    options: {
      alpha?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    this.alpha = options.alpha ?? 1.0;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const w = new Float64Array(p);
    let intercept = 0;

    // Center data if fitIntercept
    const yMean = this.fitIntercept
      ? Array.from(y).reduce((a, b) => a + b, 0) / n
      : 0;

    // Coordinate descent
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        // Compute partial residual
        let rho = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i] ?? new Float64Array(p);
          let pred = intercept;
          for (let k = 0; k < p; k++) {
            if (k !== j) pred += (w[k] ?? 0) * (xi[k] ?? 0);
          }
          rho += (xi[j] ?? 0) * ((y[i] ?? 0) - yMean - pred);
        }
        rho /= n;
        const normSq =
          Array.from(X).reduce((s, xi) => s + (xi[j] ?? 0) ** 2, 0) / n;
        const wOld = w[j] ?? 0;
        w[j] = normSq > 0 ? softThreshold(rho, this.alpha) / normSq : 0;
        const delta = Math.abs((w[j] ?? 0) - wOld);
        if (delta > maxDelta) maxDelta = delta;
      }
      if (this.fitIntercept) {
        let predSum = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i] ?? new Float64Array(p);
          let pred = 0;
          for (let j = 0; j < p; j++) {
            pred += (w[j] ?? 0) * (xi[j] ?? 0);
          }
          predSum += (y[i] ?? 0) - pred;
        }
        intercept = predSum / n;
      }
      if (maxDelta < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("Lasso");
    const coef = this.coef_;
    return new Float64Array(
      X.map((xi) => {
        let pred = this.intercept_;
        for (let j = 0; j < xi.length; j++) {
          pred += (coef[j] ?? 0) * (xi[j] ?? 0);
        }
        return pred;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}

export class ElasticNet extends Lasso {
  l1Ratio: number;

  constructor(
    options: {
      alpha?: number;
      l1Ratio?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    super(options);
    this.l1Ratio = options.l1Ratio ?? 0.5;
  }

  override fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const w = new Float64Array(p);
    let intercept = 0;
    const l1 = this.alpha * this.l1Ratio;
    const l2 = this.alpha * (1 - this.l1Ratio);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i] ?? new Float64Array(p);
          let pred = intercept;
          for (let k = 0; k < p; k++) {
            if (k !== j) pred += (w[k] ?? 0) * (xi[k] ?? 0);
          }
          rho += (xi[j] ?? 0) * ((y[i] ?? 0) - pred);
        }
        rho /= n;
        const normSq =
          Array.from(X).reduce((s, xi) => s + (xi[j] ?? 0) ** 2, 0) / n + l2;
        const wOld = w[j] ?? 0;
        w[j] = normSq > 0 ? softThreshold(rho, l1) / normSq : 0;
        const delta = Math.abs((w[j] ?? 0) - wOld);
        if (delta > maxDelta) maxDelta = delta;
      }
      if (this.fitIntercept) {
        let predSum = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i] ?? new Float64Array(p);
          let pred = 0;
          for (let j = 0; j < p; j++) {
            pred += (w[j] ?? 0) * (xi[j] ?? 0);
          }
          predSum += (y[i] ?? 0) - pred;
        }
        intercept = predSum / n;
      }
      if (maxDelta < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = intercept;
    return this;
  }
}
