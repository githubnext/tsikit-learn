/**
 * SGD Classifier and Regressor.
 * Mirrors sklearn.linear_model.SGDClassifier / SGDRegressor.
 */

import { NotFittedError } from "../exceptions.js";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class SGDClassifier {
  loss: string;
  alpha: number;
  maxIter: number;
  tol: number;
  eta0: number;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      loss?: string;
      alpha?: number;
      maxIter?: number;
      tol?: number;
      eta0?: number;
      fitIntercept?: boolean;
    } = {},
  ) {
    this.loss = options.loss ?? "hinge";
    this.alpha = options.alpha ?? 1e-4;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.eta0 = options.eta0 ?? 0.01;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.classes_ = new Float64Array(
      Array.from(new Set(Array.from(y))).sort((a, b) => a - b),
    );

    const w = new Float64Array(p);
    let b = 0;
    const posClass = (this.classes_[this.classes_.length - 1]) ?? 1;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let dot = b;
        for (let j = 0; j < p; j++) {
          dot += (w[j] ?? 0) * (xi[j] ?? 0);
        }
        const yLabel = (y[i] ?? 0) === posClass ? 1 : -1;

        let grad = 0;
        if (this.loss === "hinge") {
          const margin = yLabel * dot;
          if (margin < 1) {
            grad = -yLabel;
            totalLoss += 1 - margin;
          }
        } else {
          // log loss
          const p2 = sigmoid(yLabel * dot);
          grad = -(1 - p2) * yLabel;
          totalLoss += -Math.log(p2 + 1e-15);
        }

        for (let j = 0; j < p; j++) {
          w[j] = (w[j] ?? 0) * (1 - this.eta0 * this.alpha) - this.eta0 * grad * (xi[j] ?? 0);
        }
        if (this.fitIntercept) {
          b -= this.eta0 * grad;
        }
      }
      if (totalLoss / n < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("SGDClassifier");
    const classes = this.classes_ as Float64Array;
    const coef = this.coef_;
    return new Float64Array(
      X.map((xi) => {
        let dot = this.intercept_;
        for (let j = 0; j < xi.length; j++) {
          dot += (coef[j] ?? 0) * (xi[j] ?? 0);
        }
        return dot >= 0 ? (classes[classes.length - 1] ?? 1) : (classes[0] ?? 0);
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class SGDRegressor {
  alpha: number;
  maxIter: number;
  tol: number;
  eta0: number;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;

  constructor(
    options: {
      alpha?: number;
      maxIter?: number;
      tol?: number;
      eta0?: number;
      fitIntercept?: boolean;
    } = {},
  ) {
    this.alpha = options.alpha ?? 1e-4;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.eta0 = options.eta0 ?? 0.01;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const w = new Float64Array(p);
    let b = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let pred = b;
        for (let j = 0; j < p; j++) {
          pred += (w[j] ?? 0) * (xi[j] ?? 0);
        }
        const err = pred - (y[i] ?? 0);
        totalLoss += err ** 2;
        for (let j = 0; j < p; j++) {
          w[j] = (w[j] ?? 0) * (1 - this.eta0 * this.alpha) - this.eta0 * err * (xi[j] ?? 0);
        }
        if (this.fitIntercept) {
          b -= this.eta0 * err;
        }
      }
      if (totalLoss / n < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("SGDRegressor");
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
