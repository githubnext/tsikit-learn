/**
 * Perceptron classifier.
 * Mirrors sklearn.linear_model.Perceptron.
 */

import { NotFittedError } from "../exceptions.js";

export class Perceptron {
  alpha: number;
  maxIter: number;
  tol: number;
  fitIntercept: boolean;
  eta0: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      alpha?: number;
      maxIter?: number;
      tol?: number;
      fitIntercept?: boolean;
      eta0?: number;
    } = {},
  ) {
    this.alpha = options.alpha ?? 1e-4;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.fitIntercept = options.fitIntercept ?? true;
    this.eta0 = options.eta0 ?? 1.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.classes_ = new Float64Array(
      Array.from(new Set(Array.from(y))).sort((a, b) => a - b),
    );

    const w = new Float64Array(p);
    let b = 0;
    const posClass = this.classes_[this.classes_.length - 1] ?? 1;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let errors = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let dot = b;
        for (let j = 0; j < p; j++) {
          dot += (w[j] ?? 0) * (xi[j] ?? 0);
        }
        const yBin = (y[i] ?? 0) === posClass ? 1 : -1;
        const pred = dot >= 0 ? 1 : -1;
        if (pred !== yBin) {
          errors++;
          for (let j = 0; j < p; j++) {
            w[j] = (w[j] ?? 0) + this.eta0 * yBin * (xi[j] ?? 0);
          }
          if (this.fitIntercept) {
            b += this.eta0 * yBin;
          }
        }
      }
      if (errors === 0) break;
    }

    this.coef_ = w;
    this.intercept_ = b;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("Perceptron");
    const classes = this.classes_ as Float64Array;
    const coef = this.coef_;
    return new Float64Array(
      X.map((xi) => {
        let dot = this.intercept_;
        for (let j = 0; j < xi.length; j++) {
          dot += (coef[j] ?? 0) * (xi[j] ?? 0);
        }
        return dot >= 0
          ? (classes[classes.length - 1] ?? 1)
          : (classes[0] ?? 0);
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
