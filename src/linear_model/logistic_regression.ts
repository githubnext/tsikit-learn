/**
 * Logistic Regression classifier.
 * Mirrors sklearn.linear_model.LogisticRegression.
 */

import { NotFittedError } from "../exceptions.js";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class LogisticRegression {
  C: number;
  maxIter: number;
  tol: number;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      C?: number;
      maxIter?: number;
      tol?: number;
      fitIntercept?: boolean;
    } = {},
  ) {
    this.C = options.C ?? 1.0;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort(
      (a, b) => a - b,
    );
    this.classes_ = new Float64Array(uniqueClasses);

    // Binary logistic regression via gradient descent
    const w = new Float64Array(nFeatures);
    let b = 0;
    const lr = 0.1;
    const lambda = 1 / (this.C * n);

    // Map labels to 0/1
    const yBin = new Float64Array(n);
    const posClass = uniqueClasses[uniqueClasses.length - 1] ?? 1;
    for (let i = 0; i < n; i++) {
      yBin[i] = (y[i] ?? 0) === posClass ? 1 : 0;
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      const gradW = new Float64Array(nFeatures);
      let gradB = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(nFeatures);
        let dot = b;
        for (let j = 0; j < nFeatures; j++) {
          dot += (w[j] ?? 0) * (xi[j] ?? 0);
        }
        const p = sigmoid(dot);
        const err = p - (yBin[i] ?? 0);
        for (let j = 0; j < nFeatures; j++) {
          gradW[j] = (gradW[j] ?? 0) + err * (xi[j] ?? 0);
        }
        gradB += err;
      }

      let maxGrad = 0;
      for (let j = 0; j < nFeatures; j++) {
        const g = (gradW[j] ?? 0) / n + lambda * (w[j] ?? 0);
        w[j] = (w[j] ?? 0) - lr * g;
        if (Math.abs(g) > maxGrad) maxGrad = Math.abs(g);
      }
      if (this.fitIntercept) {
        b -= lr * (gradB / n);
      }
      if (maxGrad < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;
    return this;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (this.coef_ === null) throw new NotFittedError("LogisticRegression");
    return X.map((xi) => {
      let dot = this.intercept_;
      for (let j = 0; j < xi.length; j++) {
        dot += ((this.coef_ as Float64Array)[j] ?? 0) * (xi[j] ?? 0);
      }
      const p = sigmoid(dot);
      return new Float64Array([1 - p, p]);
    });
  }

  predict(X: Float64Array[]): Float64Array {
    const proba = this.predictProba(X);
    const classes = this.classes_ as Float64Array;
    return new Float64Array(
      proba.map((p) =>
        (p[1] ?? 0) >= 0.5 ? (classes[1] ?? 1) : (classes[0] ?? 0),
      ),
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
