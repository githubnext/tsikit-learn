/**
 * PassiveAggressiveClassifier and PassiveAggressiveRegressor.
 * Mirrors sklearn.linear_model.PassiveAggressiveClassifier/Regressor.
 */

import { NotFittedError } from "../exceptions.js";

export interface PassiveAggressiveOptions {
  C?: number;
  maxIter?: number;
  tol?: number;
  lossClassifier?: "hinge" | "squared_hinge";
  lossRegressor?: "epsilon_insensitive" | "squared_epsilon_insensitive";
  epsilon?: number;
}

export class PassiveAggressiveClassifier {
  C: number;
  maxIter: number;
  tol: number;
  loss: "hinge" | "squared_hinge";

  coef_: Float64Array | null = null;
  intercept_: Float64Array | null = null;
  classes_: Int32Array | null = null;

  constructor(options: PassiveAggressiveOptions = {}) {
    this.C = options.C ?? 1.0;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.loss = options.lossClassifier ?? "hinge";
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));
    const nFeatures = X[0]?.length ?? 0;

    // Binary or multiclass via OvR
    if (this.classes_.length === 2) {
      const posClass = this.classes_[1] ?? 1;
      const yw = new Float64Array(y.length).map((_, i) =>
        (y[i] ?? 0) === posClass ? 1 : -1,
      );
      const w = new Float64Array(nFeatures);
      const b = new Float64Array(1);
      this._trainBinary(X, yw, w, b);
      this.coef_ = w;
      this.intercept_ = b;
    } else {
      // One-vs-rest
      const coefs: Float64Array[] = [];
      const intercepts: Float64Array[] = [];
      for (let k = 0; k < this.classes_.length; k++) {
        const cls = this.classes_[k] ?? 0;
        const yw = new Float64Array(y.length).map((_, i) =>
          (y[i] ?? 0) === cls ? 1 : -1,
        );
        const w = new Float64Array(nFeatures);
        const b = new Float64Array(1);
        this._trainBinary(X, yw, w, b);
        coefs.push(w);
        intercepts.push(b);
      }
      // Flatten for storage (nClasses x nFeatures)
      const flat = new Float64Array(this.classes_.length * nFeatures);
      const flatB = new Float64Array(this.classes_.length);
      for (let k = 0; k < this.classes_.length; k++) {
        for (let j = 0; j < nFeatures; j++) flat[k * nFeatures + j] = coefs[k]![j] ?? 0;
        flatB[k] = intercepts[k]![0] ?? 0;
      }
      this.coef_ = flat;
      this.intercept_ = flatB;
    }
    return this;
  }

  private _trainBinary(
    X: Float64Array[],
    y: Float64Array,
    w: Float64Array,
    b: Float64Array,
  ): void {
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxUpdate = 0;
      for (let i = 0; i < X.length; i++) {
        const xi = X[i]!;
        const yi = y[i] ?? 0;
        let score = b[0] ?? 0;
        for (let j = 0; j < xi.length; j++) score += (w[j] ?? 0) * (xi[j] ?? 0);

        let loss: number;
        if (this.loss === "hinge") {
          loss = Math.max(0, 1 - yi * score);
        } else {
          loss = Math.max(0, 1 - yi * score) ** 2;
        }

        if (loss > 0) {
          let normSq = 1;
          for (let j = 0; j < xi.length; j++) normSq += (xi[j] ?? 0) ** 2;

          const tau =
            this.loss === "hinge"
              ? Math.min(this.C, loss / normSq)
              : Math.min(this.C, loss / (2 * normSq));

          for (let j = 0; j < xi.length; j++) {
            const upd = tau * yi * (xi[j] ?? 0);
            w[j]! += upd;
            maxUpdate = Math.max(maxUpdate, Math.abs(upd));
          }
          b[0]! += tau * yi;
        }
      }
      if (maxUpdate < this.tol) break;
    }
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.coef_ || !this.classes_) throw new NotFittedError("PassiveAggressiveClassifier");
    const nFeatures = X[0]?.length ?? 0;

    if (this.classes_.length === 2) {
      return new Int32Array(
        X.map((xi) => {
          let score = this.intercept_![0] ?? 0;
          for (let j = 0; j < nFeatures; j++) score += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
          return score >= 0 ? (this.classes_![1] ?? 1) : (this.classes_![0] ?? 0);
        }),
      );
    } else {
      const nClasses = this.classes_.length;
      return new Int32Array(
        X.map((xi) => {
          let bestScore = Number.NEGATIVE_INFINITY;
          let bestClass = 0;
          for (let k = 0; k < nClasses; k++) {
            let score = this.intercept_![k] ?? 0;
            for (let j = 0; j < nFeatures; j++)
              score += (this.coef_![k * nFeatures + j] ?? 0) * (xi[j] ?? 0);
            if (score > bestScore) {
              bestScore = score;
              bestClass = this.classes_![k] ?? 0;
            }
          }
          return bestClass;
        }),
      );
    }
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

export class PassiveAggressiveRegressor {
  C: number;
  maxIter: number;
  tol: number;
  epsilon: number;
  loss: "epsilon_insensitive" | "squared_epsilon_insensitive";

  coef_: Float64Array | null = null;
  intercept_: Float64Array | null = null;

  constructor(options: PassiveAggressiveOptions = {}) {
    this.C = options.C ?? 1.0;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.epsilon = options.epsilon ?? 0.1;
    this.loss = options.lossRegressor ?? "epsilon_insensitive";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nFeatures = X[0]?.length ?? 0;
    const w = new Float64Array(nFeatures);
    let b = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxUpdate = 0;
      for (let i = 0; i < X.length; i++) {
        const xi = X[i]!;
        const yi = y[i] ?? 0;
        let pred = b;
        for (let j = 0; j < xi.length; j++) pred += (w[j] ?? 0) * (xi[j] ?? 0);

        const residual = yi - pred;
        const absRes = Math.abs(residual);

        let loss: number;
        if (this.loss === "epsilon_insensitive") {
          loss = Math.max(0, absRes - this.epsilon);
        } else {
          loss = Math.max(0, absRes - this.epsilon) ** 2;
        }

        if (loss > 0) {
          let normSq = 1;
          for (let j = 0; j < xi.length; j++) normSq += (xi[j] ?? 0) ** 2;

          const tau =
            this.loss === "epsilon_insensitive"
              ? Math.min(this.C, loss / normSq)
              : Math.min(this.C, loss / (2 * normSq));

          const sign = residual >= 0 ? 1 : -1;
          for (let j = 0; j < xi.length; j++) {
            const upd = tau * sign * (xi[j] ?? 0);
            w[j]! += upd;
            maxUpdate = Math.max(maxUpdate, Math.abs(upd));
          }
          b += tau * sign;
        }
      }
      if (maxUpdate < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = new Float64Array([b]);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("PassiveAggressiveRegressor");
    return new Float64Array(
      X.map((xi) => {
        let pred = this.intercept_![0] ?? 0;
        for (let j = 0; j < xi.length; j++) pred += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
        return pred;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((preds[i] ?? 0) - (y[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot < 1e-10 ? 1 : 1 - ssRes / ssTot;
  }
}
