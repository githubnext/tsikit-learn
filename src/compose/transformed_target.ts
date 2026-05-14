/**
 * TransformedTargetRegressor.
 * Mirrors sklearn.compose.TransformedTargetRegressor.
 */

import { NotFittedError } from "../exceptions.js";

export interface TransformableTarget {
  fit(y: Float64Array): this;
  transform(y: Float64Array): Float64Array;
  inverseTransform(y: Float64Array): Float64Array;
}

export interface FittableRegressor {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
}

export interface TransformedTargetRegressorOptions {
  regressor?: FittableRegressor;
  transformer?: TransformableTarget;
  func?: (y: Float64Array) => Float64Array;
  inverseFunc?: (y: Float64Array) => Float64Array;
  checkInverse?: boolean;
}

export class TransformedTargetRegressor {
  regressor_: FittableRegressor | null = null;
  transformer_: TransformableTarget | null = null;
  func: ((y: Float64Array) => Float64Array) | null;
  inverseFunc: ((y: Float64Array) => Float64Array) | null;

  private regressorOpt: FittableRegressor | null;
  private transformerOpt: TransformableTarget | null;

  constructor(opts: TransformedTargetRegressorOptions = {}) {
    this.regressorOpt = opts.regressor ?? null;
    this.transformerOpt = opts.transformer ?? null;
    this.func = opts.func ?? null;
    this.inverseFunc = opts.inverseFunc ?? null;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    let yTrans: Float64Array;

    if (this.func) {
      yTrans = this.func(y);
    } else if (this.transformerOpt) {
      this.transformer_ = this.transformerOpt;
      this.transformer_.fit(y);
      yTrans = this.transformer_.transform(y);
    } else {
      // Default: identity
      yTrans = Float64Array.from(y);
    }

    const reg = this.regressorOpt ?? createDefaultRegressor();
    this.regressor_ = reg;
    reg.fit(X, yTrans);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.regressor_) throw new NotFittedError("TransformedTargetRegressor");
    const predsTrans = this.regressor_.predict(X);

    if (this.inverseFunc) {
      return this.inverseFunc(predsTrans);
    } else if (this.transformer_) {
      return this.transformer_.inverseTransform(predsTrans);
    }
    return predsTrans;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}

function createDefaultRegressor(): FittableRegressor {
  let coef: Float64Array | null = null;
  let intercept = 0;
  return {
    fit(X: Float64Array[], y: Float64Array) {
      const n = X.length;
      const d = X[0]?.length ?? 0;
      coef = new Float64Array(d);
      const lr = 0.01;
      for (let iter = 0; iter < 200; iter++) {
        for (let i = 0; i < n; i++) {
          const xi = X[i] as Float64Array;
          let pred = intercept;
          for (let j = 0; j < d; j++) pred += (coef![j] ?? 0) * (xi[j] ?? 0);
          const err = (y[i] ?? 0) - pred;
          intercept += lr * err;
          for (let j = 0; j < d; j++) coef![j]! += lr * err * (xi[j] ?? 0);
        }
      }
      return this;
    },
    predict(X: Float64Array[]) {
      return Float64Array.from(X, (xi) => {
        let pred = intercept;
        for (let j = 0; j < xi.length; j++) pred += (coef![j] ?? 0) * (xi[j] ?? 0);
        return pred;
      });
    },
  };
}
