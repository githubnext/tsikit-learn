/**
 * MultiOutputClassifier and MultiOutputRegressor.
 * Mirrors sklearn.multioutput.
 */

import { NotFittedError } from "../exceptions.js";

export interface MultiOutputClassifierOptions {
  estimator: {
    fit(X: Float64Array[], y: Int32Array): unknown;
    predict(X: Float64Array[]): Int32Array;
    score?(X: Float64Array[], y: Int32Array): number;
  };
  nJobs?: number;
}

export class MultiOutputClassifier {
  estimator: MultiOutputClassifierOptions["estimator"];
  estimators_: MultiOutputClassifierOptions["estimator"][] | null = null;

  constructor(options: MultiOutputClassifierOptions) {
    this.estimator = options.estimator;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const nOutputs = Y.length;
    this.estimators_ = [];
    for (let k = 0; k < nOutputs; k++) {
      // Clone estimator by using Object.create - simple approach
      const est = Object.create(
        Object.getPrototypeOf(this.estimator) as object,
      ) as typeof this.estimator;
      Object.assign(est, JSON.parse(JSON.stringify(this.estimator)));
      est.fit(X, Y[k] as Int32Array);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (!this.estimators_)
      throw new NotFittedError("MultiOutputClassifier is not fitted.");
    return this.estimators_.map((est) => est.predict(X));
  }

  score(X: Float64Array[], Y: Int32Array[]): number {
    const preds = this.predict(X);
    let totalScore = 0;
    const n = (Y[0] ?? new Int32Array(0)).length;
    for (let k = 0; k < Y.length; k++) {
      const yk = Y[k] as Int32Array;
      const pk = preds[k] as Int32Array;
      let correct = 0;
      for (let i = 0; i < n; i++) if ((yk[i] ?? 0) === (pk[i] ?? 0)) correct++;
      totalScore += correct / n;
    }
    return totalScore / Y.length;
  }
}

export interface MultiOutputRegressorOptions {
  estimator: {
    fit(X: Float64Array[], y: Float64Array): unknown;
    predict(X: Float64Array[]): Float64Array;
    score?(X: Float64Array[], y: Float64Array): number;
  };
  nJobs?: number;
}

export class MultiOutputRegressor {
  estimator: MultiOutputRegressorOptions["estimator"];
  estimators_: MultiOutputRegressorOptions["estimator"][] | null = null;

  constructor(options: MultiOutputRegressorOptions) {
    this.estimator = options.estimator;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const nOutputs = Y.length;
    this.estimators_ = [];
    for (let k = 0; k < nOutputs; k++) {
      const est = Object.create(
        Object.getPrototypeOf(this.estimator) as object,
      ) as typeof this.estimator;
      Object.assign(est, JSON.parse(JSON.stringify(this.estimator)));
      est.fit(X, Y[k] as Float64Array);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.estimators_)
      throw new NotFittedError("MultiOutputRegressor is not fitted.");
    return this.estimators_.map((est) => est.predict(X));
  }

  score(X: Float64Array[], Y: Float64Array[]): number {
    const preds = this.predict(X);
    let totalScore = 0;
    for (let k = 0; k < Y.length; k++) {
      const yk = Y[k] as Float64Array;
      const pk = preds[k] as Float64Array;
      const n = yk.length;
      let ssRes = 0;
      let ssTot = 0;
      let mean = 0;
      for (let i = 0; i < n; i++) mean += yk[i] ?? 0;
      mean /= n;
      for (let i = 0; i < n; i++) {
        ssRes += ((yk[i] ?? 0) - (pk[i] ?? 0)) ** 2;
        ssTot += ((yk[i] ?? 0) - mean) ** 2;
      }
      totalScore += 1 - ssRes / (ssTot || 1);
    }
    return totalScore / Y.length;
  }
}

export class ClassifierChain {
  estimator: MultiOutputClassifierOptions["estimator"];
  order: number[] | "random" | null;
  estimators_: MultiOutputClassifierOptions["estimator"][] | null = null;
  order_: number[] | null = null;

  constructor(options: {
    estimator: MultiOutputClassifierOptions["estimator"];
    order?: number[] | "random" | null;
  }) {
    this.estimator = options.estimator;
    this.order = options.order ?? null;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const nOutputs = Y.length;
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    this.order_ =
      this.order === "random"
        ? Array.from({ length: nOutputs }, (_, i) => i).sort(
            () => Math.random() - 0.5,
          )
        : (this.order ?? Array.from({ length: nOutputs }, (_, i) => i));

    this.estimators_ = [];
    let augX: Float64Array[] = X.map((xi) => new Float64Array(xi));

    for (let idx = 0; idx < nOutputs; idx++) {
      const k = this.order_[idx] ?? idx;
      const est = Object.create(
        Object.getPrototypeOf(this.estimator) as object,
      ) as typeof this.estimator;
      Object.assign(est, JSON.parse(JSON.stringify(this.estimator)));
      est.fit(augX, Y[k] as Int32Array);
      this.estimators_.push(est);
      // Augment X with predictions
      const preds = est.predict(augX);
      augX = augX.map((xi, i) => {
        const newXi = new Float64Array(p + idx + 1);
        for (let j = 0; j < xi.length; j++) newXi[j] = xi[j] ?? 0;
        newXi[xi.length] = preds[i] ?? 0;
        return newXi;
      });
      void n;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (!this.estimators_ || !this.order_)
      throw new NotFittedError("ClassifierChain is not fitted.");
    const nOutputs = this.estimators_.length;
    const results: Int32Array[] = Array.from(
      { length: nOutputs },
      () => new Int32Array(X.length),
    );
    let augX: Float64Array[] = X.map((xi) => new Float64Array(xi));

    for (let idx = 0; idx < nOutputs; idx++) {
      const k = this.order_[idx] ?? idx;
      const preds = (this.estimators_[idx] as typeof this.estimator).predict(
        augX,
      );
      results[k] = preds;
      augX = augX.map((xi, i) => {
        const newXi = new Float64Array(xi.length + 1);
        for (let j = 0; j < xi.length; j++) newXi[j] = xi[j] ?? 0;
        newXi[xi.length] = preds[i] ?? 0;
        return newXi;
      });
    }
    return results;
  }
}
