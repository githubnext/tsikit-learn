/**
 * Additional multioutput estimators: RegressorChain, ClassifierChain.
 * Mirrors sklearn.multioutput extras.
 */

import { NotFittedError } from "../exceptions.js";

export type BaseRegressor = {
  fit(X: Float64Array[], y: Float64Array): BaseRegressor;
  predict(X: Float64Array[]): Float64Array;
};

export type BaseClassifier = {
  fit(X: Float64Array[], y: Int32Array): BaseClassifier;
  predict(X: Float64Array[]): Int32Array;
};

export class RegressorChain {
  base: BaseRegressor;
  order: number[] | null;
  randomState: number;

  private estimators_: BaseRegressor[] = [];
  private order_: number[] = [];
  private nTargets_: number = 0;

  constructor(
    base: BaseRegressor,
    options: { order?: number[] | null; randomState?: number } = {},
  ) {
    this.base = base;
    this.order = options.order ?? null;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const nTargets = Y[0]?.length ?? 0;
    this.nTargets_ = nTargets;

    if (this.order) {
      this.order_ = this.order;
    } else {
      this.order_ = Array.from({ length: nTargets }, (_, i) => i);
    }

    const Xaug: Float64Array[] = X.map((row) => row.slice());
    this.estimators_ = [];

    for (const targetIdx of this.order_) {
      const yTarget = new Float64Array(n);
      for (let i = 0; i < n; i++) yTarget[i] = Y[i]?.[targetIdx] ?? 0;

      const estimator = Object.assign({}, this.base) as BaseRegressor;
      estimator.fit(Xaug, yTarget);
      this.estimators_.push(estimator);

      // Augment X with predictions
      const preds = estimator.predict(Xaug);
      for (let i = 0; i < n; i++) {
        const newRow = new Float64Array(Xaug[i]!.length + 1);
        newRow.set(Xaug[i]!);
        newRow[Xaug[i]!.length] = preds[i] ?? 0;
        Xaug[i] = newRow;
      }
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (this.estimators_.length === 0) throw new NotFittedError("RegressorChain is not fitted");
    const n = X.length;
    const nTargets = this.nTargets_;

    const Y: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nTargets));
    const Xaug = X.map((row) => row.slice());

    for (let k = 0; k < this.order_.length; k++) {
      const targetIdx = this.order_[k] ?? k;
      const estimator = this.estimators_[k];
      if (!estimator) continue;

      const preds = estimator.predict(Xaug);
      for (let i = 0; i < n; i++) {
        Y[i]![targetIdx] = preds[i] ?? 0;
        const newRow = new Float64Array(Xaug[i]!.length + 1);
        newRow.set(Xaug[i]!);
        newRow[Xaug[i]!.length] = preds[i] ?? 0;
        Xaug[i] = newRow;
      }
    }

    return Y;
  }
}

export class ClassifierChain {
  base: BaseClassifier;
  order: number[] | null;

  private estimators_: BaseClassifier[] = [];
  private order_: number[] = [];
  private nTargets_: number = 0;

  constructor(
    base: BaseClassifier,
    options: { order?: number[] | null } = {},
  ) {
    this.base = base;
    this.order = options.order ?? null;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const n = X.length;
    const nTargets = Y[0]?.length ?? 0;
    this.nTargets_ = nTargets;

    this.order_ = this.order ?? Array.from({ length: nTargets }, (_, i) => i);

    const Xaug: Float64Array[] = X.map((row) => row.slice());
    this.estimators_ = [];

    for (const targetIdx of this.order_) {
      const yTarget = new Int32Array(n);
      for (let i = 0; i < n; i++) yTarget[i] = Y[i]?.[targetIdx] ?? 0;

      const estimator = Object.assign({}, this.base) as BaseClassifier;
      estimator.fit(Xaug, yTarget);
      this.estimators_.push(estimator);

      const preds = estimator.predict(Xaug);
      for (let i = 0; i < n; i++) {
        const newRow = new Float64Array(Xaug[i]!.length + 1);
        newRow.set(Xaug[i]!);
        newRow[Xaug[i]!.length] = preds[i] ?? 0;
        Xaug[i] = newRow;
      }
    }

    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (this.estimators_.length === 0) throw new NotFittedError("ClassifierChain is not fitted");
    const n = X.length;
    const nTargets = this.nTargets_;

    const Y: Int32Array[] = Array.from({ length: n }, () => new Int32Array(nTargets));
    const Xaug = X.map((row) => row.slice());

    for (let k = 0; k < this.order_.length; k++) {
      const targetIdx = this.order_[k] ?? k;
      const estimator = this.estimators_[k];
      if (!estimator) continue;

      const preds = estimator.predict(Xaug);
      for (let i = 0; i < n; i++) {
        Y[i]![targetIdx] = preds[i] ?? 0;
        const newRow = new Float64Array(Xaug[i]!.length + 1);
        newRow.set(Xaug[i]!);
        newRow[Xaug[i]!.length] = preds[i] ?? 0;
        Xaug[i] = newRow;
      }
    }

    return Y;
  }
}
