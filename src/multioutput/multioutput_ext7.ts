/**
 * Multi-output extensions: MultiOutputClassifier extensions.
 * Mirrors sklearn.multioutput advanced methods.
 */

import { BaseEstimator } from "../base.js";

type MultiTargetEstimator = {
  fit(X: Float64Array[], y: Int32Array): unknown;
  predict(X: Float64Array[]): Int32Array;
  score(X: Float64Array[], y: Int32Array): number;
};

type MultiTargetRegressorEstimator = {
  fit(X: Float64Array[], y: Float64Array): unknown;
  predict(X: Float64Array[]): Float64Array;
  score(X: Float64Array[], y: Float64Array): number;
};

export interface MultiLabelClassifierParams {
  classifier_chain?: boolean;
  order?: number[] | null;
}

/** MultiLabelClassifier: multi-label binary relevance or chain classifier. */
export class MultiLabelClassifier extends BaseEstimator {
  estimator: MultiTargetEstimator;
  classifier_chain: boolean;
  order: number[] | null;
  estimators_: MultiTargetEstimator[] = [];
  classes_: Int32Array[] = [];
  n_outputs_ = 0;

  constructor(estimator: MultiTargetEstimator, params: MultiLabelClassifierParams = {}) {
    super();
    this.estimator = estimator;
    this.classifier_chain = params.classifier_chain ?? false;
    this.order = params.order ?? null;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    this.n_outputs_ = Y[0]?.length ?? 0;
    const order = this.order ?? Array.from({ length: this.n_outputs_ }, (_, i) => i);
    for (let i = 0; i < this.n_outputs_; i++) {
      const target = new Int32Array(Y.map((yi) => yi[order[i]!] ?? 0));
      const est = Object.create(this.estimator) as MultiTargetEstimator;
      if (this.classifier_chain && i > 0) {
        // Augment X with previous predictions
        const augX = X.map((xi, idx) => {
          const aug = new Float64Array(xi.length + i);
          for (let k = 0; k < xi.length; k++) aug[k] = xi[k] ?? 0;
          for (let j = 0; j < i; j++) aug[xi.length + j] = (this.estimators_[j]?.predict([X[idx]!])[0] ?? 0);
          return aug;
        });
        est.fit(augX, target);
      } else {
        est.fit(X, target);
      }
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    const n = X.length;
    const order = this.order ?? Array.from({ length: this.n_outputs_ }, (_, i) => i);
    const result: Int32Array[] = Array.from({ length: n }, () => new Int32Array(this.n_outputs_));
    for (let i = 0; i < this.n_outputs_; i++) {
      let Xpred = X;
      if (this.classifier_chain && i > 0) {
        Xpred = X.map((xi, idx) => {
          const aug = new Float64Array(xi.length + i);
          for (let k = 0; k < xi.length; k++) aug[k] = xi[k] ?? 0;
          for (let j = 0; j < i; j++) aug[xi.length + j] = result[idx]?.[order[j]!] ?? 0;
          return aug;
        });
      }
      const pred = this.estimators_[i]!.predict(Xpred);
      for (let idx = 0; idx < n; idx++) result[idx]![order[i]!] = pred[idx] ?? 0;
    }
    return result;
  }

  score(X: Float64Array[], Y: Int32Array[]): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < Y.length; i++) {
      const pred = preds[i]!, true_ = Y[i]!;
      let allMatch = true;
      for (let k = 0; k < true_.length; k++) if (pred[k] !== true_[k]) { allMatch = false; break; }
      if (allMatch) correct++;
    }
    return correct / Y.length;
  }
}
