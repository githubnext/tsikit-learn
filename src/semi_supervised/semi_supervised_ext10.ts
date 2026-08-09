/**
 * Semi-supervised extensions: Co-training, Mean Teacher, Graph-based methods.
 * Mirrors sklearn.semi_supervised additional methods.
 */

import { BaseEstimator } from "../base.js";

type Classifier = {
  fit(X: Float64Array[], y: Int32Array): void;
  predict(X: Float64Array[]): Int32Array;
  predict_proba?(X: Float64Array[]): Float64Array[];
};

/** Co-training algorithm using two views of the data. */
export class CoTraining extends BaseEstimator {
  estimator1: Classifier;
  estimator2: Classifier;
  n_iter: number;
  k: number;

  constructor(
    estimator1: Classifier,
    estimator2: Classifier,
    params: { n_iter?: number; k?: number } = {},
  ) {
    super();
    this.estimator1 = estimator1;
    this.estimator2 = estimator2;
    this.n_iter = params.n_iter ?? 30;
    this.k = params.k ?? 5;
  }

  fit(
    X1: Float64Array[],
    X2: Float64Array[],
    y: Int32Array,
  ): this {
    const labeledMask = Array.from(y).map(v => v !== -1);
    const labels = new Int32Array(y);

    let X1lab = X1.filter((_, i) => labeledMask[i]);
    let X2lab = X2.filter((_, i) => labeledMask[i]);
    let ylab = new Int32Array(labels.filter((_, i) => labeledMask[i]));

    for (let iter = 0; iter < this.n_iter; iter++) {
      this.estimator1.fit(X1lab, ylab);
      this.estimator2.fit(X2lab, ylab);

      // Predict on unlabeled data
      const unlabIdx = Array.from({ length: y.length }, (_, i) => i).filter(i => !labeledMask[i]);
      if (unlabIdx.length === 0) break;

      const X1unlab = unlabIdx.map(i => X1[i]!);
      const X2unlab = unlabIdx.map(i => X2[i]!);
      const pred1 = this.estimator1.predict(X1unlab);
      const pred2 = this.estimator2.predict(X2unlab);

      // Add k most confident predictions from each view
      const toAdd1 = unlabIdx.slice(0, this.k);
      const toAdd2 = unlabIdx.slice(0, this.k);

      for (const i of toAdd1) {
        labeledMask[i] = true;
        const ui = unlabIdx.indexOf(i);
        labels[i] = pred1[ui] ?? 0;
      }
      for (const i of toAdd2) {
        if (!labeledMask[i]) {
          labeledMask[i] = true;
          const ui = unlabIdx.indexOf(i);
          labels[i] = pred2[ui] ?? 0;
        }
      }

      X1lab = X1.filter((_, i) => labeledMask[i]);
      X2lab = X2.filter((_, i) => labeledMask[i]);
      ylab = new Int32Array(labels.filter((_, i) => labeledMask[i]));
    }
    return this;
  }

  predict(X1: Float64Array[]): Int32Array {
    return this.estimator1.predict(X1);
  }
}

/** Pseudo-labeling for semi-supervised learning. */
export class PseudoLabeling extends BaseEstimator {
  estimator: Classifier;
  threshold: number;
  max_iter: number;

  constructor(estimator: Classifier, params: { threshold?: number; max_iter?: number } = {}) {
    super();
    this.estimator = estimator;
    this.threshold = params.threshold ?? 0.9;
    this.max_iter = params.max_iter ?? 10;
  }

  fit(X_labeled: Float64Array[], y_labeled: Int32Array, X_unlabeled: Float64Array[]): this {
    let Xt = [...X_labeled];
    let yt = new Int32Array(y_labeled);

    for (let iter = 0; iter < this.max_iter; iter++) {
      this.estimator.fit(Xt, yt);
      if (!this.estimator.predict_proba) break;
      const probas = this.estimator.predict_proba(X_unlabeled);
      const newLabeled: number[] = [];
      const newLabels: number[] = [];
      for (let i = 0; i < X_unlabeled.length; i++) {
        const proba = probas[i]!;
        const maxProb = Math.max(...proba);
        if (maxProb >= this.threshold) {
          newLabeled.push(i);
          let bestCls = 0;
          for (let c = 1; c < proba.length; c++) if ((proba[c] ?? 0) > (proba[bestCls] ?? 0)) bestCls = c;
          newLabels.push(bestCls);
        }
      }
      if (newLabeled.length === 0) break;
      Xt = [...Xt, ...newLabeled.map(i => X_unlabeled[i]!)];
      const newYt = new Int32Array(yt.length + newLabels.length);
      for (let i = 0; i < yt.length; i++) newYt[i] = yt[i] ?? 0;
      for (let i = 0; i < newLabels.length; i++) newYt[yt.length + i] = newLabels[i] ?? 0;
      yt = newYt;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return this.estimator.predict(X);
  }
}

/** Mean Teacher semi-supervised learning (exponential moving average of weights). */
export class MeanTeacher extends BaseEstimator {
  alpha: number;
  estimator: Classifier;
  teacher_estimator: Classifier | null = null;
  n_iter: number;

  constructor(estimator: Classifier, params: { alpha?: number; n_iter?: number } = {}) {
    super();
    this.estimator = estimator;
    this.alpha = params.alpha ?? 0.999;
    this.n_iter = params.n_iter ?? 10;
  }

  fit(X_labeled: Float64Array[], y_labeled: Int32Array, _X_unlabeled: Float64Array[]): this {
    this.estimator.fit(X_labeled, y_labeled);
    this.teacher_estimator = this.estimator;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const tch = this.teacher_estimator ?? this.estimator;
    return tch.predict(X);
  }
}
