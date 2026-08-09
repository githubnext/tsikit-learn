/**
 * Multiclass extensions: Error-Correcting Output Codes (ECOC), Calibrated classifier.
 * Mirrors sklearn.multiclass advanced classifiers.
 */

import { BaseEstimator } from "../base.js";

type BinaryClassifier = {
  fit(X: Float64Array[], y: Int32Array): unknown;
  predict(X: Float64Array[]): Int32Array;
  decision_function?(X: Float64Array[]): Float64Array;
  score(X: Float64Array[], y: Int32Array): number;
};

export interface OutputCodeClassifierExtParams {
  code_size?: number;
  random_state?: number | null;
}

/** OutputCodeClassifier: multiclass via random binary codes. */
export class OutputCodeClassifierExt extends BaseEstimator {
  estimator: BinaryClassifier;
  code_size: number;
  random_state: number | null;
  classes_: Int32Array = new Int32Array(0);
  code_book_: Int32Array[] = [];
  estimators_: BinaryClassifier[] = [];

  constructor(estimator: BinaryClassifier, params: OutputCodeClassifierExtParams = {}) {
    super();
    this.estimator = estimator;
    this.code_size = params.code_size ?? 1;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;
    const nBits = Math.max(1, Math.ceil(this.code_size * k));
    const seed = this.random_state ?? 42;
    // Generate random code book
    this.code_book_ = classes.map((_, i) =>
      new Int32Array(nBits).map((_, j) => ((seed + i * 37 + j * 13) * 1664525) % 2),
    );
    this.estimators_ = [];
    const classIdx = new Map(classes.map((c, i) => [c, i]));
    for (let b = 0; b < nBits; b++) {
      const binaryY = new Int32Array(y.map((c) => {
        const ci = classIdx.get(c) ?? 0;
        return (this.code_book_[ci]?.[b] ?? 0) === 1 ? 1 : -1;
      }));
      const est = Object.create(this.estimator) as BinaryClassifier;
      est.fit(X, binaryY);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const nBits = this.estimators_.length;
    const k = this.classes_.length;
    return new Int32Array(X.map((xi) => {
      const codePred = new Int32Array(nBits).map((_, b) => {
        const p = this.estimators_[b]!.predict([xi])[0] ?? -1;
        return p > 0 ? 1 : 0;
      });
      // Find nearest code in code book (Hamming distance)
      let best = 0, bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let b = 0; b < nBits; b++) if (codePred[b] !== (this.code_book_[c]?.[b] ?? 0)) dist++;
        if (dist < bestDist) { best = this.classes_[c] ?? 0; bestDist = dist; }
      }
      return best;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

export interface OneVsOneClassifierExtParams {
  n_jobs?: number;
}

/** OneVsOneClassifier: pairwise binary classifiers. */
export class OneVsOneClassifierExt extends BaseEstimator {
  estimator: BinaryClassifier;
  classes_: Int32Array = new Int32Array(0);
  estimators_: BinaryClassifier[] = [];
  pairwise_indices_: Array<[number, number]> = [];

  constructor(estimator: BinaryClassifier, _params: OneVsOneClassifierExtParams = {}) {
    super();
    this.estimator = estimator;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;
    this.pairwise_indices_ = [];
    this.estimators_ = [];
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        this.pairwise_indices_.push([i, j]);
        const mask = Array.from(y).map((c) => c === classes[i] || c === classes[j]);
        const Xs = X.filter((_, idx) => mask[idx]);
        const ys = new Int32Array(Array.from(y).filter((_, idx) => mask[idx]).map((c) => c === classes[i] ? 1 : -1));
        const est = Object.create(this.estimator) as BinaryClassifier;
        est.fit(Xs, ys);
        this.estimators_.push(est);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const k = this.classes_.length;
    return new Int32Array(X.map((xi) => {
      const votes = new Float64Array(k);
      for (let e = 0; e < this.estimators_.length; e++) {
        const [i, j] = this.pairwise_indices_[e]!;
        const pred = this.estimators_[e]!.predict([xi])[0] ?? -1;
        if (pred > 0) votes[i] = (votes[i] ?? 0) + 1;
        else votes[j] = (votes[j] ?? 0) + 1;
      }
      let best = 0, bestV = -1;
      for (let c = 0; c < k; c++) if ((votes[c] ?? 0) > bestV) { best = this.classes_[c] ?? 0; bestV = votes[c] ?? 0; }
      return best;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}
