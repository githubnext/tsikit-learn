/**
 * Multiclass meta-estimators.
 * Mirrors sklearn.multiclass: OneVsRestClassifier, OneVsOneClassifier.
 */

import { NotFittedError } from "../exceptions.js";

export interface BinaryClassifier {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
  score?(X: Float64Array[], y: Float64Array): number;
}

export class OneVsRestClassifier {
  estimator: BinaryClassifier;
  estimators_: BinaryClassifier[] | null = null;
  classes_: Float64Array | null = null;

  constructor(estimator: BinaryClassifier) {
    this.estimator = estimator;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    this.estimators_ = [];

    for (const cls of uniqueClasses) {
      const yBin = new Float64Array(y.length);
      for (let i = 0; i < y.length; i++) {
        yBin[i] = (y[i] ?? 0) === cls ? 1 : 0;
      }
      const est = Object.create(Object.getPrototypeOf(this.estimator) as object) as BinaryClassifier;
      Object.assign(est, this.estimator);
      est.fit(X, yBin);
      this.estimators_.push(est);
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_ === null || this.classes_ === null)
      throw new NotFittedError("OneVsRestClassifier");

    const classes = this.classes_;
    const n = X.length;
    const nClasses = classes.length;

    // Get decision scores for each class
    const scores: Float64Array[] = this.estimators_.map((est) => est.predict(X));

    return new Float64Array(
      Array.from({ length: n }, (_, i) => {
        let maxScore = -Infinity;
        let bestClass = classes[0] ?? 0;
        for (let c = 0; c < nClasses; c++) {
          const score = (scores[c] ?? new Float64Array(n))[i] ?? 0;
          if (score > maxScore) {
            maxScore = score;
            bestClass = classes[c] ?? 0;
          }
        }
        return bestClass;
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

export class OneVsOneClassifier {
  estimator: BinaryClassifier;
  estimators_: BinaryClassifier[] | null = null;
  classes_: Float64Array | null = null;
  pairIndices_: [number, number][] | null = null;

  constructor(estimator: BinaryClassifier) {
    this.estimator = estimator;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    this.estimators_ = [];
    this.pairIndices_ = [];

    for (let i = 0; i < uniqueClasses.length; i++) {
      for (let j = i + 1; j < uniqueClasses.length; j++) {
        const ci = uniqueClasses[i] as number;
        const cj = uniqueClasses[j] as number;
        this.pairIndices_.push([i, j]);

        // Filter samples for these two classes
        const mask: number[] = [];
        for (let k = 0; k < y.length; k++) {
          if ((y[k] ?? 0) === ci || (y[k] ?? 0) === cj) mask.push(k);
        }
        const XSub = mask.map((k) => X[k] ?? new Float64Array(0));
        const ySub = new Float64Array(mask.map((k) => ((y[k] ?? 0) === ci ? 0 : 1)));

        const est = Object.create(Object.getPrototypeOf(this.estimator) as object) as BinaryClassifier;
        Object.assign(est, this.estimator);
        est.fit(XSub, ySub);
        this.estimators_.push(est);
      }
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_ === null || this.classes_ === null || this.pairIndices_ === null)
      throw new NotFittedError("OneVsOneClassifier");

    const classes = this.classes_;
    const n = X.length;
    const nClasses = classes.length;

    return new Float64Array(
      Array.from({ length: n }, (_, i) => {
        const votes = new Int32Array(nClasses);
        for (let e = 0; e < this.estimators_!.length; e++) {
          const est = this.estimators_![e] as BinaryClassifier;
          const [ci, cj] = this.pairIndices_![e] as [number, number];
          const pred = (est.predict([X[i] ?? new Float64Array(0)]))[0] ?? 0;
          if (pred === 0) votes[ci] = (votes[ci] ?? 0) + 1;
          else votes[cj] = (votes[cj] ?? 0) + 1;
        }

        let bestIdx = 0;
        let bestVotes = votes[0] ?? 0;
        for (let c = 1; c < nClasses; c++) {
          if ((votes[c] ?? 0) > bestVotes) {
            bestVotes = votes[c] ?? 0;
            bestIdx = c;
          }
        }
        return classes[bestIdx] ?? 0;
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
