/**
 * SequentialFeatureSelector: greedy forward or backward feature selection.
 * Mirrors sklearn.feature_selection.SequentialFeatureSelector.
 */

import { BaseEstimator } from "../base.js";
import { NotFittedError } from "../exceptions.js";

export type SFSEstimator = {
  fit(X: Float64Array[], y: Float64Array | Int32Array): unknown;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
};

export interface SequentialFeatureSelectorOptions {
  nFeaturesToSelect?: number | "auto";
  direction?: "forward" | "backward";
  scoring?: (est: SFSEstimator, X: Float64Array[], y: Float64Array | Int32Array) => number;
  cv?: number;
  tol?: number | null;
}

function subsetCols(X: Float64Array[], cols: number[]): Float64Array[] {
  return X.map((row) => {
    const out = new Float64Array(cols.length);
    for (let i = 0; i < cols.length; i++) out[i] = row[cols[i]!] ?? 0;
    return out;
  });
}

function cvScore(
  estimator: SFSEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  cv: number,
): number {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  let totalScore = 0;
  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize;
    const end = fold === cv - 1 ? n : start + foldSize;
    const trainX: Float64Array[] = [];
    const testX: Float64Array[] = [];
    const trainY: number[] = [];
    const testY: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i >= start && i < end) {
        testX.push(X[i]!);
        testY.push(y[i] ?? 0);
      } else {
        trainX.push(X[i]!);
        trainY.push(y[i] ?? 0);
      }
    }
    const yTrain = y instanceof Int32Array ? new Int32Array(trainY) : new Float64Array(trainY);
    const yTest = y instanceof Int32Array ? new Int32Array(testY) : new Float64Array(testY);
    estimator.fit(trainX, yTrain);
    totalScore += estimator.score(testX, yTest);
  }
  return totalScore / cv;
}

export class SequentialFeatureSelector extends BaseEstimator {
  estimator: SFSEstimator;
  nFeaturesToSelect: number | "auto";
  direction: "forward" | "backward";
  cv: number;
  tol: number | null;

  supportMask_: boolean[] | null = null;
  nFeaturesIn_: number | null = null;

  constructor(estimator: SFSEstimator, opts: SequentialFeatureSelectorOptions = {}) {
    super();
    this.estimator = estimator;
    this.nFeaturesToSelect = opts.nFeaturesToSelect ?? "auto";
    this.direction = opts.direction ?? "forward";
    this.cv = opts.cv ?? 5;
    this.tol = opts.tol ?? null;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const nFeatures = X[0]?.length ?? 0;
    this.nFeaturesIn_ = nFeatures;

    const target = this.nFeaturesToSelect === "auto"
      ? Math.floor(nFeatures / 2)
      : this.nFeaturesToSelect;

    const selected: Set<number> = new Set();
    const remaining: Set<number> = new Set(Array.from({ length: nFeatures }, (_, i) => i));

    if (this.direction === "backward") {
      for (let i = 0; i < nFeatures; i++) selected.add(i);
      remaining.clear();
    }

    const nToSelect = this.direction === "forward" ? target : nFeatures - target;

    for (let step = 0; step < nToSelect; step++) {
      let bestScore = -Number.POSITIVE_INFINITY;
      let bestFeature = -1;

      if (this.direction === "forward") {
        for (const f of remaining) {
          const cols = [...selected, f].sort((a, b) => a - b);
          const Xsub = subsetCols(X, cols);
          const score = cvScore(this.estimator, Xsub, y, this.cv);
          if (score > bestScore) { bestScore = score; bestFeature = f; }
        }
        if (bestFeature >= 0) {
          selected.add(bestFeature);
          remaining.delete(bestFeature);
        }
      } else {
        for (const f of selected) {
          const cols = [...selected].filter(x => x !== f).sort((a, b) => a - b);
          const Xsub = subsetCols(X, cols);
          const score = cvScore(this.estimator, Xsub, y, this.cv);
          if (score > bestScore) { bestScore = score; bestFeature = f; }
        }
        if (bestFeature >= 0) {
          selected.delete(bestFeature);
        }
      }
    }

    this.supportMask_ = Array.from({ length: nFeatures }, (_, i) => selected.has(i));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.supportMask_) throw new NotFittedError("SequentialFeatureSelector");
    const cols = this.supportMask_.map((v, i) => v ? i : -1).filter(i => i >= 0);
    return subsetCols(X, cols);
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupport(): boolean[] {
    if (!this.supportMask_) throw new NotFittedError("SequentialFeatureSelector");
    return this.supportMask_;
  }
}
