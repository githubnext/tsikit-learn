/**
 * SelfTrainingClassifier — semi-supervised learning via self-training.
 * Unlabeled samples must have label -1.
 */

import { NotFittedError } from "../exceptions.js";

export interface SelfTrainingBaseEstimator {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba(X: Float64Array[]): Float64Array[];
}

export interface SelfTrainingOptions {
  threshold?: number;
  maxIter?: number;
  criterion?: "threshold" | "k_best";
  kBest?: number;
}

export class SelfTrainingClassifier {
  private estimator: SelfTrainingBaseEstimator;
  private threshold: number;
  private maxIter: number;
  private criterion: "threshold" | "k_best";
  private kBest: number;
  private fitted = false;

  constructor(
    estimator: SelfTrainingBaseEstimator,
    options: SelfTrainingOptions = {},
  ) {
    this.estimator = estimator;
    this.threshold = options.threshold ?? 0.75;
    this.maxIter = options.maxIter ?? 10;
    this.criterion = options.criterion ?? "threshold";
    this.kBest = options.kBest ?? 10;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const labels = Int32Array.from(y);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const labeledIdx: number[] = [];
      for (let i = 0; i < n; i++)
        if ((labels[i] ?? -1) !== -1) labeledIdx.push(i);

      if (labeledIdx.length === 0) break;

      const Xl = labeledIdx.map((i) => X[i] ?? new Float64Array(0));
      const yl = Int32Array.from(labeledIdx, (i) => labels[i] ?? 0);

      this.estimator.fit(Xl, yl);

      const unlabeledIdx: number[] = [];
      for (let i = 0; i < n; i++)
        if ((labels[i] ?? -1) === -1) unlabeledIdx.push(i);

      if (unlabeledIdx.length === 0) break;

      const Xu = unlabeledIdx.map((i) => X[i] ?? new Float64Array(0));
      const proba = this.estimator.predictProba(Xu);
      const preds = this.estimator.predict(Xu);

      let added = 0;

      if (this.criterion === "threshold") {
        for (let k = 0; k < unlabeledIdx.length; k++) {
          const row = proba[k] ?? new Float64Array(0);
          let maxP = 0;
          for (let c = 0; c < row.length; c++)
            if ((row[c] ?? 0) > maxP) maxP = row[c] ?? 0;
          if (maxP >= this.threshold) {
            labels[unlabeledIdx[k] ?? 0] = preds[k] ?? 0;
            added++;
          }
        }
      } else {
        // k_best: pick top-k by max probability
        const scores = unlabeledIdx.map((_, k) => {
          const row = proba[k] ?? new Float64Array(0);
          let maxP = 0;
          for (let c = 0; c < row.length; c++)
            if ((row[c] ?? 0) > maxP) maxP = row[c] ?? 0;
          return maxP;
        });
        const sorted = scores
          .map((s, k) => ({ s, k }))
          .sort((a, b) => b.s - a.s)
          .slice(0, this.kBest);
        for (const { k } of sorted) {
          labels[unlabeledIdx[k] ?? 0] = preds[k] ?? 0;
          added++;
        }
      }

      if (added === 0) break;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("SelfTrainingClassifier");
    return this.estimator.predict(X);
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("SelfTrainingClassifier");
    return this.estimator.predictProba(X);
  }
}
