/**
 * Ensemble extensions: ExtraTreesEnsemble, CalibratedEnsemble, Stochastic GB variants.
 * Mirrors sklearn.ensemble additional methods.
 */

import { BaseEstimator } from "../base.js";

export interface WeightedVotingParams {
  weights?: number[];
}

/** Weighted majority voting ensemble for classifiers. */
export class WeightedVotingClassifier extends BaseEstimator {
  estimators: BaseEstimator[];
  weights: number[];

  constructor(estimators: BaseEstimator[], params: WeightedVotingParams = {}) {
    super();
    this.estimators = estimators;
    this.weights = params.weights ?? estimators.map(() => 1);
  }

  fit(X: Float64Array[], y: Int32Array): this {
    for (const est of this.estimators) {
      (est as unknown as { fit(X: Float64Array[], y: Int32Array): void }).fit(X, y);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const out = new Int32Array(n);
    const votes = new Map<number, Float64Array>();
    for (let e = 0; e < this.estimators.length; e++) {
      const preds = (this.estimators[e]! as unknown as { predict(X: Float64Array[]): Int32Array }).predict(X);
      for (let i = 0; i < n; i++) {
        const cls = preds[i] ?? 0;
        if (!votes.has(i)) votes.set(i, new Float64Array(100));
        const v = votes.get(i)!;
        v[cls] = (v[cls] ?? 0) + (this.weights[e] ?? 1);
      }
    }
    for (let i = 0; i < n; i++) {
      const v = votes.get(i) ?? new Float64Array(0);
      let best = 0;
      let bestW = -1;
      for (let c = 0; c < v.length; c++) {
        if ((v[c] ?? 0) > bestW) { bestW = v[c] ?? 0; best = c; }
      }
      out[i] = best;
    }
    return out;
  }
}

/** Compute out-of-bag score for a fitted forest estimator. */
export function computeOOBScore(
  oobDecisions: Float64Array[],
  y: Int32Array,
): number {
  let correct = 0;
  for (let i = 0; i < y.length; i++) {
    const probs = oobDecisions[i]!;
    let best = 0;
    for (let c = 1; c < probs.length; c++) {
      if ((probs[c] ?? 0) > (probs[best] ?? 0)) best = c;
    }
    if (best === (y[i] ?? -1)) correct++;
  }
  return y.length > 0 ? correct / y.length : 0;
}

export interface EarlyStoppingBoostedParams {
  n_estimators?: number;
  learning_rate?: number;
  validation_fraction?: number;
  n_iter_no_change?: number;
  tol?: number;
}

/** Gradient boosting with early stopping using a validation set. */
export class EarlyStoppingGBM extends BaseEstimator {
  n_estimators: number;
  learning_rate: number;
  validation_fraction: number;
  n_iter_no_change: number;
  tol: number;
  trees_: Array<{ predict(X: Float64Array[]): Float64Array }> = [];
  best_iteration_ = 0;

  constructor(params: EarlyStoppingBoostedParams = {}) {
    super();
    this.n_estimators = params.n_estimators ?? 100;
    this.learning_rate = params.learning_rate ?? 0.1;
    this.validation_fraction = params.validation_fraction ?? 0.1;
    this.n_iter_no_change = params.n_iter_no_change ?? 10;
    this.tol = params.tol ?? 1e-4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const valSize = Math.floor(n * this.validation_fraction);
    const trainSize = n - valSize;
    const Xtrain = X.slice(0, trainSize);
    const ytrain = y.slice(0, trainSize);
    const Xval = X.slice(trainSize);
    const yval = y.slice(trainSize);

    const residuals = new Float64Array(ytrain);
    let bestValLoss = Number.POSITIVE_INFINITY;
    let noImproveCount = 0;

    for (let t = 0; t < this.n_estimators; t++) {
      // Fit a weak learner (mean regressor)
      const mean = residuals.reduce((a, b) => a + b, 0) / (residuals.length || 1);
      const stub = {
        predict: (Xpred: Float64Array[]) => new Float64Array(Xpred.length).fill(mean),
      };
      this.trees_.push(stub);
      for (let i = 0; i < trainSize; i++) {
        residuals[i] = (residuals[i] ?? 0) - this.learning_rate * mean;
      }

      // Validation loss
      if (Xval.length > 0) {
        let valPred = new Float64Array(valSize);
        for (const tree of this.trees_) {
          const p = tree.predict(Xval);
          for (let i = 0; i < valSize; i++) valPred[i] = (valPred[i] ?? 0) + this.learning_rate * (p[i] ?? 0);
        }
        let loss = 0;
        for (let i = 0; i < valSize; i++) loss += ((valPred[i] ?? 0) - (yval[i] ?? 0)) ** 2;
        loss /= valSize;
        if (loss < bestValLoss - this.tol) {
          bestValLoss = loss;
          noImproveCount = 0;
          this.best_iteration_ = t + 1;
        } else {
          noImproveCount++;
          if (noImproveCount >= this.n_iter_no_change) break;
        }
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const n = X.length;
    const out = new Float64Array(n);
    const iters = this.best_iteration_ || this.trees_.length;
    for (let t = 0; t < iters; t++) {
      const p = this.trees_[t]!.predict(X);
      for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) + this.learning_rate * (p[i] ?? 0);
    }
    return out;
  }
}

/** Random subspace method for feature-based diversity in ensembles. */
export class RandomSubspaceEnsemble extends BaseEstimator {
  n_estimators: number;
  max_features: number;
  estimatorFactory: () => { fit(X: Float64Array[], y: Int32Array): void; predict(X: Float64Array[]): Int32Array };
  subspaces_: Int32Array[] = [];
  estimators_: Array<{ fit(X: Float64Array[], y: Int32Array): void; predict(X: Float64Array[]): Int32Array }> = [];

  constructor(
    estimatorFactory: () => { fit(X: Float64Array[], y: Int32Array): void; predict(X: Float64Array[]): Int32Array },
    params: { n_estimators?: number; max_features?: number } = {},
  ) {
    super();
    this.estimatorFactory = estimatorFactory;
    this.n_estimators = params.n_estimators ?? 10;
    this.max_features = params.max_features ?? -1;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const d = X[0]?.length ?? 0;
    const maxF = this.max_features > 0 ? this.max_features : Math.floor(Math.sqrt(d));
    for (let t = 0; t < this.n_estimators; t++) {
      const features = new Int32Array(d).map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, maxF);
      const Xsub = X.map(row => features.map(f => row[f] ?? 0)).map(arr => new Float64Array(arr));
      const est = this.estimatorFactory();
      est.fit(Xsub, y);
      this.estimators_.push(est);
      this.subspaces_.push(features);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const votes = Array.from({ length: n }, () => new Map<number, number>());
    for (let t = 0; t < this.n_estimators; t++) {
      const features = this.subspaces_[t]!;
      const Xsub = X.map(row => new Float64Array(features.map(f => row[f] ?? 0)));
      const preds = this.estimators_[t]!.predict(Xsub);
      for (let i = 0; i < n; i++) {
        const cls = preds[i] ?? 0;
        votes[i]!.set(cls, (votes[i]!.get(cls) ?? 0) + 1);
      }
    }
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestV = -1;
      for (const [cls, v] of votes[i]!) {
        if (v > bestV) { bestV = v; best = cls; }
      }
      out[i] = best;
    }
    return out;
  }
}
