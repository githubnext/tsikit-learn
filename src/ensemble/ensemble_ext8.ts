/**
 * Ensemble extensions: IsolationForest, ExtraTreesClassifier extensions.
 * Mirrors sklearn.ensemble outlier detection and more.
 */

import { BaseEstimator } from "../base.js";

export interface IsolationForestExtParams {
  n_estimators?: number;
  max_samples?: number | "auto";
  contamination?: number | "auto";
  max_features?: number;
  random_state?: number | null;
}

/** IsolationForest: ensemble method for outlier detection. */
export class IsolationForestExt extends BaseEstimator {
  n_estimators: number;
  max_samples: number | "auto";
  contamination: number | "auto";
  max_features: number;
  random_state: number | null;
  estimators_: IsolationTree[] = [];
  score_samples_cache_: Float64Array = new Float64Array(0);
  threshold_ = -0.5;
  max_samples_: number = 256;

  constructor(params: IsolationForestExtParams = {}) {
    super();
    this.n_estimators = params.n_estimators ?? 100;
    this.max_samples = params.max_samples ?? "auto";
    this.contamination = params.contamination ?? "auto";
    this.max_features = params.max_features ?? 1.0;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const ms = this.max_samples === "auto" ? Math.min(256, n) : this.max_samples;
    this.max_samples_ = ms;
    const nf = X[0]?.length ?? 0;
    const nfSub = Math.max(1, Math.floor(this.max_features <= 1 ? this.max_features * nf : this.max_features));
    this.estimators_ = [];
    const seed = this.random_state ?? 42;
    for (let t = 0; t < this.n_estimators; t++) {
      const subsample: Float64Array[] = [];
      for (let i = 0; i < ms; i++) subsample.push(X[((seed + t * 37 + i * 13) * 1664525) % n]!);
      const feats: number[] = [];
      for (let k = 0; k < nfSub; k++) feats.push(((seed + t * 13 + k * 7) * 1664525) % nf);
      const tree = new IsolationTree(Math.ceil(Math.log2(Math.max(ms, 2))));
      tree.fit(subsample.map((xi) => new Float64Array(feats.map((f) => xi[f] ?? 0))));
      (tree as unknown as { feats: number[] }).feats = feats;
      this.estimators_.push(tree);
    }
    if (this.contamination !== "auto" && typeof this.contamination === "number") {
      const scores = this._score(X);
      const sorted = Array.from(scores).sort((a, b) => a - b);
      const cutIdx = Math.floor(this.contamination * n);
      this.threshold_ = sorted[cutIdx] ?? -0.5;
    }
    return this;
  }

  private _score(X: Float64Array[]): Float64Array {
    const n = X.length;
    const scores = new Float64Array(n);
    for (const tree of this.estimators_) {
      const feats = (tree as unknown as { feats: number[] }).feats;
      for (let i = 0; i < n; i++) {
        const xi = new Float64Array((feats ?? []).map((f: number) => X[i]?.[f] ?? 0));
        const depth = tree.pathLength(xi);
        scores[i]! += depth;
      }
    }
    const avgDepth = this.max_samples_;
    const c = avgDepth <= 1 ? 1 : 2 * (Math.log(avgDepth - 1) + 0.5772) - 2 * (avgDepth - 1) / avgDepth;
    for (let i = 0; i < n; i++) scores[i] = -(2 ** (-(scores[i] ?? 0) / this.n_estimators / Math.max(c, 1e-10)));
    return scores;
  }

  score_samples(X: Float64Array[]): Float64Array {
    return this._score(X);
  }

  decision_function(X: Float64Array[]): Float64Array {
    const s = this._score(X);
    return new Float64Array(s.map((v) => v - this.threshold_));
  }

  predict(X: Float64Array[]): Int32Array {
    const df = this.decision_function(X);
    return new Int32Array(df.map((v) => v >= 0 ? 1 : -1));
  }
}

class IsolationTree {
  maxDepth: number;
  left: IsolationTree | null = null;
  right: IsolationTree | null = null;
  splitFeature = 0;
  splitValue = 0;
  size = 0;

  constructor(maxDepth: number) {
    this.maxDepth = maxDepth;
  }

  fit(X: Float64Array[], depth = 0): void {
    this.size = X.length;
    if (depth >= this.maxDepth || X.length <= 1) return;
    const nf = X[0]?.length ?? 0;
    this.splitFeature = Math.floor(Math.random() * nf);
    let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
    for (const xi of X) { const v = xi[this.splitFeature] ?? 0; if (v < min) min = v; if (v > max) max = v; }
    if (min >= max) return;
    this.splitValue = min + Math.random() * (max - min);
    const left = X.filter((xi) => (xi[this.splitFeature] ?? 0) < this.splitValue);
    const right = X.filter((xi) => (xi[this.splitFeature] ?? 0) >= this.splitValue);
    this.left = new IsolationTree(this.maxDepth);
    this.right = new IsolationTree(this.maxDepth);
    this.left.fit(left, depth + 1);
    this.right.fit(right, depth + 1);
  }

  pathLength(x: Float64Array, depth = 0): number {
    if (!this.left && !this.right) {
      const c = this.size <= 1 ? 0 : 2 * (Math.log(this.size - 1) + 0.5772) - 2 * (this.size - 1) / this.size;
      return depth + c;
    }
    if ((x[this.splitFeature] ?? 0) < this.splitValue) {
      return this.left ? this.left.pathLength(x, depth + 1) : depth + 1;
    }
    return this.right ? this.right.pathLength(x, depth + 1) : depth + 1;
  }
}

export interface StackingClassifierExtParams {
  passthrough?: boolean;
  cv?: number;
}

type Classifier = {
  fit(X: Float64Array[], y: Int32Array): unknown;
  predict_proba?(X: Float64Array[]): Float64Array[];
  predict(X: Float64Array[]): Int32Array;
  score(X: Float64Array[], y: Int32Array): number;
};

/** StackingClassifierExt: ensemble stacking with meta-learner. */
export class StackingClassifierExt extends BaseEstimator {
  estimators: Array<[string, Classifier]>;
  final_estimator: Classifier;
  passthrough: boolean;
  cv: number;
  fitted_estimators_: Classifier[] = [];

  constructor(
    estimators: Array<[string, Classifier]>,
    finalEstimator: Classifier,
    params: StackingClassifierExtParams = {},
  ) {
    super();
    this.estimators = estimators;
    this.final_estimator = finalEstimator;
    this.passthrough = params.passthrough ?? false;
    this.cv = params.cv ?? 5;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const nEst = this.estimators.length;
    const metaX: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nEst + (this.passthrough ? (X[0]?.length ?? 0) : 0)));
    // Cross-val predictions for each base estimator
    const foldSize = Math.max(1, Math.floor(n / this.cv));
    for (let e = 0; e < nEst; e++) {
      const [, est] = this.estimators[e]!;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize;
        const end = Math.min(start + foldSize, n);
        const trainX = [...X.slice(0, start), ...X.slice(end)];
        const trainY = new Int32Array([...Array.from(y).slice(0, start), ...Array.from(y).slice(end)]);
        const testX = X.slice(start, end);
        try {
          est.fit(trainX, trainY);
          const pred = est.predict(testX);
          for (let i = 0; i < pred.length; i++) metaX[start + i]![e] = pred[i] ?? 0;
        } catch { /* skip */ }
      }
    }
    if (this.passthrough) {
      const nf = X[0]?.length ?? 0;
      for (let i = 0; i < n; i++) for (let k = 0; k < nf; k++) metaX[i]![nEst + k] = X[i]?.[k] ?? 0;
    }
    this.final_estimator.fit(metaX, y);
    // Refit all base estimators on full training data
    this.fitted_estimators_ = this.estimators.map(([, est]) => {
      est.fit(X, y);
      return est;
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const nEst = this.fitted_estimators_.length;
    const metaX = X.map((xi) => {
      const row = new Float64Array(nEst + (this.passthrough ? xi.length : 0));
      for (let e = 0; e < nEst; e++) row[e] = this.fitted_estimators_[e]!.predict([xi])[0] ?? 0;
      if (this.passthrough) for (let k = 0; k < xi.length; k++) row[nEst + k] = xi[k] ?? 0;
      return row;
    });
    return this.final_estimator.predict(metaX);
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}
