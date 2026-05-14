/**
 * StackingClassifier, StackingRegressor, AdaBoostClassifier, AdaBoostRegressor.
 * Mirrors sklearn.ensemble stacking and AdaBoost estimators.
 */

import { NotFittedError } from "../exceptions.js";

// ─── StackingClassifier ───────────────────────────────────────────────────────

export interface StackableClassifier {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export interface StackableRegressor {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
}

export interface StackingClassifierOptions {
  estimators: [string, StackableClassifier][];
  finalEstimator?: StackableClassifier;
  cv?: number;
  passthrough?: boolean;
}

export class StackingClassifier {
  estimators: [string, StackableClassifier][];
  finalEstimator: StackableClassifier;
  cv: number;
  passthrough: boolean;

  fittedEstimators_: StackableClassifier[] | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: StackingClassifierOptions) {
    this.estimators = opts.estimators;
    this.cv = opts.cv ?? 5;
    this.passthrough = opts.passthrough ?? false;
    this.finalEstimator = opts.finalEstimator ?? createDefaultClassifier();
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const nEstimators = this.estimators.length;
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(y[i] ?? 0);
    this.classes_ = Int32Array.from(Array.from(classSet).sort((a, b) => a - b));

    this.fittedEstimators_ = this.estimators.map(([, est]) => {
      est.fit(X, y);
      return est;
    });

    // Build meta-features
    const metaX: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const baseFeats = this.fittedEstimators_!.map((est) => {
        if (est.predictProba) {
          return Array.from(est.predictProba(X)[i] ?? new Float64Array(0));
        }
        const pred = est.predict(X);
        return [pred[i] ?? 0];
      }).flat();
      const extra = this.passthrough ? Array.from(X[i] ?? new Float64Array(0)) : [];
      return Float64Array.from([...baseFeats, ...extra]);
    });

    this.finalEstimator.fit(metaX, y);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fittedEstimators_) throw new NotFittedError("StackingClassifier");
    const n = X.length;
    const metaX: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const baseFeats = this.fittedEstimators_!.map((est) => {
        if (est.predictProba) {
          return Array.from(est.predictProba(X)[i] ?? new Float64Array(0));
        }
        const pred = est.predict(X);
        return [pred[i] ?? 0];
      }).flat();
      const extra = this.passthrough ? Array.from(X[i] ?? new Float64Array(0)) : [];
      return Float64Array.from([...baseFeats, ...extra]);
    });
    return this.finalEstimator.predict(metaX);
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

// ─── StackingRegressor ────────────────────────────────────────────────────────

export interface StackingRegressorOptions {
  estimators: [string, StackableRegressor][];
  finalEstimator?: StackableRegressor;
  passthrough?: boolean;
}

export class StackingRegressor {
  estimators: [string, StackableRegressor][];
  finalEstimator: StackableRegressor;
  passthrough: boolean;

  fittedEstimators_: StackableRegressor[] | null = null;

  constructor(opts: StackingRegressorOptions) {
    this.estimators = opts.estimators;
    this.passthrough = opts.passthrough ?? false;
    this.finalEstimator = opts.finalEstimator ?? createDefaultRegressor();
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this.fittedEstimators_ = this.estimators.map(([, est]) => {
      est.fit(X, y);
      return est;
    });

    const metaX: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const baseFeats = this.fittedEstimators_!.map((est) => {
        const pred = est.predict(X);
        return [pred[i] ?? 0];
      }).flat();
      const extra = this.passthrough ? Array.from(X[i] ?? new Float64Array(0)) : [];
      return Float64Array.from([...baseFeats, ...extra]);
    });

    this.finalEstimator.fit(metaX, y);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fittedEstimators_) throw new NotFittedError("StackingRegressor");
    const n = X.length;
    const metaX: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const baseFeats = this.fittedEstimators_!.map((est) => {
        const pred = est.predict(X);
        return [pred[i] ?? 0];
      }).flat();
      const extra = this.passthrough ? Array.from(X[i] ?? new Float64Array(0)) : [];
      return Float64Array.from([...baseFeats, ...extra]);
    });
    return this.finalEstimator.predict(metaX);
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ss_tot === 0 ? 1 : 1 - ss_res / ss_tot;
  }
}

// ─── AdaBoostClassifier ───────────────────────────────────────────────────────

export interface AdaBoostClassifierOptions {
  nEstimators?: number;
  learningRate?: number;
  algorithm?: "SAMME" | "SAMME.R";
}

/** Simple decision stump for AdaBoost. */
class DecisionStump {
  featureIdx = 0;
  threshold = 0;
  polarity = 1;

  fit(X: Float64Array[], y: Int32Array, weights: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    let bestError = Number.POSITIVE_INFINITY;

    for (let f = 0; f < d; f++) {
      const vals = X.map((xi) => xi[f] ?? 0);
      const sorted = [...vals].sort((a, b) => a - b);
      const thresholds = sorted.slice(0, -1).map((v, i) => (v + (sorted[i + 1] ?? v)) / 2);

      for (const thresh of thresholds) {
        for (const pol of [1, -1]) {
          let error = 0;
          for (let i = 0; i < n; i++) {
            const pred = pol * ((vals[i] ?? 0) <= thresh ? -1 : 1);
            const label = (y[i] ?? 0) === 1 ? 1 : -1;
            if (pred !== label) error += weights[i] ?? 0;
          }
          if (error < bestError) {
            bestError = error;
            this.featureIdx = f;
            this.threshold = thresh;
            this.polarity = pol;
          }
        }
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return Int32Array.from(X, (xi) => {
      const val = xi[this.featureIdx] ?? 0;
      return this.polarity * (val <= this.threshold ? -1 : 1);
    });
  }
}

export class AdaBoostClassifier {
  nEstimators: number;
  learningRate: number;

  estimators_: DecisionStump[] = [];
  estimatorWeights_: Float64Array | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: AdaBoostClassifierOptions = {}) {
    this.nEstimators = opts.nEstimators ?? 50;
    this.learningRate = opts.learningRate ?? 1.0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(y[i] ?? 0);
    const classes = Int32Array.from(Array.from(classSet).sort((a, b) => a - b));
    this.classes_ = classes;

    // Binary AdaBoost: map classes to +1/-1
    const yBin = Int32Array.from(y, (label) => (label === (classes[1] ?? 1) ? 1 : -1));

    const weights = new Float64Array(n).fill(1 / n);
    const alphas = new Float64Array(this.nEstimators);
    this.estimators_ = [];

    for (let t = 0; t < this.nEstimators; t++) {
      const stump = new DecisionStump();
      stump.fit(X, yBin as Int32Array, weights);
      const preds = stump.predict(X);

      let error = 0;
      for (let i = 0; i < n; i++) {
        if (preds[i] !== yBin[i]) error += weights[i] ?? 0;
      }
      error = Math.max(error, 1e-10);
      const alpha = this.learningRate * 0.5 * Math.log((1 - error) / error);
      alphas[t]! = alpha;

      // Update weights
      let sumW = 0;
      for (let i = 0; i < n; i++) {
        const correct = preds[i] === yBin[i] ? 1 : -1;
        weights[i]! = (weights[i] ?? 0) * Math.exp(-alpha * correct);
        sumW += weights[i]!;
      }
      for (let i = 0; i < n; i++) weights[i]! /= sumW;

      this.estimators_.push(stump);
    }
    this.estimatorWeights_ = alphas;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.estimatorWeights_ || !this.classes_) throw new NotFittedError("AdaBoostClassifier");
    const n = X.length;
    const scores = new Float64Array(n);
    for (let t = 0; t < this.estimators_.length; t++) {
      const alpha = this.estimatorWeights_[t] ?? 0;
      const preds = this.estimators_[t]!.predict(X);
      for (let i = 0; i < n; i++) scores[i]! += alpha * (preds[i] ?? 0);
    }
    return Int32Array.from(scores, (s) => (s >= 0 ? (this.classes_![1] ?? 1) : (this.classes_![0] ?? 0)));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

// ─── AdaBoostRegressor ────────────────────────────────────────────────────────

export interface AdaBoostRegressorOptions {
  nEstimators?: number;
  learningRate?: number;
  loss?: "linear" | "square" | "exponential";
}

class RegressionStump {
  featureIdx = 0;
  threshold = 0;
  leftVal = 0;
  rightVal = 0;

  fit(X: Float64Array[], y: Float64Array, weights: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    let bestLoss = Number.POSITIVE_INFINITY;

    for (let f = 0; f < d; f++) {
      const vals = X.map((xi) => xi[f] ?? 0);
      const sorted = [...vals].sort((a, b) => a - b);
      const thresholds = sorted.slice(0, -1).map((v, i) => (v + (sorted[i + 1] ?? v)) / 2);
      for (const thresh of thresholds) {
        const leftIdxs = vals.map((v, i) => (v <= thresh ? i : -1)).filter((i) => i >= 0);
        const rightIdxs = vals.map((v, i) => (v > thresh ? i : -1)).filter((i) => i >= 0);
        const wLeft = leftIdxs.reduce((s, i) => s + (weights[i] ?? 0), 0);
        const wRight = rightIdxs.reduce((s, i) => s + (weights[i] ?? 0), 0);
        const lv = wLeft > 0 ? leftIdxs.reduce((s, i) => s + (weights[i] ?? 0) * (y[i] ?? 0), 0) / wLeft : 0;
        const rv = wRight > 0 ? rightIdxs.reduce((s, i) => s + (weights[i] ?? 0) * (y[i] ?? 0), 0) / wRight : 0;
        let loss = 0;
        for (let i = 0; i < n; i++) {
          const pred = (vals[i] ?? 0) <= thresh ? lv : rv;
          loss += (weights[i] ?? 0) * Math.abs((y[i] ?? 0) - pred);
        }
        if (loss < bestLoss) {
          bestLoss = loss;
          this.featureIdx = f;
          this.threshold = thresh;
          this.leftVal = lv;
          this.rightVal = rv;
        }
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return Float64Array.from(X, (xi) =>
      (xi[this.featureIdx] ?? 0) <= this.threshold ? this.leftVal : this.rightVal,
    );
  }
}

export class AdaBoostRegressor {
  nEstimators: number;
  learningRate: number;
  loss: "linear" | "square" | "exponential";

  estimators_: RegressionStump[] = [];
  estimatorWeights_: Float64Array | null = null;

  constructor(opts: AdaBoostRegressorOptions = {}) {
    this.nEstimators = opts.nEstimators ?? 50;
    this.learningRate = opts.learningRate ?? 1.0;
    this.loss = opts.loss ?? "linear";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const weights = new Float64Array(n).fill(1 / n);
    const alphas: number[] = [];
    this.estimators_ = [];

    for (let t = 0; t < this.nEstimators; t++) {
      const stump = new RegressionStump();
      stump.fit(X, y, weights);
      const preds = stump.predict(X);

      const errors = Float64Array.from({ length: n }, (_, i) =>
        Math.abs((y[i] ?? 0) - (preds[i] ?? 0)),
      );
      const maxErr = errors.reduce((mx, v) => Math.max(mx, v), 0);
      const normErrors = maxErr > 0 ? Float64Array.from(errors, (e) => e / maxErr) : errors;

      let loss = 0;
      for (let i = 0; i < n; i++) {
        const e = normErrors[i] ?? 0;
        const lossFn = this.loss === "square" ? e * e : this.loss === "exponential" ? 1 - Math.exp(-e) : e;
        loss += (weights[i] ?? 0) * lossFn;
      }
      loss = Math.min(Math.max(loss, 1e-10), 1 - 1e-10);
      const beta = loss / (1 - loss);
      const alpha = this.learningRate * Math.log(1 / beta);
      alphas.push(alpha);

      let sumW = 0;
      for (let i = 0; i < n; i++) {
        const e = normErrors[i] ?? 0;
        const lossFn = this.loss === "square" ? e * e : this.loss === "exponential" ? 1 - Math.exp(-e) : e;
        weights[i]! = (weights[i] ?? 0) * Math.pow(beta, 1 - lossFn);
        sumW += weights[i]!;
      }
      if (sumW > 0) for (let i = 0; i < n; i++) weights[i]! /= sumW;
      this.estimators_.push(stump);
    }
    this.estimatorWeights_ = Float64Array.from(alphas);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimatorWeights_) throw new NotFittedError("AdaBoostRegressor");
    const n = X.length;
    // Weighted median
    const allPreds: Float64Array[] = this.estimators_.map((e) => e.predict(X));
    return Float64Array.from({ length: n }, (_, i) => {
      const pairs = allPreds.map((p, t) => ({ val: p[i] ?? 0, w: this.estimatorWeights_![t] ?? 0 }));
      pairs.sort((a, b) => a.val - b.val);
      const totalW = pairs.reduce((s, p) => s + p.w, 0);
      let cumW = 0;
      for (const p of pairs) {
        cumW += p.w;
        if (cumW >= totalW / 2) return p.val;
      }
      return pairs[pairs.length - 1]?.val ?? 0;
    });
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ss_tot === 0 ? 1 : 1 - ss_res / ss_tot;
  }
}

// ─── Default estimator factories ─────────────────────────────────────────────

function createDefaultClassifier(): StackableClassifier {
  // Minimal logistic regression stub
  let weights: Float64Array | null = null;
  let bias = 0;
  return {
    fit(X: Float64Array[], y: Int32Array) {
      const n = X.length;
      const d = X[0]?.length ?? 0;
      weights = new Float64Array(d);
      const lr = 0.1;
      for (let iter = 0; iter < 100; iter++) {
        for (let i = 0; i < n; i++) {
          const xi = X[i] as Float64Array;
          let logit = bias;
          for (let j = 0; j < d; j++) logit += (weights![j] ?? 0) * (xi[j] ?? 0);
          const pred = 1 / (1 + Math.exp(-logit));
          const err = (y[i] ?? 0) - pred;
          bias += lr * err;
          for (let j = 0; j < d; j++) weights![j]! += lr * err * (xi[j] ?? 0);
        }
      }
      return this;
    },
    predict(X: Float64Array[]) {
      return Int32Array.from(X, (xi) => {
        let logit = bias;
        const d = xi.length;
        for (let j = 0; j < d; j++) logit += (weights![j] ?? 0) * (xi[j] ?? 0);
        return logit >= 0 ? 1 : 0;
      });
    },
  };
}

function createDefaultRegressor(): StackableRegressor {
  let weights: Float64Array | null = null;
  let bias = 0;
  return {
    fit(X: Float64Array[], y: Float64Array) {
      const n = X.length;
      const d = X[0]?.length ?? 0;
      weights = new Float64Array(d);
      const lr = 0.01;
      for (let iter = 0; iter < 200; iter++) {
        for (let i = 0; i < n; i++) {
          const xi = X[i] as Float64Array;
          let pred = bias;
          for (let j = 0; j < d; j++) pred += (weights![j] ?? 0) * (xi[j] ?? 0);
          const err = (y[i] ?? 0) - pred;
          bias += lr * err;
          for (let j = 0; j < d; j++) weights![j]! += lr * err * (xi[j] ?? 0);
        }
      }
      return this;
    },
    predict(X: Float64Array[]) {
      return Float64Array.from(X, (xi) => {
        let pred = bias;
        const d = xi.length;
        for (let j = 0; j < d; j++) pred += (weights![j] ?? 0) * (xi[j] ?? 0);
        return pred;
      });
    },
  };
}
