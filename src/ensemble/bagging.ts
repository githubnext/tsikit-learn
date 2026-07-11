/**
 * BaggingClassifier, BaggingRegressor, VotingClassifier, and AdaBoostClassifier.
 * Mirrors sklearn.ensemble bagging and voting estimators.
 */

import { NotFittedError } from "../exceptions.js";

export interface BaseClassifier {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
}

export interface BaseRegressor {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
}

export interface BaggingClassifierOptions {
  estimator?: BaseClassifier;
  nEstimators?: number;
  maxSamples?: number;
  maxFeatures?: number;
  bootstrap?: boolean;
  randomState?: number;
}

function bootstrapSample(
  X: Float64Array[],
  y: Int32Array,
  size: number,
): [Float64Array[], Int32Array] {
  const Xs: Float64Array[] = [];
  const ys: number[] = [];
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(Math.random() * X.length);
    Xs.push(X[idx]!);
    ys.push(y[idx] ?? 0);
  }
  return [Xs, new Int32Array(ys)];
}

export class BaggingClassifier {
  estimator: BaseClassifier | null;
  nEstimators: number;
  maxSamples: number;
  maxFeatures: number;
  bootstrap: boolean;

  estimators_: BaseClassifier[] = [];
  estimatorsFeatures_: Int32Array[] = [];
  classes_: Int32Array | null = null;

  constructor(
    estimator: BaseClassifier | null = null,
    options: BaggingClassifierOptions = {},
  ) {
    this.estimator = estimator;
    this.nEstimators = options.nEstimators ?? 10;
    this.maxSamples = options.maxSamples ?? 1.0;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.bootstrap = options.bootstrap ?? true;
  }

  private _makeEstimator(): BaseClassifier {
    if (this.estimator) return Object.create(this.estimator) as BaseClassifier;
    throw new Error("No base estimator provided");
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const sampleSize = Math.round(
      typeof this.maxSamples === "number" && this.maxSamples <= 1
        ? n * this.maxSamples
        : this.maxSamples,
    );
    const featureSize = Math.round(
      typeof this.maxFeatures === "number" && this.maxFeatures <= 1
        ? nFeatures * this.maxFeatures
        : this.maxFeatures,
    );

    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));

    this.estimators_ = [];
    this.estimatorsFeatures_ = [];

    for (let e = 0; e < this.nEstimators; e++) {
      // Sample features
      const featIdx: number[] = [];
      const allFeat = Array.from({ length: nFeatures }, (_, i) => i);
      for (let k = 0; k < featureSize; k++) {
        const ri = Math.floor(Math.random() * allFeat.length);
        featIdx.push(allFeat.splice(ri, 1)[0]!);
      }
      const featIdxArr = new Int32Array(featIdx);
      this.estimatorsFeatures_.push(featIdxArr);

      const [Xs, ys] = bootstrapSample(X, y, sampleSize);
      const Xf = Xs.map((row) => {
        const r = new Float64Array(featIdx.length);
        for (let k = 0; k < featIdx.length; k++) r[k] = row[featIdx[k]!] ?? 0;
        return r;
      });

      const est = this._makeEstimator();
      est.fit(Xf, ys);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.estimators_.length || !this.classes_)
      throw new NotFittedError("BaggingClassifier");

    const votes: number[][] = X.map(() =>
      new Array<number>(this.classes_!.length).fill(0),
    );

    for (let e = 0; e < this.estimators_.length; e++) {
      const featIdx = this.estimatorsFeatures_[e]!;
      const Xf = X.map((row) => {
        const r = new Float64Array(featIdx.length);
        for (let k = 0; k < featIdx.length; k++) r[k] = row[featIdx[k]!] ?? 0;
        return r;
      });
      const preds = this.estimators_[e]!.predict(Xf);
      for (let i = 0; i < X.length; i++) {
        const cls = preds[i] ?? 0;
        const ci = Array.from(this.classes_).indexOf(cls);
        if (ci >= 0) votes[i]![ci]! += 1;
      }
    }

    return new Int32Array(
      votes.map((v) => {
        let maxV = -1;
        let maxC = 0;
        for (let k = 0; k < v.length; k++) {
          if ((v[k] ?? 0) > maxV) {
            maxV = v[k] ?? 0;
            maxC = this.classes_![k] ?? 0;
          }
        }
        return maxC;
      }),
    );
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

export interface BaggingRegressorOptions {
  estimator?: BaseRegressor;
  nEstimators?: number;
  maxSamples?: number;
  maxFeatures?: number;
  bootstrap?: boolean;
}

export class BaggingRegressor {
  estimator: BaseRegressor | null;
  nEstimators: number;
  maxSamples: number;
  maxFeatures: number;
  bootstrap: boolean;

  estimators_: BaseRegressor[] = [];
  estimatorsFeatures_: Int32Array[] = [];

  constructor(
    estimator: BaseRegressor | null = null,
    options: BaggingRegressorOptions = {},
  ) {
    this.estimator = estimator;
    this.nEstimators = options.nEstimators ?? 10;
    this.maxSamples = options.maxSamples ?? 1.0;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.bootstrap = options.bootstrap ?? true;
  }

  private _makeEstimator(): BaseRegressor {
    if (this.estimator) return Object.create(this.estimator) as BaseRegressor;
    throw new Error("No base estimator provided");
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const sampleSize = Math.round(n * Math.min(1, this.maxSamples));
    const featureSize = Math.round(nFeatures * Math.min(1, this.maxFeatures));

    this.estimators_ = [];
    this.estimatorsFeatures_ = [];

    for (let e = 0; e < this.nEstimators; e++) {
      const allFeat = Array.from({ length: nFeatures }, (_, i) => i);
      const featIdx: number[] = [];
      for (let k = 0; k < featureSize; k++) {
        const ri = Math.floor(Math.random() * allFeat.length);
        featIdx.push(allFeat.splice(ri, 1)[0]!);
      }
      this.estimatorsFeatures_.push(new Int32Array(featIdx));

      const yNum: number[] = [];
      const Xs: Float64Array[] = [];
      for (let i = 0; i < sampleSize; i++) {
        const idx = Math.floor(Math.random() * n);
        Xs.push(X[idx]!);
        yNum.push(y[idx] ?? 0);
      }
      const Xf = Xs.map((row) => {
        const r = new Float64Array(featIdx.length);
        for (let k = 0; k < featIdx.length; k++) r[k] = row[featIdx[k]!] ?? 0;
        return r;
      });

      const est = this._makeEstimator();
      est.fit(Xf, new Float64Array(yNum));
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimators_.length) throw new NotFittedError("BaggingRegressor");
    const preds = new Float64Array(X.length);
    for (let e = 0; e < this.estimators_.length; e++) {
      const featIdx = this.estimatorsFeatures_[e]!;
      const Xf = X.map((row) => {
        const r = new Float64Array(featIdx.length);
        for (let k = 0; k < featIdx.length; k++) r[k] = row[featIdx[k]!] ?? 0;
        return r;
      });
      const p = this.estimators_[e]!.predict(Xf);
      for (let i = 0; i < X.length; i++)
        preds[i]! += (p[i] ?? 0) / this.nEstimators;
    }
    return preds;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((preds[i] ?? 0) - (y[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ss_tot < 1e-10 ? 1 : 1 - ss_res / ss_tot;
  }
}

export type VotingStrategy = "hard" | "soft";

export interface VotingClassifierOptions {
  voting?: VotingStrategy;
  weights?: number[];
}

export class VotingClassifier {
  estimators: [string, BaseClassifier][];
  voting: VotingStrategy;
  weights: number[] | null;

  estimators_: BaseClassifier[] = [];
  classes_: Int32Array | null = null;
  le_: Map<number, number> = new Map();

  constructor(
    estimators: [string, BaseClassifier][],
    options: VotingClassifierOptions = {},
  ) {
    this.estimators = estimators;
    this.voting = options.voting ?? "hard";
    this.weights = options.weights ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    const sorted = [...classSet].sort((a, b) => a - b);
    this.classes_ = new Int32Array(sorted);
    this.le_ = new Map(sorted.map((c, i) => [c, i]));

    this.estimators_ = this.estimators.map(([, est]) => {
      est.fit(X, y);
      return est;
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.estimators_.length || !this.classes_)
      throw new NotFittedError("VotingClassifier");

    const votes: number[][] = X.map(() =>
      new Array<number>(this.classes_!.length).fill(0),
    );

    for (let e = 0; e < this.estimators_.length; e++) {
      const w = this.weights ? (this.weights[e] ?? 1) : 1;
      const preds = this.estimators_[e]!.predict(X);
      for (let i = 0; i < X.length; i++) {
        const ci = this.le_.get(preds[i] ?? 0);
        if (ci !== undefined) votes[i]![ci]! += w;
      }
    }

    return new Int32Array(
      votes.map((v) => {
        let maxV = -1;
        let maxC = 0;
        for (let k = 0; k < v.length; k++) {
          if ((v[k] ?? 0) > maxV) {
            maxV = v[k] ?? 0;
            maxC = this.classes_![k] ?? 0;
          }
        }
        return maxC;
      }),
    );
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}
