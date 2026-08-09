/**
 * Additional neighbors: RadiusNeighborsClassifier, RadiusNeighborsRegressor.
 * Mirrors sklearn.neighbors extensions.
 */

import { NotFittedError } from "../exceptions.js";

export class RadiusNeighborsClassifier {
  radius: number;
  metric: "euclidean" | "manhattan";
  weights: "uniform" | "distance";
  outlierLabel: number | null;

  private XTrain_: Float64Array[] | null = null;
  private yTrain_: Int32Array | null = null;
  classes_: Int32Array | null = null;

  constructor(
    options: {
      radius?: number;
      metric?: "euclidean" | "manhattan";
      weights?: "uniform" | "distance";
      outlierLabel?: number | null;
    } = {},
  ) {
    this.radius = options.radius ?? 1.0;
    this.metric = options.metric ?? "euclidean";
    this.weights = options.weights ?? "uniform";
    this.outlierLabel = options.outlierLabel ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    this.classes_ = new Int32Array(Array.from(new Set(Array.from(y))).sort((a, b) => a - b));
    return this;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    if (this.metric === "manhattan") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
      return s;
    }
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.XTrain_ || !this.yTrain_) throw new NotFittedError("RadiusNeighborsClassifier is not fitted");
    const out = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const neighbors: Array<{ dist: number; label: number }> = [];
      for (let j = 0; j < this.XTrain_.length; j++) {
        const d = this._dist(X[i] ?? new Float64Array(0), this.XTrain_[j] ?? new Float64Array(0));
        if (d <= this.radius) {
          neighbors.push({ dist: d, label: this.yTrain_[j] ?? 0 });
        }
      }

      if (neighbors.length === 0) {
        out[i] = this.outlierLabel ?? (this.classes_?.[0] ?? 0);
        continue;
      }

      const votes = new Map<number, number>();
      for (const nb of neighbors) {
        const w = this.weights === "distance" ? (nb.dist === 0 ? 1e10 : 1 / nb.dist) : 1;
        votes.set(nb.label, (votes.get(nb.label) ?? 0) + w);
      }
      let bestLabel = 0;
      let bestVote = -1;
      for (const [label, vote] of votes) {
        if (vote > bestVote) {
          bestVote = vote;
          bestLabel = label;
        }
      }
      out[i] = bestLabel;
    }
    return out;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if ((y[i] ?? 0) === (pred[i] ?? 0)) correct++;
    }
    return correct / y.length;
  }
}

export class RadiusNeighborsRegressor {
  radius: number;
  metric: "euclidean" | "manhattan";
  weights: "uniform" | "distance";

  private XTrain_: Float64Array[] | null = null;
  private yTrain_: Float64Array | null = null;

  constructor(
    options: {
      radius?: number;
      metric?: "euclidean" | "manhattan";
      weights?: "uniform" | "distance";
    } = {},
  ) {
    this.radius = options.radius ?? 1.0;
    this.metric = options.metric ?? "euclidean";
    this.weights = options.weights ?? "uniform";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    return this;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    if (this.metric === "manhattan") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
      return s;
    }
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.XTrain_ || !this.yTrain_) throw new NotFittedError("RadiusNeighborsRegressor is not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let wSum = 0;
      let yWSum = 0;
      for (let j = 0; j < this.XTrain_.length; j++) {
        const d = this._dist(X[i] ?? new Float64Array(0), this.XTrain_[j] ?? new Float64Array(0));
        if (d <= this.radius) {
          const w = this.weights === "distance" ? (d === 0 ? 1e10 : 1 / d) : 1;
          wSum += w;
          yWSum += w * (this.yTrain_[j] ?? 0);
        }
      }
      out[i] = wSum > 0 ? yWSum / wSum : 0;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= y.length;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
