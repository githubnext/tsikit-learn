/**
 * Radius Neighbors Classifier and Regressor.
 * Mirrors sklearn.neighbors.RadiusNeighborsClassifier / RadiusNeighborsRegressor.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  }
  return Math.sqrt(s);
}

export class RadiusNeighborsClassifier {
  radius: number;
  weights: string;
  outlierLabel: number;

  XTrain_: Float64Array[] | null = null;
  yTrain_: Float64Array | null = null;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      radius?: number;
      weights?: string;
      outlierLabel?: number;
    } = {},
  ) {
    this.radius = options.radius ?? 1.0;
    this.weights = options.weights ?? "uniform";
    this.outlierLabel = options.outlierLabel ?? -1;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    this.classes_ = new Float64Array(
      Array.from(new Set(Array.from(y))).sort((a, b) => a - b),
    );
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.XTrain_ === null || this.yTrain_ === null)
      throw new NotFittedError("RadiusNeighborsClassifier");

    const XTrain = this.XTrain_;
    const yTrain = this.yTrain_;

    return new Float64Array(
      X.map((xi) => {
        const neighbors: { dist: number; label: number }[] = [];
        for (let j = 0; j < XTrain.length; j++) {
          const d = euclidean(xi, XTrain[j] ?? new Float64Array(0));
          if (d <= this.radius) {
            neighbors.push({ dist: d, label: yTrain[j] ?? 0 });
          }
        }

        if (neighbors.length === 0) return this.outlierLabel;

        const votes = new Map<number, number>();
        for (const { dist, label } of neighbors) {
          const w =
            this.weights === "distance" ? (dist > 0 ? 1 / dist : 1e10) : 1;
          votes.set(label, (votes.get(label) ?? 0) + w);
        }

        let bestLabel = 0;
        let bestVotes = Number.NEGATIVE_INFINITY;
        for (const [label, v] of votes) {
          if (v > bestVotes) {
            bestVotes = v;
            bestLabel = label;
          }
        }
        return bestLabel;
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

export class RadiusNeighborsRegressor {
  radius: number;
  weights: string;

  XTrain_: Float64Array[] | null = null;
  yTrain_: Float64Array | null = null;

  constructor(options: { radius?: number; weights?: string } = {}) {
    this.radius = options.radius ?? 1.0;
    this.weights = options.weights ?? "uniform";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.XTrain_ === null || this.yTrain_ === null)
      throw new NotFittedError("RadiusNeighborsRegressor");

    const XTrain = this.XTrain_;
    const yTrain = this.yTrain_;

    return new Float64Array(
      X.map((xi) => {
        let wSum = 0;
        let ySum = 0;
        for (let j = 0; j < XTrain.length; j++) {
          const d = euclidean(xi, XTrain[j] ?? new Float64Array(0));
          if (d <= this.radius) {
            const w = this.weights === "distance" ? (d > 0 ? 1 / d : 1e10) : 1;
            wSum += w;
            ySum += w * (yTrain[j] ?? 0);
          }
        }
        return wSum > 0 ? ySum / wSum : 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}
