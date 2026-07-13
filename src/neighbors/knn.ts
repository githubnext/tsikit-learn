/**
 * K-Nearest Neighbors Classifier and Regressor.
 * Mirrors sklearn.neighbors.KNeighborsClassifier / KNeighborsRegressor.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  }
  return Math.sqrt(s);
}

function manhattan(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  }
  return s;
}

type MetricFn = (a: Float64Array, b: Float64Array) => number;

function getMetric(metric: string): MetricFn {
  if (metric === "manhattan") return manhattan;
  return euclidean;
}

export class KNeighborsClassifier {
  k: number;
  metric: string;
  weights: string;

  XTrain_: Float64Array[] | null = null;
  yTrain_: Float64Array | null = null;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      k?: number;
      nNeighbors?: number;
      metric?: string;
      weights?: string;
    } = {},
  ) {
    this.k = options.k ?? options.nNeighbors ?? 5;
    this.metric = options.metric ?? "euclidean";
    this.weights = options.weights ?? "uniform";
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
      throw new NotFittedError("KNeighborsClassifier");

    const metricFn = getMetric(this.metric);
    const XTrain = this.XTrain_;
    const yTrain = this.yTrain_;
    const k = Math.min(this.k, XTrain.length);

    return new Float64Array(
      X.map((xi) => {
        const dists = XTrain.map((xj, idx) => ({
          dist: metricFn(xi, xj),
          label: yTrain[idx] ?? 0,
        }));
        dists.sort((a, b) => a.dist - b.dist);
        const neighbors = dists.slice(0, k);

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

export class KNeighborsRegressor {
  k: number;
  metric: string;
  weights: string;

  XTrain_: Float64Array[] | null = null;
  yTrain_: Float64Array | null = null;

  constructor(
    options: {
      k?: number;
      nNeighbors?: number;
      metric?: string;
      weights?: string;
    } = {},
  ) {
    this.k = options.k ?? options.nNeighbors ?? 5;
    this.metric = options.metric ?? "euclidean";
    this.weights = options.weights ?? "uniform";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.XTrain_ === null || this.yTrain_ === null)
      throw new NotFittedError("KNeighborsRegressor");

    const metricFn = getMetric(this.metric);
    const XTrain = this.XTrain_;
    const yTrain = this.yTrain_;
    const k = Math.min(this.k, XTrain.length);

    return new Float64Array(
      X.map((xi) => {
        const dists = XTrain.map((xj, idx) => ({
          dist: metricFn(xi, xj),
          y: yTrain[idx] ?? 0,
        }));
        dists.sort((a, b) => a.dist - b.dist);
        const neighbors = dists.slice(0, k);

        let wSum = 0;
        let ySum = 0;
        for (const { dist, y: yVal } of neighbors) {
          const w =
            this.weights === "distance" ? (dist > 0 ? 1 / dist : 1e10) : 1;
          wSum += w;
          ySum += w * yVal;
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
