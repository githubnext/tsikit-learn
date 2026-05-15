/**
 * NearestCentroid classifier and NearestNeighbors.
 * Mirrors sklearn.neighbors.NearestCentroid and NearestNeighbors.
 */

import { NotFittedError } from "../exceptions.js";

export interface NearestCentroidOptions {
  metric?: "euclidean" | "manhattan";
  shrinkThreshold?: number | null;
}

/**
 * NearestCentroid — classifies samples by assigning them to the class of the nearest centroid.
 */
export class NearestCentroid {
  metric: "euclidean" | "manhattan";
  shrinkThreshold: number | null;

  centroids_: Float64Array[] | null = null;
  classes_: Int32Array | null = null;
  nFeatureIn_: number = 0;

  constructor(options: NearestCentroidOptions = {}) {
    this.metric = options.metric ?? "euclidean";
    this.shrinkThreshold = options.shrinkThreshold ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeatureIn_ = p;

    const classSet = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classSet);

    this.centroids_ = classSet.map((cls) => {
      const centroid = new Float64Array(p);
      let count = 0;
      for (let i = 0; i < n; i++) {
        if ((y[i] ?? 0) === cls) {
          for (let j = 0; j < p; j++) centroid[j]! += X[i]![j] ?? 0;
          count++;
        }
      }
      if (count > 0) for (let j = 0; j < p; j++) centroid[j]! /= count;
      return centroid;
    });

    // Shrinkage (nearest shrunken centroids)
    if (this.shrinkThreshold !== null && this.shrinkThreshold > 0) {
      const overall = new Float64Array(p);
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) overall[j]! += X[i]![j] ?? 0;
      for (let j = 0; j < p; j++) overall[j]! /= n;

      // Pooled within-class std
      const std = new Float64Array(p);
      for (const cls of classSet) {
        const count = Array.from(y).filter((v) => v === cls).length;
        const centroid = this.centroids_[classSet.indexOf(cls)]!;
        for (let i = 0; i < n; i++) {
          if ((y[i] ?? 0) === cls) {
            for (let j = 0; j < p; j++) std[j]! += ((X[i]![j] ?? 0) - (centroid[j] ?? 0)) ** 2 / count;
          }
        }
      }
      for (let j = 0; j < p; j++) std[j]! = Math.sqrt((std[j] ?? 0) / classSet.length);

      // Shrink each centroid toward overall mean
      for (let c = 0; c < classSet.length; c++) {
        const centroid = this.centroids_[c]!;
        for (let j = 0; j < p; j++) {
          const s = std[j] ?? 1;
          const d = ((centroid[j] ?? 0) - (overall[j] ?? 0)) / (s + 1e-10);
          const shrunken = Math.sign(d) * Math.max(0, Math.abs(d) - this.shrinkThreshold!);
          centroid[j]! = (overall[j] ?? 0) + shrunken * (s + 1e-10);
        }
      }
    }

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.centroids_ || !this.classes_) throw new NotFittedError("NearestCentroid is not fitted");
    const out = new Int32Array(X.length);
    const k = this.classes_.length;

    for (let i = 0; i < X.length; i++) {
      let minDist = Number.POSITIVE_INFINITY;
      let bestClass = this.classes_[0]!;
      for (let c = 0; c < k; c++) {
        const centroid = this.centroids_[c]!;
        let dist = 0;
        if (this.metric === "manhattan") {
          for (let j = 0; j < centroid.length; j++) dist += Math.abs((X[i]![j] ?? 0) - (centroid[j] ?? 0));
        } else {
          for (let j = 0; j < centroid.length; j++) dist += ((X[i]![j] ?? 0) - (centroid[j] ?? 0)) ** 2;
        }
        if (dist < minDist) {
          minDist = dist;
          bestClass = this.classes_[c]!;
        }
      }
      out[i]! = bestClass;
    }
    return out;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if ((pred[i] ?? 0) === (y[i] ?? 0)) correct++;
    return correct / y.length;
  }
}

export interface NearestNeighborsOptions {
  nNeighbors?: number;
  algorithm?: "auto" | "ball_tree" | "kd_tree" | "brute";
  leafSize?: number;
  metric?: "euclidean" | "manhattan" | "chebyshev" | "minkowski";
  p?: number;
}

/**
 * NearestNeighbors — unsupervised learner for implementing neighbor searches.
 */
export class NearestNeighbors {
  nNeighbors: number;
  metric: string;
  p: number;

  private _X: Float64Array[] | null = null;
  nFeatureIn_: number = 0;
  nSamplesIn_: number = 0;

  constructor(options: NearestNeighborsOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.metric = options.metric ?? "euclidean";
    this.p = options.p ?? 2;
  }

  fit(X: Float64Array[]): this {
    this._X = X;
    this.nSamplesIn_ = X.length;
    this.nFeatureIn_ = X[0]?.length ?? 0;
    return this;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    const p = a.length;
    if (this.metric === "manhattan") {
      let s = 0;
      for (let j = 0; j < p; j++) s += Math.abs((a[j] ?? 0) - (b[j] ?? 0));
      return s;
    }
    if (this.metric === "chebyshev") {
      let s = 0;
      for (let j = 0; j < p; j++) s = Math.max(s, Math.abs((a[j] ?? 0) - (b[j] ?? 0)));
      return s;
    }
    let s = 0;
    for (let j = 0; j < p; j++) s += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  kneighbors(X: Float64Array[], nNeighbors?: number): { distances: Float64Array[]; indices: Int32Array[] } {
    if (!this._X) throw new NotFittedError("NearestNeighbors is not fitted");
    const k = nNeighbors ?? this.nNeighbors;
    const nTrain = this._X.length;

    const distances: Float64Array[] = [];
    const indices: Int32Array[] = [];

    for (const xi of X) {
      const dists = new Float64Array(nTrain);
      for (let j = 0; j < nTrain; j++) dists[j]! = this._dist(xi, this._X[j]!);
      const order = Array.from({ length: nTrain }, (_, i) => i).sort((a, b) => (dists[a] ?? 0) - (dists[b] ?? 0));
      const knn = order.slice(0, k);
      distances.push(new Float64Array(knn.map((idx) => dists[idx] ?? 0)));
      indices.push(new Int32Array(knn));
    }

    return { distances, indices };
  }

  radiusNeighbors(X: Float64Array[], radius: number): { distances: Float64Array[]; indices: Int32Array[] } {
    if (!this._X) throw new NotFittedError("NearestNeighbors is not fitted");
    const nTrain = this._X.length;

    const distances: Float64Array[] = [];
    const indices: Int32Array[] = [];

    for (const xi of X) {
      const withinRadius: Array<[number, number]> = [];
      for (let j = 0; j < nTrain; j++) {
        const d = this._dist(xi, this._X[j]!);
        if (d <= radius) withinRadius.push([d, j]);
      }
      withinRadius.sort((a, b) => a[0] - b[0]);
      distances.push(new Float64Array(withinRadius.map(([d]) => d)));
      indices.push(new Int32Array(withinRadius.map(([, idx]) => idx)));
    }

    return { distances, indices };
  }
}
