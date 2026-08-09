/**
 * KNeighborsTransformer and RadiusNeighborsTransformer —
 * transform X into a graph of nearest neighbors (as a sparse-like adjacency).
 *
 * Ports: KNeighborsTransformer, RadiusNeighborsTransformer
 */

import { BaseEstimator } from "../base.js";

/** Sparse adjacency row for a neighbors graph. */
export interface NeighborRow {
  indices: Int32Array;
  distances: Float64Array;
}

function euclidean(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export interface KNeighborsTransformerOptions {
  nNeighbors?: number;
  mode?: "distance" | "connectivity";
  metric?: "euclidean";
}

/**
 * Transforms X into a sparse graph of k-nearest-neighbor distances.
 * The "graph" is returned as an array of NeighborRow objects.
 */
export class KNeighborsTransformer extends BaseEstimator {
  nNeighbors: number;
  mode: "distance" | "connectivity";
  metric: "euclidean";
  private trainX_: Float64Array[] = [];

  constructor(options: KNeighborsTransformerOptions = {}) {
    super();
    this.nNeighbors = options.nNeighbors ?? 5;
    this.mode = options.mode ?? "distance";
    this.metric = options.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    this.trainX_ = X;
    return this;
  }

  transform(X: Float64Array[]): NeighborRow[] {
    const k = this.nNeighbors;
    return X.map((row) => {
      const dists = this.trainX_.map((tr, i) => ({
        idx: i,
        dist: euclidean(row, tr),
      }));
      dists.sort((a, b) => a.dist - b.dist);
      const neighbors = dists.slice(1, k + 1); // exclude self if present
      const indices = new Int32Array(neighbors.map((n) => n.idx));
      const distances =
        this.mode === "distance"
          ? new Float64Array(neighbors.map((n) => n.dist))
          : new Float64Array(k).fill(1);
      return { indices, distances };
    });
  }

  fitTransform(X: Float64Array[]): NeighborRow[] {
    return this.fit(X).transform(X);
  }
}

export interface RadiusNeighborsTransformerOptions {
  radius?: number;
  mode?: "distance" | "connectivity";
  metric?: "euclidean";
}

/**
 * Transforms X into a sparse graph of neighbors within a given radius.
 */
export class RadiusNeighborsTransformer extends BaseEstimator {
  radius: number;
  mode: "distance" | "connectivity";
  metric: "euclidean";
  private trainX_: Float64Array[] = [];

  constructor(options: RadiusNeighborsTransformerOptions = {}) {
    super();
    this.radius = options.radius ?? 1.0;
    this.mode = options.mode ?? "distance";
    this.metric = options.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    this.trainX_ = X;
    return this;
  }

  transform(X: Float64Array[]): NeighborRow[] {
    const r = this.radius;
    return X.map((row) => {
      const neighbors: { idx: number; dist: number }[] = [];
      for (let i = 0; i < this.trainX_.length; i++) {
        const d = euclidean(row, this.trainX_[i] ?? new Float64Array(0));
        if (d <= r && d > 0) {
          neighbors.push({ idx: i, dist: d });
        }
      }
      neighbors.sort((a, b) => a.dist - b.dist);
      const indices = new Int32Array(neighbors.map((n) => n.idx));
      const distances =
        this.mode === "distance"
          ? new Float64Array(neighbors.map((n) => n.dist))
          : new Float64Array(neighbors.length).fill(1);
      return { indices, distances };
    });
  }

  fitTransform(X: Float64Array[]): NeighborRow[] {
    return this.fit(X).transform(X);
  }
}
