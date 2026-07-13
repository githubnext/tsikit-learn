/**
 * AgglomerativeClustering and MiniBatchKMeans.
 * Mirrors sklearn.cluster.AgglomerativeClustering and MiniBatchKMeans.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

export type Linkage = "ward" | "complete" | "average" | "single";

export interface AgglomerativeClusteringOptions {
  nClusters?: number;
  linkage?: Linkage;
}

export class AgglomerativeClustering {
  nClusters: number;
  linkage: Linkage;

  labels_: Int32Array | null = null;
  nClusters_: number = 0;

  constructor(options: AgglomerativeClusteringOptions = {}) {
    this.nClusters = options.nClusters ?? 2;
    this.linkage = options.linkage ?? "ward";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    // Initialize each point as its own cluster
    const clusters: number[][] = X.map((_, i) => [i]);

    // Distance matrix
    const dist = (a: number[], b: number[]): number => {
      if (this.linkage === "single") {
        let min = Number.POSITIVE_INFINITY;
        for (const i of a)
          for (const j of b) min = Math.min(min, euclidean(X[i]!, X[j]!));
        return min;
      }
      if (this.linkage === "complete") {
        let max = Number.NEGATIVE_INFINITY;
        for (const i of a)
          for (const j of b) max = Math.max(max, euclidean(X[i]!, X[j]!));
        return max;
      }
      // average and ward both use average distance here (simplified)
      let sum = 0;
      for (const i of a) for (const j of b) sum += euclidean(X[i]!, X[j]!);
      return sum / (a.length * b.length);
    };

    while (clusters.length > this.nClusters) {
      let minD = Number.POSITIVE_INFINITY;
      let mergeI = 0;
      let mergeJ = 1;
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const d = dist(clusters[i]!, clusters[j]!);
          if (d < minD) {
            minD = d;
            mergeI = i;
            mergeJ = j;
          }
        }
      }
      clusters[mergeI] = clusters[mergeI]!.concat(clusters[mergeJ]!);
      clusters.splice(mergeJ, 1);
    }

    this.labels_ = new Int32Array(n);
    for (let k = 0; k < clusters.length; k++) {
      for (const idx of clusters[k]!) this.labels_[idx] = k;
    }
    this.nClusters_ = clusters.length;
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_!;
  }
}

export interface MiniBatchKMeansOptions {
  nClusters?: number;
  batchSize?: number;
  maxIter?: number;
  tol?: number;
}

export class MiniBatchKMeans {
  nClusters: number;
  batchSize: number;
  maxIter: number;
  tol: number;

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(options: MiniBatchKMeansOptions = {}) {
    this.nClusters = options.nClusters ?? 8;
    this.batchSize = options.batchSize ?? 100;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
  }

  private _initCenters(X: Float64Array[]): Float64Array[] {
    const indices: number[] = [];
    while (indices.length < this.nClusters) {
      const idx = Math.floor(Math.random() * X.length);
      if (!indices.includes(idx)) indices.push(idx);
    }
    return indices.map((i) => new Float64Array(X[i]!));
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    if (n === 0) throw new Error("Empty input");
    const nFeatures = X[0]?.length ?? 0;

    const centers = this._initCenters(X);
    const counts = new Float64Array(this.nClusters);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const batch: Float64Array[] = [];
      for (let i = 0; i < this.batchSize; i++) {
        batch.push(X[Math.floor(Math.random() * n)]!);
      }

      for (const x of batch) {
        let nearest = 0;
        let minD = Number.POSITIVE_INFINITY;
        for (let k = 0; k < this.nClusters; k++) {
          const d = euclidean(x, centers[k]!);
          if (d < minD) {
            minD = d;
            nearest = k;
          }
        }
        counts[nearest] = (counts[nearest] ?? 0) + 1;
        const lr = 1 / (counts[nearest] ?? 1);
        const c = centers[nearest]!;
        for (let j = 0; j < nFeatures; j++) {
          c[j] = (c[j] ?? 0) * (1 - lr) + (x[j] ?? 0) * lr;
        }
      }
    }

    this.clusterCenters_ = centers;
    this.labels_ = new Int32Array(n);
    this.inertia_ = 0;

    for (let i = 0; i < n; i++) {
      let nearest = 0;
      let minD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const d = euclidean(X[i]!, centers[k]!);
        if (d < minD) {
          minD = d;
          nearest = k;
        }
      }
      this.labels_[i] = nearest;
      this.inertia_ += minD * minD;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new NotFittedError("MiniBatchKMeans");
    const out = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let nearest = 0;
      let minD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const d = euclidean(X[i]!, this.clusterCenters_[k]!);
        if (d < minD) {
          minD = d;
          nearest = k;
        }
      }
      out[i] = nearest;
    }
    return out;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_!;
  }
}
