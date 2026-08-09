/**
 * Additional clustering algorithms: MiniBatchKMeans, OPTICS.
 * Mirrors sklearn.cluster extras.
 */

import { NotFittedError } from "../exceptions.js";

export class MiniBatchKMeans {
  nClusters: number;
  batchSize: number;
  maxIter: number;
  randomState: number;

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(
    options: {
      nClusters?: number;
      batchSize?: number;
      maxIter?: number;
      randomState?: number;
    } = {},
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.batchSize = options.batchSize ?? 100;
    this.maxIter = options.maxIter ?? 100;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const k = Math.min(this.nClusters, n);

    // Initialize centers with first k points
    let centers = X.slice(0, k).map((row) => row.slice());
    const counts = new Float64Array(k);

    let rng = this.randomState;
    const nextRand = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return rng / 4294967296;
    };

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sample a mini-batch
      const batchSize = Math.min(this.batchSize, n);
      const batchIndices: number[] = [];
      for (let b = 0; b < batchSize; b++) {
        batchIndices.push(Math.floor(nextRand() * n));
      }

      for (const idx of batchIndices) {
        const x = X[idx] ?? new Float64Array(nFeatures);
        // Assign to nearest center
        let nearest = 0;
        let minDist = Number.POSITIVE_INFINITY;
        for (let c = 0; c < k; c++) {
          let dist = 0;
          for (let j = 0; j < nFeatures; j++) {
            dist += ((x[j] ?? 0) - (centers[c]?.[j] ?? 0)) ** 2;
          }
          if (dist < minDist) {
            minDist = dist;
            nearest = c;
          }
        }
        // Update center with learning rate
        counts[nearest] = (counts[nearest] ?? 0) + 1;
        const lr = 1 / (counts[nearest] ?? 1);
        for (let j = 0; j < nFeatures; j++) {
          centers[nearest]![j] = (centers[nearest]?.[j] ?? 0) * (1 - lr) + (x[j] ?? 0) * lr;
        }
      }
    }

    this.clusterCenters_ = centers;
    // Assign labels
    const labels = new Int32Array(n);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      let nearest = 0;
      let minDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          dist += ((X[i]?.[j] ?? 0) - (centers[c]?.[j] ?? 0)) ** 2;
        }
        if (dist < minDist) {
          minDist = dist;
          nearest = c;
        }
      }
      labels[i] = nearest;
      inertia += minDist;
    }
    this.labels_ = labels;
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new NotFittedError("MiniBatchKMeans is not fitted");
    const k = this.clusterCenters_.length;
    const nFeatures = this.clusterCenters_[0]?.length ?? 0;
    const labels = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let nearest = 0;
      let minDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          dist += ((X[i]?.[j] ?? 0) - (this.clusterCenters_[c]?.[j] ?? 0)) ** 2;
        }
        if (dist < minDist) {
          minDist = dist;
          nearest = c;
        }
      }
      labels[i] = nearest;
    }
    return labels;
  }
}

export interface OPTICSOptions {
  minSamples?: number;
  maxEps?: number;
  metric?: "euclidean" | "manhattan";
  clusterMethod?: "xi" | "dbscan";
  eps?: number;
  xi?: number;
}

export class OPTICS {
  minSamples: number;
  maxEps: number;
  metric: "euclidean" | "manhattan";
  eps: number;

  labels_: Int32Array | null = null;
  reachabilityDistances_: Float64Array | null = null;
  coreDistances_: Float64Array | null = null;
  ordering_: Int32Array | null = null;

  constructor(options: OPTICSOptions = {}) {
    this.minSamples = options.minSamples ?? 5;
    this.maxEps = options.maxEps ?? Number.POSITIVE_INFINITY;
    this.metric = options.metric ?? "euclidean";
    this.eps = options.eps ?? Number.POSITIVE_INFINITY;
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

  fit(X: Float64Array[]): this {
    const n = X.length;
    // Compute distances
    const dists: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = this._dist(X[i] ?? new Float64Array(0), X[j] ?? new Float64Array(0));
        dists[i]![j] = d;
        dists[j]![i] = d;
      }
    }

    // Core distances
    const coreDists = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
    for (let i = 0; i < n; i++) {
      const row = dists[i]!.slice().sort((a, b) => a - b);
      const kDist = row[this.minSamples - 1] ?? Number.POSITIVE_INFINITY;
      if (kDist <= this.maxEps) coreDists[i] = kDist;
    }

    // OPTICS ordering
    const reachability = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
    const processed = new Uint8Array(n);
    const ordering: number[] = [];

    // Use simple priority-queue via sorted list
    for (let start = 0; start < n; start++) {
      if (processed[start]) continue;

      const seeds: Array<{ idx: number; dist: number }> = [{ idx: start, dist: 0 }];
      while (seeds.length > 0) {
        seeds.sort((a, b) => a.dist - b.dist);
        const { idx } = seeds.shift()!;
        if (processed[idx]) continue;
        processed[idx] = 1;
        ordering.push(idx);

        if (coreDists[idx] === Number.POSITIVE_INFINITY) continue;
        for (let j = 0; j < n; j++) {
          if (processed[j]) continue;
          const d = dists[idx]?.[j] ?? Number.POSITIVE_INFINITY;
          const newReach = Math.max(coreDists[idx] ?? Number.POSITIVE_INFINITY, d);
          if (newReach < (reachability[j] ?? Number.POSITIVE_INFINITY)) {
            reachability[j] = newReach;
            seeds.push({ idx: j, dist: newReach });
          }
        }
      }
    }

    this.reachabilityDistances_ = reachability;
    this.coreDistances_ = coreDists;
    this.ordering_ = new Int32Array(ordering);

    // DBSCAN-style cluster extraction
    const eps = this.eps;
    const labels = new Int32Array(n).fill(-1);
    let clusterId = -1;
    for (const idx of ordering) {
      if ((reachability[idx] ?? Number.POSITIVE_INFINITY) > eps) {
        if ((coreDists[idx] ?? Number.POSITIVE_INFINITY) <= eps) {
          clusterId++;
          labels[idx] = clusterId;
        }
      } else {
        labels[idx] = clusterId;
      }
    }

    this.labels_ = labels;
    return this;
  }
}
