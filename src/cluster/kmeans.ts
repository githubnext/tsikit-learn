/**
 * KMeans and DBSCAN clustering.
 * Mirrors sklearn.cluster.KMeans and DBSCAN.
 */

import { NotFittedError } from "../exceptions.js";

function euclideanSq(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  }
  return s;
}

function euclidean(a: Float64Array, b: Float64Array): number {
  return Math.sqrt(euclideanSq(a, b));
}

export class KMeans {
  nClusters: number;
  maxIter: number;
  tol: number;
  nInit: number;

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(
    options: {
      nClusters?: number;
      maxIter?: number;
      tol?: number;
      nInit?: number;
    } = {},
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-4;
    this.nInit = options.nInit ?? 10;
  }

  private _kmeanspp(X: Float64Array[], k: number): Float64Array[] {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const centers: Float64Array[] = [];

    // Pick first center randomly
    centers.push(new Float64Array(X[Math.floor(Math.random() * n)] ?? new Float64Array(p)));

    for (let c = 1; c < k; c++) {
      const dists = X.map((xi) => {
        let minD = Number.POSITIVE_INFINITY;
        for (const center of centers) {
          const d = euclideanSq(xi, center);
          if (d < minD) minD = d;
        }
        return minD;
      });
      const totalDist = dists.reduce((a, b) => a + b, 0);
      let rand = Math.random() * totalDist;
      let selected = 0;
      for (let i = 0; i < n; i++) {
        rand -= dists[i] ?? 0;
        if (rand <= 0) {
          selected = i;
          break;
        }
      }
      centers.push(new Float64Array(X[selected] ?? new Float64Array(p)));
    }
    return centers;
  }

  private _run(
    X: Float64Array[],
    k: number,
  ): { centers: Float64Array[]; labels: Int32Array; inertia: number } {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    let centers = this._kmeanspp(X, k);
    const labels = new Int32Array(n);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Assignment step
      for (let i = 0; i < n; i++) {
        let minDist = Number.POSITIVE_INFINITY;
        let minIdx = 0;
        for (let c = 0; c < centers.length; c++) {
          const d = euclideanSq(X[i] ?? new Float64Array(p), centers[c] ?? new Float64Array(p));
          if (d < minDist) {
            minDist = d;
            minIdx = c;
          }
        }
        labels[i] = minIdx;
      }

      // Update step
      const newCenters: Float64Array[] = Array.from({ length: k }, () => new Float64Array(p));
      const counts = new Int32Array(k);
      for (let i = 0; i < n; i++) {
        const c = labels[i] ?? 0;
        counts[c] = (counts[c] ?? 0) + 1;
        const xi = X[i] ?? new Float64Array(p);
        const center = newCenters[c] ?? new Float64Array(p);
        for (let j = 0; j < p; j++) {
          center[j] = (center[j] ?? 0) + (xi[j] ?? 0);
        }
      }

      let maxShift = 0;
      for (let c = 0; c < k; c++) {
        const cnt = counts[c] ?? 0;
        const center = newCenters[c] ?? new Float64Array(p);
        if (cnt > 0) {
          for (let j = 0; j < p; j++) {
            center[j] = (center[j] ?? 0) / cnt;
          }
        } else {
          // Re-initialize empty cluster to a random point
          const randIdx = Math.floor(Math.random() * n);
          newCenters[c] = new Float64Array(X[randIdx] ?? new Float64Array(p));
        }
        const shift = euclideanSq(centers[c] ?? new Float64Array(p), newCenters[c] ?? new Float64Array(p));
        if (shift > maxShift) maxShift = shift;
      }
      centers = newCenters;
      if (maxShift < this.tol ** 2) break;
    }

    // Compute inertia
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      inertia += euclideanSq(X[i] ?? new Float64Array(p), centers[labels[i] ?? 0] ?? new Float64Array(p));
    }

    return { centers, labels, inertia };
  }

  fit(X: Float64Array[]): this {
    const k = Math.min(this.nClusters, X.length);
    let best: ReturnType<typeof this._run> | null = null;

    for (let init = 0; init < this.nInit; init++) {
      const result = this._run(X, k);
      if (best === null || result.inertia < best.inertia) {
        best = result;
      }
    }

    this.clusterCenters_ = best?.centers ?? [];
    this.labels_ = best?.labels ?? new Int32Array(X.length);
    this.inertia_ = best?.inertia ?? 0;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this.clusterCenters_ === null) throw new NotFittedError("KMeans");
    const centers = this.clusterCenters_;
    const p = (centers[0] ?? new Float64Array(0)).length;
    return new Int32Array(
      X.map((xi) => {
        let minDist = Number.POSITIVE_INFINITY;
        let minIdx = 0;
        for (let c = 0; c < centers.length; c++) {
          const d = euclideanSq(xi, centers[c] ?? new Float64Array(p));
          if (d < minDist) {
            minDist = d;
            minIdx = c;
          }
        }
        return minIdx;
      }),
    );
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ as Int32Array;
  }

  score(X: Float64Array[]): number {
    return -this._computeInertia(X, this.clusterCenters_ ?? []);
  }

  private _computeInertia(X: Float64Array[], centers: Float64Array[]): number {
    const p = (centers[0] ?? new Float64Array(0)).length;
    let inertia = 0;
    for (const xi of X) {
      let minDist = Number.POSITIVE_INFINITY;
      for (const c of centers) {
        const d = euclideanSq(xi, c.length ? c : new Float64Array(p));
        if (d < minDist) minDist = d;
      }
      inertia += minDist;
    }
    return inertia;
  }
}

export class DBSCAN {
  eps: number;
  minSamples: number;
  metric: string;

  labels_: Int32Array | null = null;
  coreIndices_: Int32Array | null = null;

  constructor(
    options: {
      eps?: number;
      minSamples?: number;
      metric?: string;
    } = {},
  ) {
    this.eps = options.eps ?? 0.5;
    this.minSamples = options.minSamples ?? 5;
    this.metric = options.metric ?? "euclidean";
  }

  fitPredict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const labels = new Int32Array(n).fill(-2); // -2 = unvisited, -1 = noise
    const clusterId = 0;
    const coreIndices: number[] = [];

    function getNeighbors(idx: number): number[] {
      const neighbors: number[] = [];
      const xi = X[idx] ?? new Float64Array(0);
      for (let j = 0; j < n; j++) {
        if (euclidean(xi, X[j] ?? new Float64Array(0)) <= 0.5) {
          // placeholder - use eps below
        }
      }
      return neighbors;
    }
    void getNeighbors; // suppress unused warning

    const eps = this.eps;
    const minSamples = this.minSamples;

    function neighbors(idx: number): number[] {
      const xi = X[idx] ?? new Float64Array(0);
      const result: number[] = [];
      for (let j = 0; j < n; j++) {
        if (euclidean(xi, X[j] ?? new Float64Array(0)) <= eps) {
          result.push(j);
        }
      }
      return result;
    }

    for (let i = 0; i < n; i++) {
      if (labels[i] !== -2) continue;
      const nb = neighbors(i);
      if (nb.length < minSamples) {
        labels[i] = -1;
        continue;
      }

      coreIndices.push(i);
      labels[i] = clusterId;
      const queue = [...nb.filter((j) => j !== i)];

      while (queue.length > 0) {
        const j = queue.shift() as number;
        if (labels[j] === -1) {
          labels[j] = clusterId;
        }
        if (labels[j] !== -2) continue;
        labels[j] = clusterId;
        const jNb = neighbors(j);
        if (jNb.length >= minSamples) {
          coreIndices.push(j);
          for (const k of jNb) {
            if (labels[k] === -2 || labels[k] === -1) {
              queue.push(k);
            }
          }
        }
      }
      clusterId++;
    }

    // Fix any remaining unvisited (noise)
    for (let i = 0; i < n; i++) {
      if (labels[i] === -2) labels[i] = -1;
    }

    this.labels_ = labels;
    this.coreIndices_ = new Int32Array(coreIndices);
    return labels;
  }

  fit(X: Float64Array[]): this {
    this.fitPredict(X);
    return this;
  }
}
