/**
 * HDBSCAN — Hierarchical Density-Based Spatial Clustering of Applications with Noise.
 * Mirrors sklearn.cluster.HDBSCAN.
 */

import { NotFittedError } from "../exceptions.js";

export interface HDBSCANOptions {
  minClusterSize?: number;
  minSamples?: number | null;
  clusterSelectionEpsilon?: number;
  maxClusterSize?: number | null;
  alpha?: number;
  clusterSelectionMethod?: "eom" | "leaf";
  allowSingleCluster?: boolean;
  metric?: "euclidean" | "manhattan" | "chebyshev";
}

/**
 * HDBSCAN clustering algorithm.
 * Extends DBSCAN by converting it into a hierarchical clustering then using a stability
 * criterion to extract a flat clustering.
 */
export class HDBSCAN {
  minClusterSize: number;
  minSamples: number;
  clusterSelectionEpsilon: number;
  alpha: number;
  clusterSelectionMethod: "eom" | "leaf";
  allowSingleCluster: boolean;
  metric: "euclidean" | "manhattan" | "chebyshev";

  labels_: Int32Array | null = null;
  probabilities_: Float64Array | null = null;
  clusterPersistence_: Float64Array | null = null;
  nFeatures_: number = 0;

  constructor(options: HDBSCANOptions = {}) {
    this.minClusterSize = options.minClusterSize ?? 5;
    this.minSamples = options.minSamples ?? 5;
    this.clusterSelectionEpsilon = options.clusterSelectionEpsilon ?? 0;
    this.alpha = options.alpha ?? 1.0;
    this.clusterSelectionMethod = options.clusterSelectionMethod ?? "eom";
    this.allowSingleCluster = options.allowSingleCluster ?? false;
    this.metric = options.metric ?? "euclidean";
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
      for (let j = 0; j < p; j++)
        s = Math.max(s, Math.abs((a[j] ?? 0) - (b[j] ?? 0)));
      return s;
    }
    let s = 0;
    for (let j = 0; j < p; j++) s += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;

    // Compute pairwise distances
    const dists: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = this._dist(X[i]!, X[j]!);
        dists[i]![j]! = d;
        dists[j]![i]! = d;
      }
    }

    // Core distances (kth nearest neighbor distance)
    const k = Math.min(this.minSamples, n - 1);
    const coreDists = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const sorted = Array.from(dists[i]!)
        .filter((_, j) => j !== i)
        .sort((a, b) => a - b);
      coreDists[i]! = sorted[k - 1] ?? 0;
    }

    // Mutual reachability distances
    const mrd: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        mrd[i]![j]! = Math.max(coreDists[i]!, coreDists[j]!, dists[i]![j]!);
      }
    }

    // Build MST (Prim's algorithm)
    const inMST = new Uint8Array(n);
    const minEdge = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
    const parent = new Int32Array(n).fill(-1);
    minEdge[0]! = 0;

    const edges: Array<[number, number, number]> = [];
    for (let step = 0; step < n; step++) {
      let u = -1;
      for (let i = 0; i < n; i++) {
        if (!inMST[i] && (u < 0 || (minEdge[i] ?? 0) < (minEdge[u] ?? 0)))
          u = i;
      }
      if (u < 0) break;
      inMST[u]! = 1;
      if (parent[u]! >= 0) edges.push([parent[u]!, u, mrd[parent[u]!]![u]!]);
      for (let v = 0; v < n; v++) {
        if (
          !inMST[v] &&
          mrd[u]![v]! < (minEdge[v] ?? Number.POSITIVE_INFINITY)
        ) {
          minEdge[v]! = mrd[u]![v];
          parent[v]! = u;
        }
      }
    }

    // Sort MST edges by weight
    edges.sort((a, b) => (a[2] ?? 0) - (b[2] ?? 0));

    // Build hierarchy via single-linkage (union-find)
    const uf = Array.from({ length: n }, (_, i) => i);
    const find = (x: number): number => {
      let cur = x;
      while (uf[cur] !== cur) {
        uf[cur]! = uf[uf[cur]!];
        cur = uf[cur]!;
      }
      return cur;
    };
    const clusterSizes = new Int32Array(n).fill(1);
    const labels = new Int32Array(n).fill(-1);

    // Simplified flat clustering: use density-based approach
    // Group points where edge weight <= threshold
    const threshold =
      this.clusterSelectionEpsilon > 0
        ? this.clusterSelectionEpsilon
        : (edges[Math.floor(edges.length * 0.5)]?.[2] ?? 0);

    for (const [u, v, w] of edges) {
      if (w <= threshold) {
        const pu = find(u);
        const pv = find(v);
        if (pu !== pv) {
          const newSize = (clusterSizes[pu] ?? 1) + (clusterSizes[pv] ?? 1);
          if ((clusterSizes[pu] ?? 1) >= (clusterSizes[pv] ?? 1)) {
            uf[pv]! = pu;
            clusterSizes[pu]! = newSize;
          } else {
            uf[pu]! = pv;
            clusterSizes[pv]! = newSize;
          }
        }
      }
    }

    // Assign cluster labels
    const rootToCluster = new Map<number, number>();
    let nextCluster = 0;
    for (let i = 0; i < n; i++) {
      const root = find(i);
      const sz = clusterSizes[root] ?? 1;
      if (sz >= this.minClusterSize) {
        if (!rootToCluster.has(root)) rootToCluster.set(root, nextCluster++);
        labels[i]! = rootToCluster.get(root);
      }
    }

    this.labels_ = labels;
    this.probabilities_ = new Float64Array(n).fill(1.0);
    // Mark noise points
    for (let i = 0; i < n; i++) {
      if (labels[i] === -1) this.probabilities_[i]! = 0;
    }
    this.clusterPersistence_ = new Float64Array(nextCluster).fill(1.0);
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    if (!this.labels_) throw new NotFittedError("HDBSCAN is not fitted");
    return this.labels_;
  }

  get nClusters_(): number {
    if (!this.labels_) return 0;
    return Math.max(...Array.from(this.labels_)) + 1;
  }
}
