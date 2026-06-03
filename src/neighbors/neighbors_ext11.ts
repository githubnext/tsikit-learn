/**
 * Neighbor extensions: Locality-sensitive hashing, cover tree, annoy-style ANNS.
 * Mirrors sklearn.neighbors additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Approximate nearest neighbors using random projections (LSH-style). */
export class RandomProjectionANN extends BaseEstimator {
  n_tables: number;
  n_bits: number;
  projections_: Float64Array[][] = [];
  hash_tables_: Map<string, number[]>[] = [];
  X_train_: Float64Array[] = [];

  constructor(params: { n_tables?: number; n_bits?: number } = {}) {
    super();
    this.n_tables = params.n_tables ?? 5;
    this.n_bits = params.n_bits ?? 8;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    this.X_train_ = X;
    this.projections_ = [];
    this.hash_tables_ = [];

    for (let t = 0; t < this.n_tables; t++) {
      const projs = Array.from({ length: this.n_bits }, () => {
        const v = new Float64Array(d);
        let norm = 0;
        for (let f = 0; f < d; f++) { v[f] = (Math.random() - 0.5) * 2; norm += v[f]! ** 2; }
        norm = Math.sqrt(norm) || 1;
        for (let f = 0; f < d; f++) v[f] = (v[f] ?? 0) / norm;
        return v;
      });
      this.projections_.push(projs);

      const table = new Map<string, number[]>();
      for (let i = 0; i < X.length; i++) {
        const key = this._hashRow(X[i]!, projs);
        if (!table.has(key)) table.set(key, []);
        table.get(key)!.push(i);
      }
      this.hash_tables_.push(table);
    }
    return this;
  }

  private _hashRow(x: Float64Array, projs: Float64Array[]): string {
    return projs.map(p => {
      let dot = 0;
      for (let f = 0; f < x.length; f++) dot += (x[f] ?? 0) * (p[f] ?? 0);
      return dot >= 0 ? '1' : '0';
    }).join('');
  }

  kneighbors(X: Float64Array[], n_neighbors = 5): { indices: Int32Array[]; distances: Float64Array[] } {
    const out_idx: Int32Array[] = [];
    const out_dist: Float64Array[] = [];
    for (const query of X) {
      const candidates = new Set<number>();
      for (let t = 0; t < this.n_tables; t++) {
        const key = this._hashRow(query, this.projections_[t]!);
        for (const idx of (this.hash_tables_[t]!.get(key) ?? [])) candidates.add(idx);
      }
      // If not enough candidates, fall back to all
      const pool = candidates.size >= n_neighbors ? [...candidates] : Array.from({ length: this.X_train_.length }, (_, i) => i);
      const dists = pool.map(i => {
        let d = 0;
        for (let f = 0; f < query.length; f++) d += ((query[f] ?? 0) - (this.X_train_[i]?.[f] ?? 0)) ** 2;
        return { i, d: Math.sqrt(d) };
      }).sort((a, b) => a.d - b.d).slice(0, n_neighbors);
      out_idx.push(new Int32Array(dists.map(e => e.i)));
      out_dist.push(new Float64Array(dists.map(e => e.d)));
    }
    return { indices: out_idx, distances: out_dist };
  }
}

/** Radius neighbors graph construction. */
export function radiusNeighborsGraph(
  X: Float64Array[],
  radius: number,
  mode: "connectivity" | "distance" = "connectivity",
): Float64Array[] {
  const n = X.length;
  const graph: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      let d = 0;
      for (let f = 0; f < X[i]!.length; f++) d += ((X[i]?.[f] ?? 0) - (X[j]?.[f] ?? 0)) ** 2;
      d = Math.sqrt(d);
      if (d <= radius) graph[i]![j] = mode === "connectivity" ? 1 : d;
    }
  }
  return graph;
}

/** Nearest centroid classifier. */
export class CentroidClassifier extends BaseEstimator {
  centroids_: Map<number, Float64Array> = new Map();
  classes_: Int32Array = new Int32Array(0);

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))];
    this.classes_ = new Int32Array(classes);
    const sums = new Map<number, { sum: Float64Array; cnt: number }>();
    const d = X[0]?.length ?? 0;
    for (const cls of classes) sums.set(cls, { sum: new Float64Array(d), cnt: 0 });
    for (let i = 0; i < X.length; i++) {
      const cls = y[i] ?? 0;
      const entry = sums.get(cls)!;
      for (let f = 0; f < d; f++) entry.sum[f] = (entry.sum[f] ?? 0) + (X[i]?.[f] ?? 0);
      entry.cnt++;
    }
    for (const [cls, { sum, cnt }] of sums) {
      this.centroids_.set(cls, sum.map(v => v / cnt));
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => {
      let best = 0;
      let bestD = Number.POSITIVE_INFINITY;
      for (const [cls, centroid] of this.centroids_) {
        let d = 0;
        for (let f = 0; f < row.length; f++) d += ((row[f] ?? 0) - (centroid[f] ?? 0)) ** 2;
        if (d < bestD) { bestD = d; best = cls; }
      }
      return best;
    }));
  }
}

/** K-nearest neighbors density estimation. */
export function knnDensityEstimate(
  X_train: Float64Array[],
  X_query: Float64Array[],
  k = 5,
): Float64Array {
  const n = X_train.length;
  const d = X_train[0]?.length ?? 1;
  const out = new Float64Array(X_query.length);
  for (let q = 0; q < X_query.length; q++) {
    const dists = X_train.map((x, i) => {
      let dist = 0;
      for (let f = 0; f < d; f++) dist += ((X_query[q]?.[f] ?? 0) - (x[f] ?? 0)) ** 2;
      return { i, dist: Math.sqrt(dist) };
    }).sort((a, b) => a.dist - b.dist);
    const kthDist = dists[Math.min(k - 1, n - 1)]?.dist ?? 1;
    // Volume of d-ball
    const volBall = (Math.PI ** (d / 2) / gamma(d / 2 + 1)) * kthDist ** d;
    out[q] = k / (n * volBall || 1);
  }
  return out;
}

function gamma(n: number): number {
  if (n === 0.5) return Math.sqrt(Math.PI);
  if (n === 1) return 1;
  if (n === 1.5) return 0.5 * Math.sqrt(Math.PI);
  return n > 0 ? (n - 1) * gamma(n - 1) : 1;
}
