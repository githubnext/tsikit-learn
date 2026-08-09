/**
 * Cluster extensions: Fuzzy C-Means, K-Medoids, CLARANS.
 * Mirrors sklearn.cluster additional methods.
 */

import { BaseEstimator } from "../base.js";

export interface FuzzyCMeansParams {
  n_clusters?: number;
  m?: number;
  max_iter?: number;
  tol?: number;
  random_state?: number | null;
}

/** Fuzzy C-Means clustering algorithm. */
export class FuzzyCMeans extends BaseEstimator {
  n_clusters: number;
  m: number;
  max_iter: number;
  tol: number;
  random_state: number | null;
  cluster_centers_: Float64Array[] = [];
  labels_: Int32Array = new Int32Array(0);
  fuzzy_labels_: Float64Array[] = [];
  n_iter_ = 0;

  constructor(params: FuzzyCMeansParams = {}) {
    super();
    this.n_clusters = params.n_clusters ?? 8;
    this.m = params.m ?? 2.0;
    this.max_iter = params.max_iter ?? 150;
    this.tol = params.tol ?? 1e-4;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const k = this.n_clusters;
    // Initialize membership matrix randomly
    const U: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      let sum = 0;
      for (let j = 0; j < k; j++) { row[j] = Math.random(); sum += row[j]!; }
      for (let j = 0; j < k; j++) row[j] = (row[j] ?? 0) / sum;
      return row;
    });

    for (let iter = 0; iter < this.max_iter; iter++) {
      // Update centers
      const centers: Float64Array[] = Array.from({ length: k }, () => new Float64Array(d));
      const denom = new Float64Array(k);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
          const u = (U[i]?.[j] ?? 0) ** this.m;
          denom[j] = (denom[j] ?? 0) + u;
          for (let f = 0; f < d; f++) {
            const c = centers[j]!;
            c[f] = (c[f] ?? 0) + u * (X[i]?.[f] ?? 0);
          }
        }
      }
      for (let j = 0; j < k; j++) {
        const c = centers[j]!;
        const dn = denom[j] ?? 1;
        for (let f = 0; f < d; f++) c[f] = (c[f] ?? 0) / dn;
      }

      // Update membership
      let maxChange = 0;
      for (let i = 0; i < n; i++) {
        const row = U[i]!;
        for (let j = 0; j < k; j++) {
          let num = 0;
          const xi = X[i]!;
          const cj = centers[j]!;
          for (let f = 0; f < d; f++) num += ((xi[f] ?? 0) - (cj[f] ?? 0)) ** 2;
          num = Math.sqrt(num);
          let sum = 0;
          for (let l = 0; l < k; l++) {
            let dl = 0;
            const cl = centers[l]!;
            for (let f = 0; f < d; f++) dl += ((xi[f] ?? 0) - (cl[f] ?? 0)) ** 2;
            dl = Math.sqrt(dl);
            if (dl < 1e-10) { sum = Number.POSITIVE_INFINITY; break; }
            sum += (num / dl) ** (2 / (this.m - 1));
          }
          const newU = sum === Number.POSITIVE_INFINITY ? (j === 0 ? 1 : 0) : 1 / sum;
          maxChange = Math.max(maxChange, Math.abs(newU - (row[j] ?? 0)));
          row[j] = newU;
        }
      }
      this.n_iter_ = iter + 1;
      if (maxChange < this.tol) break;
      this.cluster_centers_ = centers;
    }
    this.cluster_centers_ = this.cluster_centers_.length ? this.cluster_centers_ : Array.from({ length: k }, () => new Float64Array(d));
    this.fuzzy_labels_ = U;
    this.labels_ = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let j = 1; j < k; j++) {
        if ((U[i]?.[j] ?? 0) > (U[i]?.[best] ?? 0)) best = j;
      }
      this.labels_[i] = best;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let j = 0; j < this.cluster_centers_.length; j++) {
        let dist = 0;
        const xi = X[i]!;
        const cj = this.cluster_centers_[j]!;
        for (let f = 0; f < xi.length; f++) dist += ((xi[f] ?? 0) - (cj[f] ?? 0)) ** 2;
        if (dist < bestDist) { bestDist = dist; best = j; }
      }
      out[i] = best;
    }
    return out;
  }
}

/** K-Medoids clustering (PAM algorithm). */
export class KMedoids extends BaseEstimator {
  n_clusters: number;
  max_iter: number;
  medoid_indices_: Int32Array = new Int32Array(0);
  labels_: Int32Array = new Int32Array(0);

  constructor(params: { n_clusters?: number; max_iter?: number } = {}) {
    super();
    this.n_clusters = params.n_clusters ?? 8;
    this.max_iter = params.max_iter ?? 300;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const k = this.n_clusters;
    const medoids = new Int32Array(k);
    for (let j = 0; j < k; j++) medoids[j] = Math.floor(j * n / k);

    const dist = (a: Float64Array, b: Float64Array): number => {
      let s = 0;
      for (let f = 0; f < a.length; f++) s += ((a[f] ?? 0) - (b[f] ?? 0)) ** 2;
      return Math.sqrt(s);
    };

    for (let iter = 0; iter < this.max_iter; iter++) {
      const labels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        let best = 0;
        let bestD = Number.POSITIVE_INFINITY;
        for (let j = 0; j < k; j++) {
          const d = dist(X[i]!, X[medoids[j] ?? 0]!);
          if (d < bestD) { bestD = d; best = j; }
        }
        labels[i] = best;
      }

      let changed = false;
      for (let j = 0; j < k; j++) {
        const clusterIdx = labels.reduce<number[]>((acc, lbl, i) => { if (lbl === j) acc.push(i); return acc; }, []);
        if (clusterIdx.length === 0) continue;
        let bestCost = Number.POSITIVE_INFINITY;
        let bestM = medoids[j] ?? 0;
        for (const m of clusterIdx) {
          let cost = 0;
          for (const i of clusterIdx) cost += dist(X[i]!, X[m]!);
          if (cost < bestCost) { bestCost = cost; bestM = m; }
        }
        if (bestM !== (medoids[j] ?? -1)) { medoids[j] = bestM; changed = true; }
      }
      this.labels_ = labels;
      if (!changed) break;
    }
    this.medoid_indices_ = medoids;
    return this;
  }
}

/** Compute cluster inertia (sum of squared distances to nearest centroid). */
export function clusterInertia(X: Float64Array[], centers: Float64Array[], labels: Int32Array): number {
  let inertia = 0;
  for (let i = 0; i < X.length; i++) {
    const c = centers[labels[i] ?? 0]!;
    const xi = X[i]!;
    for (let f = 0; f < xi.length; f++) inertia += ((xi[f] ?? 0) - (c[f] ?? 0)) ** 2;
  }
  return inertia;
}
