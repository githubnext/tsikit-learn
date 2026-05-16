/**
 * FeatureAgglomeration — hierarchical clustering applied to features (columns).
 * Each sample's features are grouped; the representative value (mean/median/max)
 * of each group becomes the transformed feature.
 *
 * Ports: FeatureAgglomeration
 */

import { BaseEstimator } from "../base.js";

export interface FeatureAgglomerationOptions {
  nClusters?: number;
  poolingFunc?: "mean" | "median" | "max" | "min";
  linkage?: "ward" | "complete" | "average" | "single";
}

function columnMean(X: Float64Array[], col: number): number {
  let s = 0;
  for (const row of X) s += row[col] ?? 0;
  return s / X.length;
}

function colDist(X: Float64Array[], a: number, b: number): number {
  const ma = columnMean(X, a);
  const mb = columnMean(X, b);
  return Math.abs(ma - mb);
}

/**
 * Agglomerative (bottom-up) clustering on columns using average-column-value distance.
 * Returns an array mapping each column → cluster index (0-based).
 */
function agglomerateCols(
  X: Float64Array[],
  nClusters: number,
  _linkage: string,
): Int32Array {
  const nFeatures = X[0]?.length ?? 0;
  if (nClusters >= nFeatures) {
    return Int32Array.from({ length: nFeatures }, (_, i) => i);
  }
  // Start: each feature is its own cluster
  const assignments = Int32Array.from({ length: nFeatures }, (_, i) => i);
  let nActive = nFeatures;
  // Track which features belong to each cluster
  const clusters: number[][] = Array.from({ length: nFeatures }, (_, i) => [i]);

  while (nActive > nClusters) {
    // Find two closest clusters (by mean column distance)
    let minDist = Number.POSITIVE_INFINITY;
    let mergeA = -1;
    let mergeB = -1;
    const activeIds = [...new Set(Array.from(assignments))].sort((a, b) => a - b);
    for (let ai = 0; ai < activeIds.length; ai++) {
      for (let bi = ai + 1; bi < activeIds.length; bi++) {
        const ca = activeIds[ai] ?? 0;
        const cb = activeIds[bi] ?? 0;
        const colsA = clusters[ca] ?? [];
        const colsB = clusters[cb] ?? [];
        // average linkage between column groups
        let d = 0;
        let count = 0;
        for (const fa of colsA) {
          for (const fb of colsB) {
            d += colDist(X, fa, fb);
            count++;
          }
        }
        d = count > 0 ? d / count : Number.POSITIVE_INFINITY;
        if (d < minDist) {
          minDist = d;
          mergeA = ca;
          mergeB = cb;
        }
      }
    }
    if (mergeA < 0 || mergeB < 0) break;
    // Merge mergeB into mergeA
    const colsB = clusters[mergeB] ?? [];
    for (const col of colsB) {
      assignments[col] = mergeA;
    }
    clusters[mergeA] = [...(clusters[mergeA] ?? []), ...colsB];
    clusters[mergeB] = [];
    nActive--;
  }
  // Remap cluster IDs to 0..nClusters-1
  const idMap = new Map<number, number>();
  let nextId = 0;
  for (let i = 0; i < assignments.length; i++) {
    const a = assignments[i] ?? 0;
    if (!idMap.has(a)) idMap.set(a, nextId++);
    assignments[i] = idMap.get(a) ?? 0;
  }
  return assignments;
}

/**
 * Cluster features using hierarchical clustering and pool each group.
 */
export class FeatureAgglomeration extends BaseEstimator {
  nClusters: number;
  poolingFunc: "mean" | "median" | "max" | "min";
  linkage: "ward" | "complete" | "average" | "single";

  labels_!: Int32Array;
  nClusters_!: number;

  constructor(options: FeatureAgglomerationOptions = {}) {
    super();
    this.nClusters = options.nClusters ?? 2;
    this.poolingFunc = options.poolingFunc ?? "mean";
    this.linkage = options.linkage ?? "ward";
  }

  fit(X: Float64Array[]): this {
    this.labels_ = agglomerateCols(X, this.nClusters, this.linkage);
    this.nClusters_ = new Set(Array.from(this.labels_)).size;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.labels_ === undefined) throw new Error("Not fitted");
    const k = this.nClusters_;
    return X.map((row) => {
      const groups: number[][] = Array.from({ length: k }, () => []);
      for (let j = 0; j < row.length; j++) {
        const cid = this.labels_[j] ?? 0;
        (groups[cid] ?? []).push(row[j] ?? 0);
      }
      const out = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        const vals = groups[c] ?? [];
        if (vals.length === 0) { out[c] = 0; continue; }
        if (this.poolingFunc === "mean") {
          out[c] = vals.reduce((a, b) => a + b, 0) / vals.length;
        } else if (this.poolingFunc === "median") {
          const s = [...vals].sort((a, b) => a - b);
          const m = Math.floor(s.length / 2);
          out[c] = s.length % 2 === 0
            ? ((s[m - 1] ?? 0) + (s[m] ?? 0)) / 2
            : (s[m] ?? 0);
        } else if (this.poolingFunc === "max") {
          out[c] = Math.max(...vals);
        } else {
          out[c] = Math.min(...vals);
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  /** Reconstruct original shape from reduced representation. */
  inverseTransform(Xred: Float64Array[]): Float64Array[] {
    if (this.labels_ === undefined) throw new Error("Not fitted");
    const nFeatures = this.labels_.length;
    return Xred.map((row) => {
      const out = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        out[j] = row[this.labels_[j] ?? 0] ?? 0;
      }
      return out;
    });
  }
}
