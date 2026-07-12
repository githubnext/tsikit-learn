/**
 * Biclustering algorithms: SpectralBiclustering and SpectralCoclustering.
 * Port of sklearn.cluster.bicluster
 */

import { NotFittedError } from "../exceptions.js";

function svd2(
  matrix: Float64Array[],
  nComponents: number,
): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
  const m = matrix.length;
  const n = matrix[0]?.length ?? 0;
  const k = Math.min(nComponents, Math.min(m, n));
  const U: Float64Array[] = Array.from(
    { length: m },
    () => new Float64Array(k),
  );
  const S = new Float64Array(k);
  const Vt: Float64Array[] = Array.from(
    { length: k },
    () => new Float64Array(n),
  );
  for (let c = 0; c < k; c++) {
    const v = new Float64Array(n);
    v[c % n] = 1;
    for (let _iter = 0; _iter < 30; _iter++) {
      const u = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++)
          u[i]! += (matrix[i]?.[j] ?? 0) * (v[j] ?? 0);
      }
      const newV = new Float64Array(n);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++)
          newV[j]! += (matrix[i]?.[j] ?? 0) * (u[i] ?? 0);
      }
      let norm = 0;
      for (let j = 0; j < n; j++) norm += (newV[j] ?? 0) ** 2;
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let j = 0; j < n; j++) v[j] = (newV[j] ?? 0) / norm;
    }
    const u = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) u[i]! += (matrix[i]?.[j] ?? 0) * (v[j] ?? 0);
    }
    let sigma = 0;
    for (let i = 0; i < m; i++) sigma += (u[i] ?? 0) ** 2;
    sigma = Math.sqrt(sigma);
    S[c] = sigma;
    if (sigma > 1e-12) {
      for (let i = 0; i < m; i++) U[i]![c] = (u[i] ?? 0) / sigma;
    }
    for (let j = 0; j < n; j++) Vt[c]![j] = v[j] ?? 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        matrix[i]![j] =
          (matrix[i]?.[j] ?? 0) - (U[i]?.[c] ?? 0) * sigma * (Vt[c]?.[j] ?? 0);
      }
    }
  }
  return { U, S, Vt };
}

function kmeansSimple(X: Float64Array[], k: number, maxIter = 100): Int32Array {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const labels = new Int32Array(n);
  const centers: Float64Array[] = Array.from({ length: k }, (_, i) =>
    (X[i % n] ?? new Float64Array(d)).slice(),
  );
  for (let _iter = 0; _iter < maxIter; _iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let j = 0; j < k; j++) {
        let dist = 0;
        for (let l = 0; l < d; l++) {
          const diff = (X[i]?.[l] ?? 0) - (centers[j]?.[l] ?? 0);
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          best = j;
        }
      }
      if (labels[i] !== best) {
        labels[i] = best;
        changed = true;
      }
    }
    if (!changed) break;
    const counts = new Int32Array(k);
    for (let j = 0; j < k; j++) centers[j] = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const c = labels[i]!;
      counts[c]!++;
      for (let l = 0; l < d; l++) centers[c]![l]! += X[i]?.[l] ?? 0;
    }
    for (let j = 0; j < k; j++) {
      if ((counts[j] ?? 0) > 0) {
        for (let l = 0; l < d; l++) centers[j]![l]! /= counts[j];
      }
    }
  }
  return labels;
}

export interface SpectralBiclusteringParams {
  nClusters?: number | [number, number];
  method?: "bistochastic" | "scale" | "log";
  nComponents?: number;
  nInit?: number;
}

/** Spectral biclustering. Port of sklearn.cluster.SpectralBiclustering */
export class SpectralBiclustering {
  nClusters: number | [number, number];
  method: string;
  nComponents: number;
  nInit: number;
  rowLabels_?: Int32Array;
  columnLabels_?: Int32Array;
  biclusters_?: [Int32Array, Int32Array][];

  constructor(params: SpectralBiclusteringParams = {}) {
    this.nClusters = params.nClusters ?? 3;
    this.method = params.method ?? "bistochastic";
    this.nComponents = params.nComponents ?? 6;
    this.nInit = params.nInit ?? 10;
  }

  fit(X: Float64Array[]): this {
    const nRows = X.length;
    const nCols = X[0]?.length ?? 0;
    const [nRowClusters, nColClusters] = Array.isArray(this.nClusters)
      ? this.nClusters
      : [this.nClusters, this.nClusters];
    const normalized = X.map((row) => row.slice());
    const k = Math.min(this.nComponents, Math.min(nRows, nCols));
    const { U, Vt } = svd2(normalized, k);
    const rowVecs = U.slice(0, nRows);
    const colVecs = Array.from({ length: nCols }, (_, j) => {
      const v = new Float64Array(k);
      for (let c = 0; c < k; c++) v[c] = Vt[c]?.[j] ?? 0;
      return v;
    });
    this.rowLabels_ = kmeansSimple(rowVecs, nRowClusters, 100);
    this.columnLabels_ = kmeansSimple(colVecs, nColClusters, 100);
    this.biclusters_ = [];
    for (let r = 0; r < nRowClusters; r++) {
      for (let c = 0; c < nColClusters; c++) {
        const rowIdx = Array.from({ length: nRows }, (_, i) => i).filter(
          (i) => this.rowLabels_![i] === r,
        );
        const colIdx = Array.from({ length: nCols }, (_, j) => j).filter(
          (j) => this.columnLabels_![j] === c,
        );
        this.biclusters_.push([new Int32Array(rowIdx), new Int32Array(colIdx)]);
      }
    }
    return this;
  }

  getBicluster(i: number): [Int32Array, Int32Array] {
    if (!this.biclusters_) throw new NotFittedError("SpectralBiclustering");
    return this.biclusters_[i]!;
  }
}

export interface SpectralCoclusteringParams {
  nClusters?: number;
  nSvdVecs?: number | null;
  nInit?: number;
}

/** Spectral co-clustering. Port of sklearn.cluster.SpectralCoclustering */
export class SpectralCoclustering {
  nClusters: number;
  nInit: number;
  rowLabels_?: Int32Array;
  columnLabels_?: Int32Array;
  biclusters_?: [Int32Array, Int32Array][];

  constructor(params: SpectralCoclusteringParams = {}) {
    this.nClusters = params.nClusters ?? 3;
    this.nInit = params.nInit ?? 10;
  }

  fit(X: Float64Array[]): this {
    const nRows = X.length;
    const nCols = X[0]?.length ?? 0;
    const k = this.nClusters;
    const rowSums = new Float64Array(nRows);
    const colSums = new Float64Array(nCols);
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < nCols; j++) {
        rowSums[i]! += X[i]?.[j] ?? 0;
        colSums[j]! += X[i]?.[j] ?? 0;
      }
    }
    const normalized = X.map((row, i) => {
      const nr = new Float64Array(nCols);
      const rs = Math.sqrt(rowSums[i]! || 1);
      for (let j = 0; j < nCols; j++) {
        const cs = Math.sqrt(colSums[j]! || 1);
        nr[j] = (row[j] ?? 0) / (rs * cs);
      }
      return nr;
    });
    const { U, Vt } = svd2(normalized, k + 1);
    const rowVecs = U.slice(0, nRows).map((u) => u.slice(1));
    const colVecs = Array.from({ length: nCols }, (_, j) => {
      const v = new Float64Array(k);
      for (let c = 1; c <= k; c++) v[c - 1] = Vt[c]?.[j] ?? 0;
      return v;
    });
    this.rowLabels_ = kmeansSimple(rowVecs, k, 100);
    this.columnLabels_ = kmeansSimple(colVecs, k, 100);
    this.biclusters_ = [];
    for (let c = 0; c < k; c++) {
      const rowIdx = Array.from({ length: nRows }, (_, i) => i).filter(
        (i) => this.rowLabels_![i] === c,
      );
      const colIdx = Array.from({ length: nCols }, (_, j) => j).filter(
        (j) => this.columnLabels_![j] === c,
      );
      this.biclusters_.push([new Int32Array(rowIdx), new Int32Array(colIdx)]);
    }
    return this;
  }

  getBicluster(i: number): [Int32Array, Int32Array] {
    if (!this.biclusters_) throw new NotFittedError("SpectralCoclustering");
    return this.biclusters_[i]!;
  }
}
