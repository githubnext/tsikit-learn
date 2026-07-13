/**
 * Spectral biclustering extension — ChessboardBicluster and SpectralCocluster.
 */

export interface BiclusterResult {
  rowIndices: Int32Array[];
  colIndices: Int32Array[];
  biclusterScores: Float64Array;
}

function svdTruncated(A: Float64Array[], k: number): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
  // Power iteration for truncated SVD
  const n = A.length, m = A[0]?.length ?? 0;
  const U: Float64Array[] = [], Vt: Float64Array[] = [], S = new Float64Array(k);

  const Acopy = A.map((row) => new Float64Array(row));
  for (let rank = 0; rank < Math.min(k, Math.min(n, m)); rank++) {
    // Random initialization
    let v = Float64Array.from({ length: m }, () => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
    v = v.map((vi) => vi / Math.max(norm, 1e-12));

    for (let iter = 0; iter < 30; iter++) {
      // u = A * v
      let u = Float64Array.from({ length: n }, (_, i) => (Acopy[i] as Float64Array).reduce((s, aij, j) => s + aij * (v[j] ?? 0), 0));
      const su = Math.sqrt(u.reduce((s, ui) => s + ui * ui, 0));
      u = u.map((ui) => ui / Math.max(su, 1e-12));

      // v = A^T * u
      let vNew = new Float64Array(m);
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) vNew[j]! += (u[i] ?? 0) * ((Acopy[i] as Float64Array)[j] ?? 0);
      const sv = Math.sqrt(vNew.reduce((s, vi) => s + vi * vi, 0));
      vNew = vNew.map((vi) => vi / Math.max(sv, 1e-12));
      v = vNew;
      S[rank] = sv;
    }

    // Compute u = A * v
    const u = Float64Array.from({ length: n }, (_, i) => (Acopy[i] as Float64Array).reduce((s, aij, j) => s + aij * (v[j] ?? 0), 0));
    const sigma = Math.sqrt(u.reduce((s, ui) => s + ui * ui, 0));
    S[rank] = sigma;
    const uNorm = sigma > 0 ? u.map((ui) => ui / sigma) : u;
    U.push(uNorm);
    Vt.push(new Float64Array(v));

    // Deflate
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        (Acopy[i]! as Float64Array)[j]! -= sigma * (uNorm[i] ?? 0) * (v[j] ?? 0);
      }
    }
  }
  return { U, S, Vt };
}

export class SpectralCocluster {
  nClusters: number;
  randomState: number;
  rowLabels_: Int32Array | null = null;
  columnLabels_: Int32Array | null = null;
  nIter_: number = 0;

  constructor(nClusters = 3, randomState = 42) {
    this.nClusters = nClusters;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, m = X[0]?.length ?? 0;

    // Normalize: D_r^{-1/2} * X * D_c^{-1/2}
    const rowSums = Float64Array.from(X, (row) => Math.sqrt(Math.max(row.reduce((s, v) => s + Math.abs(v), 0), 1e-12)));
    const colSums = new Float64Array(m);
    for (const row of X) for (let j = 0; j < m; j++) colSums[j]! += Math.abs(row[j] ?? 0);
    const colSqrt = colSums.map((v) => Math.sqrt(Math.max(v, 1e-12)));

    const Xnorm = X.map((row, i) => Float64Array.from({ length: m }, (_, j) => (row[j] ?? 0) / (rowSums[i] ?? 1) / (colSqrt[j] ?? 1)));

    // Compute SVD
    const k = Math.min(this.nClusters + 1, Math.min(n, m));
    const { U, Vt } = svdTruncated(Xnorm, k);

    // Use singular vectors (skip first) for k-means
    const nVecs = Math.min(this.nClusters, U.length - 1);
    const rowFeatures = Array.from({ length: n }, (_, i) =>
      Float64Array.from({ length: nVecs }, (_, v) => (U[v + 1] ?? new Float64Array(n))[i] ?? 0)
    );
    const colFeatures = Array.from({ length: m }, (_, j) =>
      Float64Array.from({ length: nVecs }, (_, v) => (Vt[v + 1] ?? new Float64Array(m))[j] ?? 0)
    );

    // K-means clustering on row/col features
    this.rowLabels_ = kMeansLabels(rowFeatures, this.nClusters);
    this.columnLabels_ = kMeansLabels(colFeatures, this.nClusters);
    return this;
  }

  getBicluster(idx: number): BiclusterResult {
    if (!this.rowLabels_ || !this.columnLabels_) throw new Error("Not fitted");
    const rowIdx = Int32Array.from(Array.from(this.rowLabels_).flatMap((l, i) => l === idx ? [i] : []));
    const colIdx = Int32Array.from(Array.from(this.columnLabels_).flatMap((l, j) => l === idx ? [j] : []));
    return { rowIndices: [rowIdx], colIndices: [colIdx], biclusterScores: new Float64Array([1.0]) };
  }
}

function kMeansLabels(X: Float64Array[], k: number, maxIter = 50): Int32Array {
  const n = X.length;
  if (n === 0) return new Int32Array(0);
  // Initialize centroids
  let centroids = Array.from({ length: k }, (_, i) => new Float64Array(X[i % n] as Float64Array));
  let labels = new Int32Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    const newLabels = Int32Array.from({ length: n }, (_, i) => {
      let best = 0, bestDist = Number.POSITIVE_INFINITY;
      for (let j = 0; j < k; j++) {
        const dist = (X[i] as Float64Array).reduce((s, v, d) => s + (v - ((centroids[j] as Float64Array)[d] ?? 0)) ** 2, 0);
        if (dist < bestDist) { bestDist = dist; best = j; }
      }
      return best;
    });

    // Update centroids
    centroids = Array.from({ length: k }, (_, j) => {
      const members = Array.from({ length: n }, (__, i) => i).filter((i) => newLabels[i] === j);
      if (members.length === 0) return centroids[j] as Float64Array;
      const p = (X[0] as Float64Array).length;
      return Float64Array.from({ length: p }, (_, d) => members.reduce((s, i) => s + ((X[i] as Float64Array)[d] ?? 0), 0) / members.length);
    });

    if (newLabels.every((l, i) => l === labels[i])) break;
    labels = newLabels;
  }
  return labels;
}

export class ChessboardBicluster {
  nRowClusters: number;
  nColClusters: number;
  randomState: number;
  rowLabels_: Int32Array | null = null;
  columnLabels_: Int32Array | null = null;

  constructor(nRowClusters = 3, nColClusters = 3, randomState = 42) {
    this.nRowClusters = nRowClusters;
    this.nColClusters = nColClusters;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const m = X[0]?.length ?? 0;
    // Iterative bipartite spectral clustering
    const { U, Vt } = svdTruncated(X, Math.min(this.nRowClusters, this.nColClusters) + 1);
    const nVecs = Math.min(this.nRowClusters, U.length);
    const n = X.length;
    const rowFeatures = Array.from({ length: n }, (_, i) => Float64Array.from({ length: nVecs }, (_, v) => (U[v] as Float64Array)[i] ?? 0));
    const colFeatures = Array.from({ length: m }, (_, j) => Float64Array.from({ length: nVecs }, (_, v) => (Vt[v] as Float64Array)[j] ?? 0));
    this.rowLabels_ = kMeansLabels(rowFeatures, this.nRowClusters);
    this.columnLabels_ = kMeansLabels(colFeatures, this.nColClusters);
    return this;
  }
}

export class BlockMatrixBicluster {
  nClusters: number;
  randomState: number;
  rowLabels_: Int32Array | null = null;
  columnLabels_: Int32Array | null = null;

  constructor(nClusters = 3, randomState = 42) {
    this.nClusters = nClusters;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, m = X[0]?.length ?? 0;
    // Simple block clustering: cluster rows and columns independently
    this.rowLabels_ = kMeansLabels(X, this.nClusters);
    const Xt = Array.from({ length: m }, (_, j) => Float64Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0));
    this.columnLabels_ = kMeansLabels(Xt, this.nClusters);
    return this;
  }
}
