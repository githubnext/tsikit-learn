/**
 * Bicluster extensions: SpectralCoClustering, BiclusterMixin utilities.
 */

export class SpectralCoClustering {
  rowLabels_: Int32Array = new Int32Array(0);
  columnLabels_: Int32Array = new Int32Array(0);
  biclusters_: Array<[boolean[], boolean[]]> = [];

  constructor(
    private readonly nClusters = 3,
    private readonly svdMethod: "randomized" | "arpack" = "randomized",
    private readonly seed = 42
  ) {
    void this.svdMethod;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const m = X[0]?.length ?? 1;
    // Normalize: D_row^(-1/2) X D_col^(-1/2)
    const rowSums = X.map((row) => Math.sqrt(Math.max(row.reduce((a, b) => a + b, 0), 1e-10)));
    const colSums = new Float64Array(m);
    for (const row of X) for (let j = 0; j < m; j++) colSums[j] = (colSums[j] ?? 0) + (row[j] ?? 0);
    for (let j = 0; j < m; j++) colSums[j] = Math.sqrt(Math.max(colSums[j] ?? 1, 1e-10));
    const An = X.map((row, i) => new Float64Array(row.map((v, j) => v / Math.max(rowSums[i] ?? 1, 1e-10) / Math.max(colSums[j] ?? 1, 1e-10))));
    // SVD (simplified: power iteration)
    const nVecs = this.nClusters - 1;
    const rng = this._seededRng(this.seed);
    const rowVecs: Float64Array[] = [];
    const colVecs: Float64Array[] = [];
    for (let k = 0; k < nVecs; k++) {
      let v = new Float64Array(m).map(() => rng() - 0.5);
      // Power iteration for singular vector
      for (let iter = 0; iter < 20; iter++) {
        // u = A * v
        const u = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) u[i] = (u[i] ?? 0) + (An[i]?.[j] ?? 0) * (v[j] ?? 0);
        const uNorm = Math.sqrt(u.reduce((a, b) => a + b * b, 0));
        for (let i = 0; i < n; i++) u[i] = (u[i] ?? 0) / Math.max(uNorm, 1e-10);
        // v = A^T * u
        v = new Float64Array(m);
        for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) v[j] = (v[j] ?? 0) + (An[i]?.[j] ?? 0) * (u[i] ?? 0);
        const vNorm = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
        for (let j = 0; j < m; j++) v[j] = (v[j] ?? 0) / Math.max(vNorm, 1e-10);
        // Deflate
        for (const ov of rowVecs) {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += (ov[i] ?? 0) * (u[i] ?? 0);
          for (let i = 0; i < n; i++) u[i] = (u[i] ?? 0) - dot * (ov[i] ?? 0);
        }
      }
      // Compute row vector: An * v
      const rowVec = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) rowVec[i] = (rowVec[i] ?? 0) + (An[i]?.[j] ?? 0) * (v[j] ?? 0);
      rowVecs.push(rowVec);
      colVecs.push(v);
    }
    // K-means on row/col concatenated vectors
    this.rowLabels_ = this._kmeans(rowVecs.length > 0 ? X.map((_, i) => new Float64Array(rowVecs.map((rv) => rv[i] ?? 0))) : X.map(() => new Float64Array(1).fill(0)));
    this.columnLabels_ = this._kmeans(Array.from({ length: m }, (_, j) => new Float64Array(colVecs.map((cv) => cv[j] ?? 0))));
    // Build biclusters
    this.biclusters_ = Array.from({ length: this.nClusters }, (_, k) => {
      const rowMask = Array.from({ length: n }, (__, i) => this.rowLabels_[i] === k);
      const colMask = Array.from({ length: m }, (__, j) => this.columnLabels_[j] === k);
      return [rowMask, colMask] as [boolean[], boolean[]];
    });
    return this;
  }

  private _kmeans(X: Float64Array[]): Int32Array {
    const n = X.length;
    const k = this.nClusters;
    const rng = this._seededRng(this.seed + 1);
    let centers = Array.from({ length: k }, () => X[Math.floor(rng() * n)] ?? new Float64Array(1));
    let labels = new Int32Array(n);
    for (let iter = 0; iter < 50; iter++) {
      const newLabels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Number.POSITIVE_INFINITY;
        for (let c = 0; c < k; c++) {
          let d = 0;
          const xi = X[i]!;
          const ci = centers[c]!;
          for (let f = 0; f < xi.length; f++) d += ((xi[f] ?? 0) - (ci[f] ?? 0)) ** 2;
          if (d < bestD) { bestD = d; best = c; }
        }
        newLabels[i] = best;
      }
      // Update centers
      const nF = X[0]?.length ?? 1;
      const newCenters = Array.from({ length: k }, () => ({ sum: new Float64Array(nF), cnt: 0 }));
      for (let i = 0; i < n; i++) {
        const c = newLabels[i]!;
        newCenters[c]!.cnt++;
        const xi = X[i]!;
        for (let f = 0; f < nF; f++) newCenters[c]!.sum[f] = (newCenters[c]!.sum[f] ?? 0) + (xi[f] ?? 0);
      }
      centers = newCenters.map((nc) => new Float64Array(nc.sum.map((v) => v / Math.max(nc.cnt, 1))));
      const changed = newLabels.some((l, i) => l !== labels[i]);
      labels = newLabels;
      if (!changed) break;
    }
    return labels;
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }

  getBicluster(i: number): [boolean[], boolean[]] {
    return this.biclusters_[i] ?? [[], []];
  }
}

export class SpectralBiclusteringExt {
  rowLabels_: Int32Array = new Int32Array(0);
  columnLabels_: Int32Array = new Int32Array(0);

  constructor(private readonly nClusters: [number, number] | number = [3, 3]) {}

  fit(X: Float64Array[]): this {
    const nRowClusters = Array.isArray(this.nClusters) ? this.nClusters[0]! : this.nClusters;
    const nColClusters = Array.isArray(this.nClusters) ? this.nClusters[1]! : this.nClusters;
    const coClust = new SpectralCoClustering(Math.max(nRowClusters, nColClusters));
    coClust.fit(X);
    // Remap to correct number of clusters
    this.rowLabels_ = new Int32Array(coClust.rowLabels_.map((l) => l % nRowClusters));
    this.columnLabels_ = new Int32Array(coClust.columnLabels_.map((l) => l % nColClusters));
    return this;
  }
}
