/**
 * Cluster extensions: Spectral Clustering with Nystrom, Gaussian Mixture (diagonal), CLARA
 */

export class SpectralBiClusteringExt {
  private rowLabels_: Int32Array = new Int32Array(0);
  private colLabels_: Int32Array = new Int32Array(0);
  private fitted_ = false;

  constructor(private nClusters: number = 2, private randomState: number = 42) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    // Normalize: D_r^{-1/2} X D_c^{-1/2}
    const rowSums = X.map(row => Math.sqrt(row.reduce((s, v) => s + Math.abs(v), 0) + 1e-10));
    const colSums = new Float64Array(p);
    for (let j = 0; j < p; j++) colSums[j] = Math.sqrt(X.reduce((s, row) => s + Math.abs(row[j] ?? 0), 0) + 1e-10);
    const Xn = X.map((row, i) => new Float64Array(row.map((v, j) => v / (rowSums[i]! * (colSums[j] ?? 1)))));

    // SVD (power iteration for first few singular vectors)
    const { U, V } = this._svd(Xn, this.nClusters, n, p);

    // K-means on row/col singular vectors
    this.rowLabels_ = this._kmeans(U, this.nClusters);
    this.colLabels_ = this._kmeans(V, this.nClusters);
    this.fitted_ = true;
    return this;
  }

  private _svd(X: Float64Array[], k: number, n: number, p: number): { U: Float64Array[]; V: Float64Array[] } {
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return (rng / 0xffffffff) * 2 - 1; };
    const U: Float64Array[] = [], V: Float64Array[] = [];

    let Xk = X.map(row => row.slice());
    for (let s = 0; s < k; s++) {
      let u = new Float64Array(n).map(() => rand());
      let v = new Float64Array(p);
      // Power iteration
      for (let iter = 0; iter < 10; iter++) {
        // v = X^T u / ||X^T u||
        for (let j = 0; j < p; j++) { v[j] = 0; for (let i = 0; i < n; i++) v[j] = (v[j] ?? 0) + (Xk[i]?.[j] ?? 0) * (u[i] ?? 0); }
        const vn = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
        v = v.map(x => x / vn);
        // u = X v / ||X v||
        for (let i = 0; i < n; i++) { u[i] = 0; for (let j = 0; j < p; j++) u[i] = (u[i] ?? 0) + (Xk[i]?.[j] ?? 0) * (v[j] ?? 0); }
        const un = Math.sqrt(u.reduce((s, x) => s + x * x, 0)) || 1;
        u = u.map(x => x / un);
      }
      U.push(u); V.push(v);
      // Deflate
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) Xk[i]![j] = (Xk[i]?.[j] ?? 0) - (u[i] ?? 0) * (v[j] ?? 0);
    }
    // Transpose U, V: each row is a sample's embedding
    const Urows = Array.from({ length: n }, (_, i) => new Float64Array(U.map(u => u[i] ?? 0)));
    const Vrows = Array.from({ length: p }, (_, j) => new Float64Array(V.map(v => v[j] ?? 0)));
    return { U: Urows, V: Vrows };
  }

  private _kmeans(X: Float64Array[], k: number): Int32Array {
    let rng = this.randomState + 1;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
    const n = X.length, d = X[0]?.length ?? 0;
    const centroids = Array.from({ length: k }, () => X[Math.floor(rand() * n)]!.slice());
    const labels = new Int32Array(n);
    for (let iter = 0; iter < 100; iter++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Number.POSITIVE_INFINITY;
        for (let c = 0; c < k; c++) {
          const d2 = (X[i]!).reduce((s, v, j) => s + (v - (centroids[c]?.[j] ?? 0)) ** 2, 0);
          if (d2 < bestD) { bestD = d2; best = c; }
        }
        if (labels[i] !== best) { labels[i] = best; changed = true; }
      }
      if (!changed) break;
      const sums = Array.from({ length: k }, () => new Float64Array(d));
      const counts = new Int32Array(k);
      for (let i = 0; i < n; i++) {
        const c = labels[i] ?? 0;
        for (let j = 0; j < d; j++) sums[c]![j] = (sums[c]?.[j] ?? 0) + (X[i]?.[j] ?? 0);
        counts[c] = (counts[c] ?? 0) + 1;
      }
      for (let c = 0; c < k; c++) {
        const cnt = counts[c] ?? 1;
        for (let j = 0; j < d; j++) centroids[c]![j] = (sums[c]?.[j] ?? 0) / cnt;
      }
    }
    return labels;
  }

  get rowLabels(): Int32Array { return this.rowLabels_; }
  get colLabels(): Int32Array { return this.colLabels_; }
}

export class CLARAClusteringExt {
  private clusterCenters_: Float64Array[] = [];
  private labels_: Int32Array = new Int32Array(0);
  private inertia_: number = 0;
  private fitted_ = false;

  constructor(
    private nClusters: number = 8,
    private nSampling: number = 5,
    private samplingSize: number = 40,
    private maxIter: number = 300,
    private randomState: number = 42
  ) {}

  fit(X: Float64Array[]): this {
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
    const n = X.length;
    const sampleSize = Math.min(this.samplingSize + 2 * this.nClusters, n);

    let bestCenters: Float64Array[] = [], bestInertia = Number.POSITIVE_INFINITY;

    for (let s = 0; s < this.nSampling; s++) {
      // Random sample
      const idx = new Set<number>();
      while (idx.size < sampleSize) idx.add(Math.floor(rand() * n));
      const sample = [...idx].map(i => X[i]!);

      // PAM on sample
      const centers = this._kmedoidsPAM(sample, this.nClusters, rand);
      // Assign all X to nearest center
      let inertia = 0;
      for (const row of X) {
        let bestD = Number.POSITIVE_INFINITY;
        for (const c of centers) {
          const d = row.reduce((ss, v, j) => ss + (v - (c[j] ?? 0)) ** 2, 0);
          if (d < bestD) bestD = d;
        }
        inertia += bestD;
      }
      if (inertia < bestInertia) { bestInertia = inertia; bestCenters = centers; }
    }

    this.clusterCenters_ = bestCenters;
    this.labels_ = new Int32Array(X.map(row => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < bestCenters.length; c++) {
        const d = row.reduce((s, v, j) => s + (v - (bestCenters[c]?.[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    }));
    this.inertia_ = bestInertia;
    this.fitted_ = true;
    return this;
  }

  private _kmedoidsPAM(X: Float64Array[], k: number, rand: () => number): Float64Array[] {
    const n = X.length;
    // Initialize with random medoids
    const medoids: number[] = [];
    while (medoids.length < k) {
      const idx = Math.floor(rand() * n);
      if (!medoids.includes(idx)) medoids.push(idx);
    }
    for (let iter = 0; iter < 50; iter++) {
      let improved = false;
      for (let m = 0; m < k; m++) {
        for (let i = 0; i < n; i++) {
          if (medoids.includes(i)) continue;
          const newMedoids = [...medoids]; newMedoids[m] = i;
          if (this._totalCost(X, newMedoids) < this._totalCost(X, medoids)) {
            medoids[m] = i; improved = true;
          }
        }
      }
      if (!improved) break;
    }
    return medoids.map(i => X[i]!);
  }

  private _totalCost(X: Float64Array[], medoids: number[]): number {
    return X.reduce((s, row) => {
      const minD = Math.min(...medoids.map(m => row.reduce((ss, v, j) => ss + (v - (X[m]?.[j] ?? 0)) ** 2, 0)));
      return s + minD;
    }, 0);
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(row => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.clusterCenters_.length; c++) {
        const d = row.reduce((s, v, j) => s + (v - (this.clusterCenters_[c]?.[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    }));
  }

  get clusterCenters(): Float64Array[] { return this.clusterCenters_; }
  get labels(): Int32Array { return this.labels_; }
  get inertia(): number { return this.inertia_; }
}

export class OnlineKMeansExt {
  private centroids_: Float64Array[] = [];
  private counts_: Int32Array = new Int32Array(0);
  private fitted_ = false;

  constructor(private nClusters: number = 8, private learningRate: number = 0.01) {}

  partialFit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    if (!this.fitted_) {
      this.centroids_ = X.slice(0, this.nClusters).map(row => row.slice());
      while (this.centroids_.length < this.nClusters) this.centroids_.push(new Float64Array(p));
      this.counts_ = new Int32Array(this.nClusters);
      this.fitted_ = true;
    }
    for (const row of X) {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.nClusters; c++) {
        const d = row.reduce((s, v, j) => s + (v - (this.centroids_[c]?.[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = c; }
      }
      this.counts_[best] = (this.counts_[best] ?? 0) + 1;
      const lr = 1 / (this.counts_[best] ?? 1);
      for (let j = 0; j < p; j++) {
        this.centroids_[best]![j] = (1 - lr) * (this.centroids_[best]?.[j] ?? 0) + lr * (row[j] ?? 0);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(row => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.nClusters; c++) {
        const d = row.reduce((s, v, j) => s + (v - (this.centroids_[c]?.[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    }));
  }

  get clusterCenters(): Float64Array[] { return this.centroids_; }
}
