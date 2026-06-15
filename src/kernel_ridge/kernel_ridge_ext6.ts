/**
 * Localized Kernel Ridge Regression.
 */

export class LocalKernelRidge {
  private centroids_!: Float64Array[];
  private localModels_!: { alpha: Float64Array; support: Float64Array[] }[];
  private fitted_ = false;

  constructor(
    private nLocal = 10,
    private lambda = 1.0,
    private gamma = 1.0,
    private k = 10
  ) {}

  private rbf(x1: Float64Array, x2: Float64Array): number {
    return Math.exp(-this.gamma * x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
  }

  private dist(x1: Float64Array, x2: Float64Array): number {
    return Math.sqrt(x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    // K-means for centroids
    const m = Math.min(this.nLocal, n);
    this.centroids_ = Array.from({ length: m }, (_, i) => Float64Array.from(X[i % n]!));
    for (let iter = 0; iter < 20; iter++) {
      const sums = Array.from({ length: m }, () => ({ sum: new Float64Array(X[0]!.length), count: 0 }));
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = this.dist(X[i]!, this.centroids_[0]!);
        for (let c = 1; c < m; c++) {
          const d = this.dist(X[i]!, this.centroids_[c]!);
          if (d < bestD) { best = c; bestD = d; }
        }
        const s = sums[best]!;
        for (let d = 0; d < X[i]!.length; d++) s.sum[d]! += X[i]![d] ?? 0;
        s.count++;
      }
      for (let c = 0; c < m; c++) {
        const s = sums[c]!;
        if (s.count > 0) this.centroids_[c] = s.sum.map(v => v / s.count);
      }
    }
    // Local models
    this.localModels_ = this.centroids_.map(centroid => {
      const dists = X.map((x, i) => ({ d: this.dist(x, centroid), i }));
      dists.sort((a, b) => a.d - b.d);
      const kIdx = dists.slice(0, this.k).map(d => d.i);
      const Xk = kIdx.map(i => X[i]!);
      const yk = new Float64Array(kIdx.map(i => y[i] ?? 0));
      const K = Array.from({ length: kIdx.length }, (_, i) =>
        new Float64Array(kIdx.length).map((_, j) => this.rbf(Xk[i]!, Xk[j]!))
      );
      for (let i = 0; i < kIdx.length; i++) K[i]![i]! += this.lambda;
      const alpha = solveLocal(K, yk);
      return { alpha, support: Xk };
    });
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error("Not fitted");
    return new Float64Array(X.map(x => {
      let bestC = 0, bestD = this.dist(x, this.centroids_[0]!);
      for (let c = 1; c < this.centroids_.length; c++) {
        const d = this.dist(x, this.centroids_[c]!);
        if (d < bestD) { bestC = c; bestD = d; }
      }
      const model = this.localModels_[bestC]!;
      return model.support.reduce((s, sv, j) => s + (model.alpha[j] ?? 0) * this.rbf(x, sv), 0);
    }));
  }
}

function solveLocal(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  const M = A.map(row => Float64Array.from(row));
  const x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row]![col] ?? 0) > Math.abs(M[maxRow]![col] ?? 0)) maxRow = row;
    const tmp = M[col]; M[col] = M[maxRow]!; M[maxRow] = tmp!;
    const xtmp = x[col] ?? 0; x[col] = x[maxRow] ?? 0; x[maxRow] = xtmp;
    for (let row = col + 1; row < n; row++) {
      const f = (M[row]![col] ?? 0) / (M[col]![col] ?? 1);
      for (let k = col; k < n; k++) M[row]![k]! -= f * (M[col]![k] ?? 0);
      x[row]! -= f * (x[col] ?? 0);
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) x[i]! -= (M[i]![j] ?? 0) * (x[j] ?? 0);
    x[i]! /= M[i]![i] ?? 1;
  }
  return x;
}
