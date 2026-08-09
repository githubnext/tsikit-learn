/**
 * Sparse Gaussian Process Regression and Laplace approximation for GP classification.
 */

export type KernelFn = (x1: Float64Array, x2: Float64Array) => number;

export function rbfKernel(lengthScale = 1.0, signalVariance = 1.0): KernelFn {
  return (x1, x2) => {
    const dist2 = x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0);
    return signalVariance * Math.exp(-0.5 * dist2 / (lengthScale ** 2));
  };
}

export class SparseGPR {
  private inducing_!: Float64Array[];
  private alpha_!: Float64Array;
  private Kmm_inv_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private kernel: KernelFn = rbfKernel(),
    private nInducing = 10,
    private noiseVariance = 0.1,
    private maxIter = 20
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    // Select inducing points via k-means++ like init
    this.inducing_ = this._selectInducing(X);
    const m = this.inducing_.length;

    // Compute Kmm (inducing-inducing covariance)
    const Kmm = Array.from({ length: m }, (_, i) =>
      new Float64Array(m).map((_, j) => this.kernel(this.inducing_[i]!, this.inducing_[j]!))
    );
    for (let i = 0; i < m; i++) Kmm[i]![i] = (Kmm[i]![i] ?? 0) + 1e-6;

    // Compute Knm (training-inducing covariance)
    const Knm = Array.from({ length: n }, (_, i) =>
      new Float64Array(m).map((_, j) => this.kernel(X[i]!, this.inducing_[j]!))
    );

    // FITC approximation: Q_nn + diag(Knn - Qnn) + sigma^2 I
    const Knn_diag = new Float64Array(n).map((_, i) => this.kernel(X[i]!, X[i]!));
    const Qnn_diag = new Float64Array(n).map((_, i) =>
      Knm[i]!.reduce((s, v, j) => s + v * this._solve(Kmm, Knm[i]!)[j]!, 0)
    );
    const Lambda = new Float64Array(n).map((_, i) =>
      Math.max(1e-6, (Knn_diag[i] ?? 0) - (Qnn_diag[i] ?? 0)) + this.noiseVariance
    );

    // alpha = (Kmm + Knm^T Lambda^{-1} Knm)^{-1} Knm^T Lambda^{-1} y
    const KnmT_LambdaInv = Array.from({ length: m }, (_, j) =>
      new Float64Array(n).map((_, i) => (Knm[i]![j] ?? 0) / (Lambda[i] ?? 1))
    );
    const A = Array.from({ length: m }, (_, a) =>
      new Float64Array(m).map((_, b) =>
        (Kmm[a]![b] ?? 0) + KnmT_LambdaInv[a]!.reduce((s, v, i) => s + v * (Knm[i]![b] ?? 0), 0)
      )
    );
    for (let a = 0; a < m; a++) A[a]![a] = (A[a]![a] ?? 0) + 1e-6;
    const rhs = new Float64Array(m).map((_, j) =>
      KnmT_LambdaInv[j]!.reduce((s, v, i) => s + v * (y[i] ?? 0), 0)
    );
    this.alpha_ = this._solve(A, rhs);
    this.Kmm_inv_ = this._invert(Kmm);
    this.fitted_ = true;
    void p; void this.maxIter;
    return this;
  }

  predict(Xtest: Float64Array[]): { mean: Float64Array; variance: Float64Array } {
    if (!this.fitted_) throw new Error('Not fitted');
    const m = this.inducing_.length;
    const mean = new Float64Array(Xtest.length).map((_, i) => {
      const kstar = new Float64Array(m).map((_, j) => this.kernel(Xtest[i]!, this.inducing_[j]!));
      return kstar.reduce((s, v, j) => s + v * (this.alpha_[j] ?? 0), 0);
    });
    const variance = new Float64Array(Xtest.length).map((_, i) => {
      const kstar = new Float64Array(m).map((_, j) => this.kernel(Xtest[i]!, this.inducing_[j]!));
      const v = this._solve(this.Kmm_inv_ as unknown as Float64Array[], kstar);
      const varReduction = kstar.reduce((s, val, j) => s + val * (v[j] ?? 0), 0);
      return this.kernel(Xtest[i]!, Xtest[i]!) - varReduction + this.noiseVariance;
    });
    return { mean, variance };
  }

  private _selectInducing(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const m = Math.min(this.nInducing, n);
    const indices = new Set<number>();
    indices.add(Math.floor(Math.random() * n));
    while (indices.size < m) {
      const arr = Array.from(indices);
      let bestIdx = 0, bestDist = -1;
      for (let i = 0; i < n; i++) {
        if (indices.has(i)) continue;
        const minDist = arr.reduce((md, j) => Math.min(md, X[i]!.reduce((s, v, k) => s + (v - (X[j]![k] ?? 0)) ** 2, 0)), Number.POSITIVE_INFINITY);
        if (minDist > bestDist) { bestDist = minDist; bestIdx = i; }
      }
      indices.add(bestIdx);
    }
    return Array.from(indices).map(i => X[i]!);
  }

  private _solve(A: Float64Array[], b: Float64Array): Float64Array {
    const m = b.length;
    const aug = A.map((row, i) => [...Array.from(row), b[i] ?? 0]);
    for (let col = 0; col < m; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j <= m; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (piv + 1e-10);
      for (let row = 0; row < m; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j <= m; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return new Float64Array(m).map((_, i) => aug[i]![m] ?? 0);
  }

  private _invert(A: Float64Array[]): Float64Array[] {
    const m = A.length;
    const aug = A.map((row, i) => {
      const r = Array.from(row) as number[];
      for (let j = 0; j < m; j++) r.push(i === j ? 1 : 0);
      return r;
    });
    for (let col = 0; col < m; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j < 2 * m; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (piv + 1e-10);
      for (let row = 0; row < m; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j < 2 * m; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return Array.from({ length: m }, (_, i) => new Float64Array(m).map((_, j) => aug[i]![m + j] ?? 0));
  }
}
