/**
 * Graphical Lasso (Sparse Inverse Covariance Estimation).
 */

export class GraphicalLassoCV {
  private precision_!: Float64Array[];
  private covariance_!: Float64Array[];
  private alpha_!: number;
  private alphas: Float64Array;
  private fitted_ = false;

  constructor(alphas: Float64Array | number[] = [0.01, 0.05, 0.1, 0.5, 1.0], private cv = 5, private maxIter = 100, private tol = 1e-4) {
    this.alphas = alphas instanceof Float64Array ? alphas : new Float64Array(alphas);
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    let bestAlpha = this.alphas[0] ?? 0.1, bestScore = -Number.POSITIVE_INFINITY;
    const foldSize = Math.floor(n / this.cv);

    for (const alpha of this.alphas) {
      let cvScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize, end = fold === this.cv - 1 ? n : start + foldSize;
        const trainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
        const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
        const XTr = trainIdx.map(i => X[i]!);
        const XTe = testIdx.map(i => X[i]!);
        const { precision } = this._glasso(this._sampleCov(XTr, p), alpha, p);
        const S_test = this._sampleCov(XTe, p);
        // Log-likelihood on test: tr(S * Theta) - log_det(Theta)
        const trST = S_test.reduce((s, row, i) => s + row.reduce((ss, v, j) => ss + v * (precision[j]![i] ?? 0), 0), 0);
        cvScore += trST;
      }
      if (-cvScore / this.cv > bestScore) { bestScore = -cvScore / this.cv; bestAlpha = alpha; }
    }

    this.alpha_ = bestAlpha;
    const S = this._sampleCov(X, p);
    const { precision, covariance } = this._glasso(S, this.alpha_, p);
    this.precision_ = precision;
    this.covariance_ = covariance;
    this.fitted_ = true;
    return this;
  }

  private _sampleCov(X: Float64Array[], p: number): Float64Array[] {
    const n = X.length;
    const mean = new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[j] ?? 0), 0) / n);
    return Array.from({ length: p }, (_, i) =>
      new Float64Array(p).map((_, j) => X.reduce((s, row) => s + ((row[i] ?? 0) - mean[i]!) * ((row[j] ?? 0) - mean[j]!), 0) / n)
    );
  }

  private _glasso(S: Float64Array[], alpha: number, p: number): { precision: Float64Array[]; covariance: Float64Array[] } {
    let W = S.map((row, i) => new Float64Array(row.map((v, j) => i === j ? v + alpha : v)));
    for (let iter = 0; iter < this.maxIter; iter++) {
      const Wprev = W.map(row => new Float64Array(row));
      for (let j = 0; j < p; j++) {
        // Partition: W = [[W11, w12], [w12^T, w22]]
        const W11 = Array.from({ length: p - 1 }, (_, i) => {
          const ii = i < j ? i : i + 1;
          return new Float64Array(p - 1).map((_, k) => { const kk = k < j ? k : k + 1; return W[ii]![kk] ?? 0; });
        });
        const s12 = new Float64Array(p - 1).map((_, i) => { const ii = i < j ? i : i + 1; return S[j]![ii] ?? 0; });
        // Coordinate descent for w12 = W11 * beta, lasso penalty
        const beta = this._lasso(W11, s12, alpha);
        const w12 = new Float64Array(p - 1).map((_, i) => W11[i]!.reduce((s, v, k) => s + v * (beta[k] ?? 0), 0));
        for (let i = 0; i < p; i++) {
          if (i === j) continue;
          const ii = i < j ? i : i - 1;
          W[i]![j] = w12[ii] ?? 0;
          W[j]![i] = w12[ii] ?? 0;
        }
      }
      const diff = W.reduce((s, row, i) => s + row.reduce((ss, v, jj) => ss + (v - (Wprev[i]![jj] ?? 0)) ** 2, 0), 0);
      if (diff < this.tol) break;
    }
    // Compute precision from W (simplified inversion)
    const precision = this._invertMat(W, p);
    return { precision, covariance: W };
  }

  private _lasso(A: Float64Array[], b: Float64Array, lambda: number): Float64Array {
    const p = b.length;
    const beta = new Float64Array(p);
    for (let iter = 0; iter < 200; iter++) {
      for (let j = 0; j < p; j++) {
        const rj = b[j]! - A[j]!.reduce((s, v, k) => s + k !== j ? v * (beta[k] ?? 0) : 0, 0);
        const Ajj = A[j]![j] ?? 1;
        beta[j] = Math.sign(rj) * Math.max(Math.abs(rj) - lambda, 0) / Ajj;
      }
    }
    return beta;
  }

  private _invertMat(A: Float64Array[], p: number): Float64Array[] {
    const aug = A.map((row, i) => [...row, ...(new Float64Array(p).map((_, j) => (i === j ? 1 : 0)))]);
    for (let col = 0; col < p; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j < 2 * p; j++) aug[col]![j] = (aug[col]![j] ?? 0) / piv;
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j < 2 * p; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => aug[i]![p + j] ?? 0));
  }

  get precision(): Float64Array[] { return this.precision_; }
  get covariance(): Float64Array[] { return this.covariance_; }
  get alpha(): number { return this.alpha_; }
}
