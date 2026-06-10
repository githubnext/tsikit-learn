/**
 * Projected Gradient NMF variant and Alternating Least Squares NMF.
 */

export class NMFAlternatingLeastSquares {
  private W_!: Float64Array[];
  private H_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nComponents = 10,
    private maxIter = 200,
    private tol = 1e-4,
    private l1Ratio = 0,
    private alpha = 0
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1, k = this.nComponents;
    // Initialize W, H with small random values
    this.W_ = Array.from({ length: n }, () => new Float64Array(k).map(() => Math.random() + 0.1));
    this.H_ = Array.from({ length: k }, () => new Float64Array(p).map(() => Math.random() + 0.1));

    let prevErr = Number.POSITIVE_INFINITY;
    for (let iter = 0; iter < this.maxIter; iter++) {
      // Update H: H <- H * (W^T X) / (W^T W H)
      const WtW = Array.from({ length: k }, (_, i) =>
        new Float64Array(k).map((_, j) => this.W_.reduce((s, row) => s + (row[i] ?? 0) * (row[j] ?? 0), 0))
      );
      const WtX = Array.from({ length: k }, (_, i) =>
        new Float64Array(p).map((_, j) => this.W_.reduce((s, row, r) => s + (row[i] ?? 0) * (X[r]![j] ?? 0), 0))
      );
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < p; j++) {
          const num = WtX[i]![j] ?? 0;
          const denom = WtW[i]!.reduce((s, v, l) => s + v * (this.H_[l]![j] ?? 0), 0) + this.alpha * (this.l1Ratio + (1 - this.l1Ratio) * (this.H_[i]![j] ?? 0)) + 1e-10;
          this.H_[i]![j] = Math.max((this.H_[i]![j] ?? 0) * num / denom, 0);
        }
      }
      // Update W: W <- W * (X H^T) / (W H H^T)
      const HHt = Array.from({ length: k }, (_, i) =>
        new Float64Array(k).map((_, j) => this.H_[i]!.reduce((s, v, l) => s + v * (this.H_[j]![l] ?? 0), 0))
      );
      const XHt = Array.from({ length: n }, (_, r) =>
        new Float64Array(k).map((_, i) => X[r]!.reduce((s, v, j) => s + v * (this.H_[i]![j] ?? 0), 0))
      );
      for (let r = 0; r < n; r++) {
        for (let i = 0; i < k; i++) {
          const num = XHt[r]![i] ?? 0;
          const denom = HHt[i]!.reduce((s, v, l) => s + v * (this.W_[r]![l] ?? 0), 0) + 1e-10;
          this.W_[r]![i] = Math.max((this.W_[r]![i] ?? 0) * num / denom, 0);
        }
      }
      // Check convergence
      let err = 0;
      for (let r = 0; r < n; r++) {
        for (let j = 0; j < p; j++) {
          const approx = this.W_[r]!.reduce((s, v, i) => s + v * (this.H_[i]![j] ?? 0), 0);
          err += ((X[r]![j] ?? 0) - approx) ** 2;
        }
      }
      if (Math.abs(prevErr - err) / (prevErr + 1e-10) < this.tol) break;
      prevErr = err;
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = X.length, k = this.nComponents, p = X[0]?.length ?? 1;
    return Array.from({ length: n }, (_, r) => {
      const w = new Float64Array(k);
      for (let iter = 0; iter < 50; iter++) {
        for (let i = 0; i < k; i++) {
          const num = X[r]!.reduce((s, v, j) => s + v * (this.H_[i]![j] ?? 0), 0);
          const denom = this.H_[i]!.reduce((s, v, j) => s + v * w.reduce((ss, ww, l) => ss + ww * (this.H_[l]![j] ?? 0), 0), 0) + 1e-10;
          w[i] = Math.max((w[i] ?? 0) * num / denom, 1e-10);
        }
        void p;
      }
      return w;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get components(): Float64Array[] { return this.H_; }
}
