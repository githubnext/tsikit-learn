/**
 * Nyström Kernel Ridge Regression.
 */

export class NystromKernelRidge {
  private alpha_!: Float64Array;
  private landmarks_!: Float64Array[];
  private W_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nLandmarks = 50,
    private lambda = 1.0,
    private gamma = 1.0
  ) {}

  private rbf(x1: Float64Array, x2: Float64Array): number {
    return Math.exp(-this.gamma * x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const m = Math.min(this.nLandmarks, n);
    // Random landmarks
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j]!, idx[i]!];
    }
    this.landmarks_ = idx.slice(0, m).map(i => X[i]!);

    // Build Kmm and Knm
    const Kmm = Array.from({ length: m }, (_, i) =>
      new Float64Array(m).map((_, j) => this.rbf(this.landmarks_[i]!, this.landmarks_[j]!))
    );
    const Knm = Array.from({ length: n }, (_, i) =>
      new Float64Array(m).map((_, j) => this.rbf(X[i]!, this.landmarks_[j]!))
    );

    // Eigendecomposition of Kmm (simplified — use Cholesky approach)
    // W = Knm * (Kmm + lambda*I)^-1
    for (let i = 0; i < m; i++) Kmm[i]![i]! += this.lambda;
    // Solve Kmm * W^T = Knm^T
    this.W_ = Array.from({ length: n }, (_, i) => {
      const rhs = new Float64Array(m).map((_, j) => Knm[i]![j] ?? 0);
      return solveChol(Kmm, rhs);
    });
    // alpha_ = W * y
    this.alpha_ = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (this.W_[i]![j] ?? 0) * (y[i] ?? 0);
      this.alpha_[j] = s;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error("Not fitted");
    return new Float64Array(X.map(x => {
      let pred = 0;
      for (let j = 0; j < this.landmarks_.length; j++)
        pred += (this.alpha_[j] ?? 0) * this.rbf(x, this.landmarks_[j]!);
      return pred;
    }));
  }
}

function solveChol(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i]![j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      L[i]![j] = j === i ? Math.sqrt(Math.max(s, 1e-12)) : s / (L[j]![j] ?? 1);
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i] ?? 0;
    for (let k = 0; k < i; k++) s -= (L[i]![k] ?? 0) * (y[k] ?? 0);
    y[i] = s / (L[i]![i] ?? 1);
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i] ?? 0;
    for (let k = i + 1; k < n; k++) s -= (L[k]![i] ?? 0) * (x[k] ?? 0);
    x[i] = s / (L[i]![i] ?? 1);
  }
  return x;
}
