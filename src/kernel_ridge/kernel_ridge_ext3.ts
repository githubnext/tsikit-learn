/**
 * Multi-Kernel Ridge Regression and Kernel Ridge extensions.
 */

export type KernelFn = (x1: Float64Array, x2: Float64Array) => number;

export function rbfKernelFn(gamma = 1.0): KernelFn {
  return (x1, x2) => Math.exp(-gamma * x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
}

export function polynomialKernelFn(degree = 3, gamma = 1.0, coef0 = 1.0): KernelFn {
  return (x1, x2) => (gamma * x1.reduce((s, v, i) => s + v * (x2[i] ?? 0), 0) + coef0) ** degree;
}

export function maternKernel(nu = 1.5, lengthScale = 1.0): KernelFn {
  return (x1, x2) => {
    const dist = Math.sqrt(x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
    if (nu === 0.5) return Math.exp(-dist / lengthScale);
    if (nu === 1.5) {
      const r = Math.sqrt(3) * dist / lengthScale;
      return (1 + r) * Math.exp(-r);
    }
    if (nu === 2.5) {
      const r = Math.sqrt(5) * dist / lengthScale;
      return (1 + r + r ** 2 / 3) * Math.exp(-r);
    }
    return Math.exp(-dist / lengthScale);
  };
}

export class MultiKernelRidge {
  private alpha_!: Float64Array;
  private kernelWeights_!: Float64Array;
  private XTrain_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private kernels: KernelFn[],
    private lambda = 1.0,
    private maxIter = 50
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nKernels = this.kernels.length;
    this.XTrain_ = X;
    // Initialize equal kernel weights
    this.kernelWeights_ = new Float64Array(nKernels).fill(1 / nKernels);

    for (let mkl = 0; mkl < this.maxIter; mkl++) {
      // Compute combined kernel matrix
      const K = this._combinedKernel(X, X);
      // Solve (K + lambda*I) alpha = y
      for (let i = 0; i < n; i++) K[i]![i] = (K[i]![i] ?? 0) + this.lambda;
      this.alpha_ = this._solve(K, y);
      // Update kernel weights via gradient
      const newWeights = new Float64Array(nKernels).map((_, ki) => {
        const Ki = this._singleKernel(X, X, ki);
        const grad = -0.5 * this.alpha_.reduce((s, ai, i) =>
          s + (ai ?? 0) * Ki[i]!.reduce((cs, kv, j) => cs + kv * (this.alpha_[j] ?? 0), 0), 0
        );
        return Math.max(0, (this.kernelWeights_[ki] ?? 0) - 0.01 * grad);
      });
      const sum = newWeights.reduce((s, v) => s + v, 0);
      this.kernelWeights_ = sum > 0 ? new Float64Array(newWeights.map(v => v / sum)) : newWeights;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const K = this._combinedKernel(X, this.XTrain_);
    return new Float64Array(X.length).map((_, i) =>
      K[i]!.reduce((s, v, j) => s + v * (this.alpha_[j] ?? 0), 0)
    );
  }

  private _combinedKernel(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n1 = X1.length, n2 = X2.length;
    const K = Array.from({ length: n1 }, () => new Float64Array(n2));
    for (let ki = 0; ki < this.kernels.length; ki++) {
      const w = this.kernelWeights_[ki] ?? 0;
      for (let i = 0; i < n1; i++) {
        for (let j = 0; j < n2; j++) {
          K[i]![j] = (K[i]![j] ?? 0) + w * this.kernels[ki]!(X1[i]!, X2[j]!);
        }
      }
    }
    return K;
  }

  private _singleKernel(X1: Float64Array[], X2: Float64Array[], ki: number): Float64Array[] {
    const kFn = this.kernels[ki]!;
    return X1.map(x1 => new Float64Array(X2.map(x2 => kFn(x1, x2))));
  }

  private _solve(K: Float64Array[], y: Float64Array): Float64Array {
    const n = y.length;
    const aug = K.map((row, i) => [...Array.from(row), y[i] ?? 0]);
    for (let col = 0; col < n; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j <= n; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (piv + 1e-10);
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j <= n; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return new Float64Array(n).map((_, i) => aug[i]![n] ?? 0);
  }

  get kernelWeights(): Float64Array { return this.kernelWeights_; }
}
