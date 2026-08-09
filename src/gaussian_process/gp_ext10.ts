/**
 * Gaussian process extensions: Student-t process, Sparse GP, GP classification extensions.
 * Mirrors sklearn.gaussian_process additional methods.
 */

import { BaseEstimator } from "../base.js";

export interface StudentTProcParams {
  nu?: number;
  noise?: number;
  length_scale?: number;
}

/** Student-t Process Regression (heavy-tailed GP variant). */
export class StudentTProcess extends BaseEstimator {
  nu: number;
  noise: number;
  length_scale: number;
  X_train_: Float64Array[] = [];
  y_train_: Float64Array = new Float64Array(0);
  K_inv_: Float64Array[] = [];

  constructor(params: StudentTProcParams = {}) {
    super();
    this.nu = params.nu ?? 3.0;
    this.noise = params.noise ?? 1e-3;
    this.length_scale = params.length_scale ?? 1.0;
  }

  private rbf(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.exp(-s / (2 * this.length_scale ** 2));
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this.X_train_ = X;
    this.y_train_ = y;
    const K: Float64Array[] = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => this.rbf(X[i]!, X[j]!)),
    );
    for (let i = 0; i < n; i++) K[i]![i] = (K[i]?.[i] ?? 0) + this.noise;
    this.K_inv_ = invertMatrix(K);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const n = this.X_train_.length;
    const m = X.length;
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const k = new Float64Array(n).map((_, j) => this.rbf(X[i]!, this.X_train_[j]!));
      let s = 0;
      for (let j = 0; j < n; j++) {
        let kKinv = 0;
        for (let l = 0; l < n; l++) kKinv += (k[l] ?? 0) * (this.K_inv_[l]?.[j] ?? 0);
        s += kKinv * (this.y_train_[j] ?? 0);
      }
      out[i] = s;
    }
    return out;
  }
}

function invertMatrix(A: Float64Array[]): Float64Array[] {
  const n = A.length;
  const aug: Float64Array[] = A.map((r, i) => {
    const row = new Float64Array(2 * n);
    for (let j = 0; j < n; j++) row[j] = r[j] ?? 0;
    row[n + i] = 1;
    return row;
  });
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r]?.[col] ?? 0) > Math.abs(aug[pivot]?.[col] ?? 0)) pivot = r;
    }
    const tmp = aug[col]!;
    aug[col] = aug[pivot]!;
    aug[pivot] = tmp;
    const scale = aug[col]?.[col] ?? 1;
    if (Math.abs(scale) < 1e-12) continue;
    for (let j = 0; j < 2 * n; j++) aug[col]![j] = (aug[col]?.[j] ?? 0) / scale;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r]?.[col] ?? 0;
      for (let j = 0; j < 2 * n; j++) aug[r]![j] = (aug[r]?.[j] ?? 0) - factor * (aug[col]?.[j] ?? 0);
    }
  }
  return aug.map(r => r.slice(n));
}

/** Sparse Gaussian Process approximation via inducing points. */
export class SparseGP extends BaseEstimator {
  n_inducing: number;
  noise: number;
  length_scale: number;
  inducing_points_: Float64Array[] = [];
  alpha_: Float64Array = new Float64Array(0);

  constructor(params: { n_inducing?: number; noise?: number; length_scale?: number } = {}) {
    super();
    this.n_inducing = params.n_inducing ?? 20;
    this.noise = params.noise ?? 0.01;
    this.length_scale = params.length_scale ?? 1.0;
  }

  private rbf(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.exp(-s / (2 * this.length_scale ** 2));
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const m = Math.min(this.n_inducing, n);
    // Select inducing points via k-means style init
    const step = Math.floor(n / m);
    this.inducing_points_ = Array.from({ length: m }, (_, i) => new Float64Array(X[i * step]!));

    // Build Kuu (m x m), Kuf (m x n)
    const Kuu: Float64Array[] = Array.from({ length: m }, (_, i) =>
      new Float64Array(m).map((_, j) => this.rbf(this.inducing_points_[i]!, this.inducing_points_[j]!)),
    );
    for (let i = 0; i < m; i++) Kuu[i]![i] = (Kuu[i]?.[i] ?? 0) + 1e-6;
    const Kuf: Float64Array[] = Array.from({ length: m }, (_, i) =>
      new Float64Array(n).map((_, j) => this.rbf(this.inducing_points_[i]!, X[j]!)),
    );

    // Q = Kuf @ Kuf.T / sigma^2 + Kuu
    const sigma2 = this.noise ** 2;
    const Q: Float64Array[] = Array.from({ length: m }, () => new Float64Array(m));
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += (Kuf[i]?.[k] ?? 0) * (Kuf[j]?.[k] ?? 0);
      Q[i]![j] = s / sigma2 + (Kuu[i]?.[j] ?? 0);
    }

    // alpha = Q^-1 @ Kuf @ y / sigma^2
    const rhs = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < n; k++) rhs[i] = (rhs[i] ?? 0) + (Kuf[i]?.[k] ?? 0) * (y[k] ?? 0);
      rhs[i] = (rhs[i] ?? 0) / sigma2;
    }

    const Qinv = invertMatrix(Q);
    const alpha = new Float64Array(m);
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) alpha[i] = (alpha[i] ?? 0) + (Qinv[i]?.[j] ?? 0) * (rhs[j] ?? 0);
    this.alpha_ = alpha;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const n = X.length;
    const m = this.inducing_points_.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        out[i] = (out[i] ?? 0) + this.rbf(X[i]!, this.inducing_points_[j]!) * (this.alpha_[j] ?? 0);
      }
    }
    return out;
  }
}

/** GP Kernel: Periodic kernel for modeling cyclic patterns. */
export class PeriodicKernel {
  length_scale: number;
  periodicity: number;
  amplitude: number;

  constructor(params: { length_scale?: number; periodicity?: number; amplitude?: number } = {}) {
    this.length_scale = params.length_scale ?? 1.0;
    this.periodicity = params.periodicity ?? 1.0;
    this.amplitude = params.amplitude ?? 1.0;
  }

  call(a: Float64Array, b: Float64Array): number {
    let dist = 0;
    for (let i = 0; i < a.length; i++) dist += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    dist = Math.sqrt(dist);
    const sin = Math.sin(Math.PI * dist / this.periodicity);
    return this.amplitude ** 2 * Math.exp(-2 * sin ** 2 / this.length_scale ** 2);
  }
}
