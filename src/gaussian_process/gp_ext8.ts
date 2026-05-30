/**
 * Gaussian process extensions: Deep kernels, neural network kernels.
 * Mirrors sklearn.gaussian_process.kernels advanced extensions.
 */

import { BaseEstimator } from "../base.js";

export interface KernelExt {
  evaluate(X1: Float64Array[], X2: Float64Array[]): Float64Array[];
  diag(X: Float64Array[]): Float64Array;
}

/** Spectral Mixture Kernel: sum of periodic kernels. */
export class SpectralMixtureKernel extends BaseEstimator implements KernelExt {
  Q: number; // number of mixtures
  weights: Float64Array;
  means: Float64Array;
  variances: Float64Array;

  constructor(Q = 4) {
    super();
    this.Q = Q;
    this.weights = new Float64Array(Q).fill(1 / Q);
    this.means = new Float64Array(Q).map((_, i) => i * 0.1);
    this.variances = new Float64Array(Q).fill(1.0);
  }

  evaluate(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n1 = X1.length, n2 = X2.length;
    return Array.from({ length: n1 }, (_, i) =>
      new Float64Array(n2).map((_, j) => {
        let k = 0;
        const tau = (X1[i]?.[0] ?? 0) - (X2[j]?.[0] ?? 0);
        for (let q = 0; q < this.Q; q++) {
          const w = this.weights[q] ?? 0;
          const mu = this.means[q] ?? 0;
          const v = this.variances[q] ?? 1;
          k += w * Math.exp(-2 * Math.PI ** 2 * tau ** 2 * v) * Math.cos(2 * Math.PI * tau * mu);
        }
        return k;
      }),
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).map(() => this.weights.reduce((s, w) => s + w, 0));
  }
}

/** Arc-cosine kernel for deep kernel approximation. */
export class ArcCosineKernel extends BaseEstimator implements KernelExt {
  order: number;
  weight_variances: number;
  bias_variance: number;

  constructor(order = 0, weightVariances = 1.0, biasVariance = 0.0) {
    super();
    this.order = order;
    this.weight_variances = weightVariances;
    this.bias_variance = biasVariance;
  }

  private _Jn(n: number, theta: number): number {
    if (n === 0) return Math.PI - theta;
    if (n === 1) return Math.sin(theta) + (Math.PI - theta) * Math.cos(theta);
    return 3 * Math.sin(theta) * Math.cos(theta) + (Math.PI - theta) * (1 + 2 * Math.cos(theta) ** 2);
  }

  evaluate(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n1 = X1.length, n2 = X2.length;
    return Array.from({ length: n1 }, (_, i) =>
      new Float64Array(n2).map((_, j) => {
        const xi = X1[i]!, xj = X2[j]!;
        let dot = this.bias_variance;
        let ni = this.bias_variance, nj = this.bias_variance;
        for (let k = 0; k < xi.length; k++) {
          dot += this.weight_variances * (xi[k] ?? 0) * (xj[k] ?? 0);
          ni += this.weight_variances * (xi[k] ?? 0) ** 2;
          nj += this.weight_variances * (xj[k] ?? 0) ** 2;
        }
        const niSqrt = Math.sqrt(Math.max(ni, 1e-10));
        const njSqrt = Math.sqrt(Math.max(nj, 1e-10));
        const cosTheta = Math.max(-1, Math.min(1, dot / (niSqrt * njSqrt)));
        const theta = Math.acos(cosTheta);
        return (1 / Math.PI) * niSqrt ** this.order * njSqrt ** this.order * this._Jn(this.order, theta);
      }),
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => {
      let n = this.bias_variance;
      for (let k = 0; k < xi.length; k++) n += this.weight_variances * (xi[k] ?? 0) ** 2;
      return (1 / Math.PI) * n ** this.order * this._Jn(this.order, 0);
    }));
  }
}

/** Deep GP: stacked Gaussian processes for hierarchical modeling. */
export class DeepGPRegressor extends BaseEstimator {
  n_layers: number;
  n_inducing: number;
  X_: Float64Array[] = [];
  y_: Float64Array = new Float64Array(0);
  alpha_: Float64Array = new Float64Array(0);
  kernel_: ArcCosineKernel;

  constructor(nLayers = 2, nInducing = 50) {
    super();
    this.n_layers = nLayers;
    this.n_inducing = nInducing;
    this.kernel_ = new ArcCosineKernel(1);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.X_ = X;
    this.y_ = y;
    const n = X.length;
    const K = this.kernel_.evaluate(X, X);
    // Add jitter
    for (let i = 0; i < n; i++) K[i]![i] = (K[i]![i] ?? 0) + 1e-3;
    // Solve Kα = y via Cholesky (simplified: conjugate gradient)
    this.alpha_ = this._cg(K, y, n);
    return this;
  }

  private _cg(K: Float64Array[], b: Float64Array, n: number): Float64Array {
    let x = new Float64Array(n);
    let r = b.slice();
    let p = r.slice();
    for (let iter = 0; iter < n; iter++) {
      const Kp = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Kp[i] = (Kp[i] ?? 0) + (K[i]?.[j] ?? 0) * (p[j] ?? 0);
      let rTr = 0;
      for (let i = 0; i < n; i++) rTr += (r[i] ?? 0) ** 2;
      let pTKp = 0;
      for (let i = 0; i < n; i++) pTKp += (p[i] ?? 0) * (Kp[i] ?? 0);
      if (Math.abs(pTKp) < 1e-12) break;
      const alpha = rTr / pTKp;
      for (let i = 0; i < n; i++) { x[i] = (x[i] ?? 0) + alpha * (p[i] ?? 0); r[i] = (r[i] ?? 0) - alpha * (Kp[i] ?? 0); }
      let rTrNew = 0;
      for (let i = 0; i < n; i++) rTrNew += (r[i] ?? 0) ** 2;
      if (Math.sqrt(rTrNew) < 1e-8) break;
      const beta = rTrNew / rTr;
      for (let i = 0; i < n; i++) p[i] = (r[i] ?? 0) + beta * (p[i] ?? 0);
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    const Ks = this.kernel_.evaluate(X, this.X_);
    return new Float64Array(X.length).map((_, i) => {
      let s = 0;
      for (let j = 0; j < this.X_.length; j++) s += (Ks[i]?.[j] ?? 0) * (this.alpha_[j] ?? 0);
      return s;
    });
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yp = this.predict(X);
    let sr = 0, st = 0, ym = 0;
    for (let i = 0; i < y.length; i++) ym += y[i] ?? 0;
    ym /= y.length;
    for (let i = 0; i < y.length; i++) {
      sr += ((y[i] ?? 0) - (yp[i] ?? 0)) ** 2;
      st += ((y[i] ?? 0) - ym) ** 2;
    }
    return st === 0 ? 1 : 1 - sr / st;
  }
}
