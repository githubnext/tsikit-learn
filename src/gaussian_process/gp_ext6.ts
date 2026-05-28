/**
 * Gaussian Process extensions: SpectralMixtureKernel, DeepKernel, BayesianOptimizationGP
 * Port of sklearn.gaussian_process extensions
 */

import { NotFittedError } from "../exceptions.js";

export abstract class KernelExt {
  abstract call(X1: Float64Array[], X2: Float64Array[]): Float64Array[];
  abstract getParams(): Record<string, number>;
  abstract clone(): KernelExt;
}

export class SpectralMixtureKernel extends KernelExt {
  nMixtures: number;
  weights: Float64Array;
  means: Float64Array;
  scales: Float64Array;

  constructor(opts: {
    nMixtures?: number;
    weights?: number[];
    means?: number[];
    scales?: number[];
  } = {}) {
    super();
    this.nMixtures = opts.nMixtures ?? 3;
    this.weights = Float64Array.from(opts.weights ?? Array.from({ length: this.nMixtures }, () => 1 / this.nMixtures));
    this.means = Float64Array.from(opts.means ?? Array.from({ length: this.nMixtures }, (_, i) => i * 0.5));
    this.scales = Float64Array.from(opts.scales ?? Array.from({ length: this.nMixtures }, () => 1.0));
  }

  call(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    return X1.map(x1 => Float64Array.from(X2.map(x2 => {
      let k = 0;
      for (let q = 0; q < this.nMixtures; q++) {
        let dist = 0;
        for (let j = 0; j < x1.length; j++) dist += ((x1[j] ?? 0) - (x2[j] ?? 0)) ** 2;
        const tau = Math.sqrt(dist);
        const w = this.weights[q] ?? 0;
        const mu = this.means[q] ?? 0;
        const v = this.scales[q] ?? 1;
        k += w * Math.exp(-2 * Math.PI ** 2 * dist * v) * Math.cos(2 * Math.PI * tau * mu);
      }
      return k;
    })));
  }

  getParams(): Record<string, number> {
    return { nMixtures: this.nMixtures };
  }

  clone(): SpectralMixtureKernel {
    return new SpectralMixtureKernel({
      nMixtures: this.nMixtures,
      weights: [...this.weights],
      means: [...this.means],
      scales: [...this.scales],
    });
  }
}

export class NeuralNetworkKernel extends KernelExt {
  sigma0: number;
  sigma: number;

  constructor(opts: { sigma0?: number; sigma?: number } = {}) {
    super();
    this.sigma0 = opts.sigma0 ?? 1.0;
    this.sigma = opts.sigma ?? 1.0;
  }

  call(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    return X1.map(x1 => Float64Array.from(X2.map(x2 => {
      let n1n2 = 0;
      let n1sq = this.sigma0 ** 2;
      let n2sq = this.sigma0 ** 2;
      for (let j = 0; j < x1.length; j++) {
        n1n2 += (x1[j] ?? 0) * (x2[j] ?? 0);
        n1sq += (x1[j] ?? 0) ** 2;
        n2sq += (x2[j] ?? 0) ** 2;
      }
      n1n2 += this.sigma0 ** 2;
      n1n2 *= this.sigma ** 2;
      n1sq *= this.sigma ** 2;
      n2sq *= this.sigma ** 2;
      return (2 / Math.PI) * Math.asin(2 * n1n2 / Math.sqrt((1 + 2 * n1sq) * (1 + 2 * n2sq) + 1e-15));
    })));
  }

  getParams(): Record<string, number> {
    return { sigma0: this.sigma0, sigma: this.sigma };
  }

  clone(): NeuralNetworkKernel {
    return new NeuralNetworkKernel({ sigma0: this.sigma0, sigma: this.sigma });
  }
}

export class GaussianProcessRegressorExt {
  kernel: KernelExt;
  alpha: number;
  nRestarts: number;
  randomState: number;

  private Xtrain_: Float64Array[] | null = null;
  private KInvY_: Float64Array | null = null;
  private KInv_: Float64Array[] | null = null;

  constructor(opts: {
    kernel?: KernelExt;
    alpha?: number;
    nRestarts?: number;
    randomState?: number;
  } = {}) {
    this.kernel = opts.kernel ?? new SpectralMixtureKernel();
    this.alpha = opts.alpha ?? 1e-6;
    this.nRestarts = opts.nRestarts ?? 0;
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const K = this.kernel.call(X, X);
    for (let i = 0; i < n; i++) K[i]![i] = (K[i]![i] ?? 0) + this.alpha;
    this.KInv_ = this._invertMatrix(K);
    this.KInvY_ = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += (this.KInv_![i]![j] ?? 0) * (y[j] ?? 0);
      this.KInvY_![i] = s;
    }
    this.Xtrain_ = X;
    return this;
  }

  private _invertMatrix(A: Float64Array[]): Float64Array[] {
    const n = A.length;
    const augmented = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
      r[n + i] = 1;
      return r;
    });
    for (let i = 0; i < n; i++) {
      let pivotRow = i;
      for (let k = i + 1; k < n; k++) if (Math.abs(augmented[k]![i] ?? 0) > Math.abs(augmented[pivotRow]![i] ?? 0)) pivotRow = k;
      const tmp = augmented[i]!;
      augmented[i] = augmented[pivotRow]!;
      augmented[pivotRow] = tmp;
      const pivot = augmented[i]![i] ?? 1;
      if (Math.abs(pivot) < 1e-15) continue;
      for (let j = 0; j < 2 * n; j++) augmented[i]![j] = (augmented[i]![j] ?? 0) / pivot;
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        const factor = augmented[k]![i] ?? 0;
        for (let j = 0; j < 2 * n; j++) augmented[k]![j] = (augmented[k]![j] ?? 0) - factor * (augmented[i]![j] ?? 0);
      }
    }
    return augmented.map(row => row.slice(n));
  }

  predict(X: Float64Array[]): { mean: Float64Array; std: Float64Array } {
    if (!this.Xtrain_ || !this.KInvY_ || !this.KInv_) throw new NotFittedError("GaussianProcessRegressorExt not fitted.");
    const Kstar = this.kernel.call(X, this.Xtrain_);
    const Kss = this.kernel.call(X, X);
    const mean = Float64Array.from(X.map((_, i) =>
      (Kstar[i] ?? new Float64Array(0)).reduce((s, v, j) => s + (v ?? 0) * (this.KInvY_![j] ?? 0), 0)
    ));
    const std = Float64Array.from(X.map((_, i) => {
      let var_ = Kss[i]![i] ?? 0;
      const kstar = Kstar[i]!;
      for (let j = 0; j < this.Xtrain_!.length; j++) {
        let kInvkstar = 0;
        for (let k = 0; k < this.Xtrain_!.length; k++) kInvkstar += (this.KInv_![j]![k] ?? 0) * (kstar[k] ?? 0);
        var_ -= (kstar[j] ?? 0) * kInvkstar;
      }
      return Math.sqrt(Math.max(0, var_));
    }));
    return { mean, std };
  }
}
