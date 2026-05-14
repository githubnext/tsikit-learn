/**
 * Gaussian Process Regressor and Classifier.
 * Mirrors sklearn.gaussian_process.GaussianProcessRegressor.
 */

import { NotFittedError } from "../exceptions.js";

export interface GPKernel {
  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[];
  diag(X: Float64Array[]): Float64Array;
}

export class RBFKernel implements GPKernel {
  lengthScale: number;
  constructor(lengthScale = 1.0) {
    this.lengthScale = lengthScale;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let dSq = 0;
        for (let k = 0; k < xi.length; k++) dSq += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        (K[i] as Float64Array)[j] = Math.exp(-0.5 * dSq / (this.lengthScale ** 2));
      }
    }
    return K;
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }
}

export class ConstantKernel implements GPKernel {
  constantValue: number;
  constructor(constantValue = 1.0) {
    this.constantValue = constantValue;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    return Array.from({ length: X1.length }, () => new Float64Array(X2.length).fill(this.constantValue));
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(this.constantValue);
  }
}

export interface GaussianProcessRegressorOptions {
  kernel?: GPKernel | null;
  alpha?: number;
  normalizeY?: boolean;
}

export class GaussianProcessRegressor {
  kernel: GPKernel;
  alpha: number;
  normalizeY: boolean;

  xTrain_: Float64Array[] | null = null;
  yTrain_: Float64Array | null = null;
  alpha_: Float64Array | null = null;
  L_: Float64Array[] | null = null;
  yTrainMean_: number = 0;
  yTrainStd_: number = 1;

  constructor(options: GaussianProcessRegressorOptions = {}) {
    this.kernel = options.kernel ?? new RBFKernel();
    this.alpha = options.alpha ?? 1e-10;
    this.normalizeY = options.normalizeY ?? false;
  }

  private _choleskyDecomp(A: Float64Array[]): Float64Array[] {
    const n = A.length;
    const L: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = (A[i] as Float64Array)[j] ?? 0;
        for (let k = 0; k < j; k++) sum -= ((L[i] as Float64Array)[k] ?? 0) * ((L[j] as Float64Array)[k] ?? 0);
        if (i === j) {
          (L[i] as Float64Array)[j] = Math.sqrt(Math.max(sum, 0));
        } else {
          const ljj = (L[j] as Float64Array)[j] ?? 1;
          (L[i] as Float64Array)[j] = ljj !== 0 ? sum / ljj : 0;
        }
      }
    }
    return L;
  }

  private _solveLower(L: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = b[i] ?? 0;
      for (let j = 0; j < i; j++) sum -= ((L[i] as Float64Array)[j] ?? 0) * (x[j] ?? 0);
      x[i] = sum / ((L[i] as Float64Array)[i] ?? 1);
    }
    return x;
  }

  private _solveUpper(Lt: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = b[i] ?? 0;
      for (let j = i + 1; j < n; j++) sum -= ((Lt[j] as Float64Array)[i] ?? 0) * (x[j] ?? 0);
      x[i] = sum / ((Lt[i] as Float64Array)[i] ?? 1);
    }
    return x;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this.xTrain_ = X;

    let yNorm = new Float64Array(y);
    if (this.normalizeY) {
      let mean = 0;
      for (let i = 0; i < n; i++) mean += y[i] ?? 0;
      mean /= n;
      let std = 0;
      for (let i = 0; i < n; i++) std += ((y[i] ?? 0) - mean) ** 2;
      std = Math.sqrt(std / n) || 1;
      this.yTrainMean_ = mean;
      this.yTrainStd_ = std;
      yNorm = Float64Array.from(y.map(v => (v - mean) / std));
    }
    this.yTrain_ = yNorm;

    const K = this.kernel.compute(X, X);
    for (let i = 0; i < n; i++) (K[i] as Float64Array)[i] = ((K[i] as Float64Array)[i] ?? 0) + this.alpha;

    this.L_ = this._choleskyDecomp(K);
    const v = this._solveLower(this.L_, yNorm);
    this.alpha_ = this._solveUpper(this.L_, v);
    return this;
  }

  predict(X: Float64Array[], returnStd = false): { mean: Float64Array; std?: Float64Array } {
    if (!this.xTrain_ || !this.alpha_ || !this.L_) throw new NotFittedError("GaussianProcessRegressor is not fitted.");
    const KStar = this.kernel.compute(X, this.xTrain_);
    const n = X.length;
    const mean = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < this.xTrain_.length; j++) sum += ((KStar[i] as Float64Array)[j] ?? 0) * (this.alpha_[j] ?? 0);
      mean[i] = sum * this.yTrainStd_ + this.yTrainMean_;
    }

    if (!returnStd) return { mean };

    const kDiag = this.kernel.diag(X);
    const std = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = this._solveLower(this.L_, KStar[i] as Float64Array);
      let vSq = 0;
      for (let j = 0; j < v.length; j++) vSq += (v[j] ?? 0) ** 2;
      std[i] = Math.sqrt(Math.max((kDiag[i] ?? 0) - vSq, 0)) * this.yTrainStd_;
    }
    return { mean, std };
  }

  score(X: Float64Array[], y: Float64Array): number {
    const { mean: preds } = this.predict(X);
    const n = y.length;
    let ymean = 0;
    for (let i = 0; i < n; i++) ymean += y[i] ?? 0;
    ymean /= n;
    let ssRes = 0; let ssTot = 0;
    for (let i = 0; i < n; i++) {
      ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - ymean) ** 2;
    }
    return 1 - ssRes / (ssTot || 1);
  }
}
