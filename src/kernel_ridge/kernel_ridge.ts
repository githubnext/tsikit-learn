/**
 * KernelRidge regression.
 * Mirrors sklearn.kernel_ridge.KernelRidge.
 */

import { NotFittedError } from "../exceptions.js";

export type KernelType = "linear" | "rbf" | "poly" | "sigmoid";

export interface KernelRidgeOptions {
  alpha?: number;
  kernel?: KernelType;
  gamma?: number | null;
  degree?: number;
  coef0?: number;
}

function computeKernel(
  X: Float64Array[],
  Y: Float64Array[],
  kernel: KernelType,
  gamma: number,
  degree: number,
  coef0: number,
): Float64Array[] {
  const n = X.length;
  const m = Y.length;
  const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(0);
    for (let j = 0; j < m; j++) {
      const yj = Y[j] ?? new Float64Array(0);
      let dot = 0;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
      let val: number;
      if (kernel === "linear") {
        val = dot;
      } else if (kernel === "rbf") {
        let distSq = 0;
        for (let k = 0; k < xi.length; k++) distSq += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
        val = Math.exp(-gamma * distSq);
      } else if (kernel === "poly") {
        val = (gamma * dot + coef0) ** degree;
      } else { // sigmoid
        val = Math.tanh(gamma * dot + coef0);
      }
      (K[i] as Float64Array)[j] = val;
    }
  }
  return K;
}

export class KernelRidge {
  alpha: number;
  kernel: KernelType;
  gamma: number | null;
  degree: number;
  coef0: number;

  dualCoef_: Float64Array | null = null;
  xFit_: Float64Array[] | null = null;

  constructor(options: KernelRidgeOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.kernel = options.kernel ?? "linear";
    this.gamma = options.gamma ?? null;
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 1;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const gamma = this.gamma ?? (p > 0 ? 1 / p : 1);

    const K = computeKernel(X, X, this.kernel, gamma, this.degree, this.coef0);
    // Add alpha * I
    for (let i = 0; i < n; i++) (K[i] as Float64Array)[i] = ((K[i] as Float64Array)[i] ?? 0) + this.alpha;

    // Solve (K + alpha*I) * dual_coef = y using Cholesky-like (Gaussian elimination)
    // Simple Gaussian elimination with partial pivoting
    const aug = K.map((row, i) => {
      const r = new Float64Array(n + 1);
      for (let j = 0; j < n; j++) r[j] = (row as Float64Array)[j] ?? 0;
      r[n] = y[i] ?? 0;
      return r;
    });

    for (let col = 0; col < n; col++) {
      // Find pivot
      let maxRow = col;
      let maxVal = Math.abs((aug[col] as Float64Array)[col] ?? 0);
      for (let row = col + 1; row < n; row++) {
        const v = Math.abs((aug[row] as Float64Array)[col] ?? 0);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      if (maxRow !== col) { [aug[col], aug[maxRow]] = [aug[maxRow] as Float64Array, aug[col] as Float64Array]; }
      const pivot = (aug[col] as Float64Array)[col] ?? 0;
      if (Math.abs(pivot) < 1e-12) continue;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = ((aug[row] as Float64Array)[col] ?? 0) / pivot;
        for (let j = col; j <= n; j++) {
          (aug[row] as Float64Array)[j] = ((aug[row] as Float64Array)[j] ?? 0) - factor * ((aug[col] as Float64Array)[j] ?? 0);
        }
      }
      for (let j = col + 1; j <= n; j++) {
        (aug[col] as Float64Array)[j] = ((aug[col] as Float64Array)[j] ?? 0) / pivot;
      }
      (aug[col] as Float64Array)[col] = 1;
    }

    this.dualCoef_ = Float64Array.from(aug.map(row => (row as Float64Array)[n] ?? 0));
    this.xFit_ = X;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.dualCoef_ || !this.xFit_) throw new NotFittedError("KernelRidge is not fitted.");
    const p = (this.xFit_[0] ?? new Float64Array(0)).length;
    const gamma = this.gamma ?? (p > 0 ? 1 / p : 1);
    const K = computeKernel(X, this.xFit_, this.kernel, gamma, this.degree, this.coef0);
    const n = X.length;
    const nTrain = this.xFit_.length;
    const preds = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < nTrain; j++) sum += ((K[i] as Float64Array)[j] ?? 0) * (this.dualCoef_[j] ?? 0);
      preds[i] = sum;
    }
    return preds;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const n = y.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += y[i] ?? 0;
    mean /= n;
    let ssRes = 0; let ssTot = 0;
    for (let i = 0; i < n; i++) {
      ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return 1 - ssRes / (ssTot || 1);
  }
}
