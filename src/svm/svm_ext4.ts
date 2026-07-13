/**
 * OneClassSVM and NuSVR extensions.
 */

function rbfKernelMatrix(X1: Float64Array[], X2: Float64Array[], gamma: number): Float64Array[] {
  return X1.map((x1) => Float64Array.from(X2, (x2) => {
    let d2 = 0;
    for (let d = 0; d < x1.length; d++) d2 += ((x1[d] ?? 0) - (x2[d] ?? 0)) ** 2;
    return Math.exp(-gamma * d2);
  }));
}

export class OneClassSVMExt {
  nu: number;
  kernel: "rbf" | "linear";
  gamma: "scale" | "auto" | number;
  maxIter: number;
  tol: number;
  private _alphas: Float64Array | null = null;
  private _supportVectors: Float64Array[] | null = null;
  private _rho: number = 0;
  private _gamma: number = 1.0;
  nSupportVectors_: number = 0;

  constructor(nu = 0.5, kernel: "rbf" | "linear" = "rbf", gamma: "scale" | "auto" | number = "scale", maxIter = 1000, tol = 1e-4) {
    this.nu = nu;
    this.kernel = kernel;
    this.gamma = gamma;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  private _computeGamma(X: Float64Array[]): number {
    if (typeof this.gamma === "number") return this.gamma;
    const p = X[0]?.length ?? 1;
    if (this.gamma === "auto") return 1 / p;
    // scale: 1 / (p * variance)
    let mean = 0, variance = 0;
    for (const row of X) for (let d = 0; d < row.length; d++) mean += (row[d] ?? 0) / (X.length * p);
    for (const row of X) for (let d = 0; d < row.length; d++) variance += ((row[d] ?? 0) - mean) ** 2 / (X.length * p);
    return 1 / (p * Math.max(variance, 1e-12));
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this._gamma = this._computeGamma(X);

    const K = this.kernel === "rbf" ? rbfKernelMatrix(X, X, this._gamma) : X.map((x1) => Float64Array.from(X, (x2) => x1.reduce((s, v, d) => s + v * (x2[d] ?? 0), 0)));

    // SMO-like optimization for one-class SVM
    const nu = Math.min(Math.max(this.nu, 1e-5), 1.0);
    const C = 1 / (nu * n);
    const alphas = new Float64Array(n).fill(C * nu);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxViolation = 0;
      for (let i = 0; i < n; i++) {
        let fi = -(K[i] as Float64Array).reduce((s, v, j) => s + alphas[j]! * v, 0);
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          // Simple gradient step
          const kij = (K[i] as Float64Array)[j] ?? 0;
          const kii = (K[i] as Float64Array)[i] ?? 1;
          const kjj = (K[j] as Float64Array)[j] ?? 1;
          const eta = Math.max(kii + kjj - 2 * kij, 1e-12);
          const yeta = 1 / eta;
          const oldAlphai = alphas[i] ?? 0;
          const oldAlphaj = alphas[j] ?? 0;
          const deltaI = yeta * (fi - (-(K[j] as Float64Array).reduce((s, v, k) => s + alphas[k]! * v, 0)));
          alphas[i] = Math.max(0, Math.min(C, (alphas[i] ?? 0) + deltaI));
          alphas[j] = oldAlphai + oldAlphaj - (alphas[i] ?? 0);
          alphas[j] = Math.max(0, Math.min(C, alphas[j] ?? 0));
          maxViolation = Math.max(maxViolation, Math.abs((alphas[i] ?? 0) - oldAlphai));
          fi = -(K[i] as Float64Array).reduce((s, v, k) => s + (alphas[k] ?? 0) * v, 0);
        }
      }
      if (maxViolation < this.tol) break;
    }

    const svMask = Array.from({ length: n }, (_, i) => (alphas[i] ?? 0) > this.tol);
    this._supportVectors = X.filter((_, i) => svMask[i]);
    this._alphas = Float64Array.from(alphas.filter((_, i) => svMask[i]));
    this.nSupportVectors_ = this._supportVectors.length;

    // Compute rho (bias)
    const svAlphas = this._alphas;
    const svK = rbfKernelMatrix(this._supportVectors, this._supportVectors, this._gamma);
    const decisionValues = svK.map((row) => row.reduce((s, v, j) => s + (svAlphas[j] ?? 0) * v, 0));
    this._rho = decisionValues.reduce((s, v) => s + v, 0) / Math.max(decisionValues.length, 1);
    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    if (!this._supportVectors || !this._alphas) throw new Error("Not fitted");
    const K = rbfKernelMatrix(X, this._supportVectors, this._gamma);
    return Float64Array.from(K, (row) => row.reduce((s, v, j) => s + (this._alphas![j] ?? 0) * v, 0) - this._rho);
  }

  predict(X: Float64Array[]): Int32Array {
    return Int32Array.from(this.decisionFunction(X), (v) => v >= 0 ? 1 : -1);
  }
}

export class NuSVRExt {
  nu: number;
  C: number;
  kernel: "rbf" | "linear";
  gamma: "scale" | "auto" | number;
  maxIter: number;
  tol: number;
  private _alphas: Float64Array | null = null;
  private _supportVectors: Float64Array[] | null = null;
  private _supportY: Float64Array | null = null;
  private _bias: number = 0;
  private _gamma: number = 1.0;
  nSupportVectors_: number = 0;

  constructor(nu = 0.5, C = 1.0, kernel: "rbf" | "linear" = "rbf", gamma: "scale" | "auto" | number = "scale", maxIter = 1000, tol = 1e-4) {
    this.nu = nu;
    this.C = C;
    this.kernel = kernel;
    this.gamma = gamma;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this._gamma = typeof this.gamma === "number" ? this.gamma : 1 / (X[0]?.length ?? 1);
    const K = rbfKernelMatrix(X, X, this._gamma);
    const eps = this.nu;
    const C = this.C;

    // Simplified epsilon-SVR: find alphas via iterative updates
    const alphasPlus = new Float64Array(n);
    const alphasMinus = new Float64Array(n);
    const rng = () => Math.random();

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxViol = 0;
      for (let i = 0; i < n; i++) {
        const fi = K[i]?.reduce((s, v, j) => s + ((alphasPlus[j] ?? 0) - (alphasMinus[j] ?? 0)) * v, 0) ?? 0;
        const ri = fi - (y[i] ?? 0);
        const gradPlus = ri + eps;
        const gradMinus = -(ri - eps);
        if (gradPlus > 0 && (alphasPlus[i] ?? 0) < C) {
          const step = Math.min(gradPlus / Math.max(K[i]?.[i] ?? 1, 1e-12), C - (alphasPlus[i] ?? 0));
          alphasPlus[i] = (alphasPlus[i] ?? 0) + step;
          maxViol = Math.max(maxViol, step);
        }
        if (gradMinus > 0 && (alphasMinus[i] ?? 0) < C) {
          const step = Math.min(gradMinus / Math.max(K[i]?.[i] ?? 1, 1e-12), C - (alphasMinus[i] ?? 0));
          alphasMinus[i] = (alphasMinus[i] ?? 0) + step;
          maxViol = Math.max(maxViol, step);
        }
      }
      void rng;
      if (maxViol < this.tol) break;
    }

    const svMask = Array.from({ length: n }, (_, i) => Math.abs((alphasPlus[i] ?? 0) - (alphasMinus[i] ?? 0)) > this.tol);
    this._supportVectors = X.filter((_, i) => svMask[i]);
    this._supportY = Float64Array.from(y.filter((_, i) => svMask[i]));
    this._alphas = Float64Array.from({ length: this._supportVectors.length }, (_, i) => {
      const idx = svMask.findIndex((v, j) => v && j >= i);
      return (alphasPlus[idx] ?? 0) - (alphasMinus[idx] ?? 0);
    });
    this.nSupportVectors_ = this._supportVectors.length;

    // Compute bias
    const svK = rbfKernelMatrix(this._supportVectors, this._supportVectors, this._gamma);
    const preds = svK.map((row) => row.reduce((s, v, j) => s + (this._alphas![j] ?? 0) * v, 0));
    this._bias = (this._supportY as Float64Array).reduce((s, v, i) => s + v - (preds[i] ?? 0), 0) / Math.max(this._supportVectors.length, 1);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this._supportVectors || !this._alphas) throw new Error("Not fitted");
    const K = rbfKernelMatrix(X, this._supportVectors, this._gamma);
    return Float64Array.from(K, (row) => row.reduce((s, v, j) => s + (this._alphas![j] ?? 0) * v, 0) + this._bias);
  }
}
