/**
 * Additional SVM utilities: OneClassSVM, SVR extensions.
 * Mirrors sklearn.svm extras.
 */

import { NotFittedError } from "../exceptions.js";

export class OneClassSVM {
  nu: number;
  kernel: "rbf" | "linear" | "poly";
  gamma: number | "auto" | "scale";
  degree: number;
  maxIter: number;

  supportVectors_: Float64Array[] | null = null;
  dualCoef_: Float64Array | null = null;
  offset_: number = 0;

  constructor(
    options: {
      nu?: number;
      kernel?: "rbf" | "linear" | "poly";
      gamma?: number | "auto" | "scale";
      degree?: number;
      maxIter?: number;
    } = {},
  ) {
    this.nu = options.nu ?? 0.5;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.degree = options.degree ?? 3;
    this.maxIter = options.maxIter ?? 1000;
  }

  private _resolveGamma(nFeatures: number, variance: number): number {
    if (typeof this.gamma === "number") return this.gamma;
    if (this.gamma === "auto") return 1 / nFeatures;
    return 1 / (nFeatures * (variance > 0 ? variance : 1));
  }

  private _kernel(a: Float64Array, b: Float64Array, gamma: number): number {
    if (this.kernel === "linear") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
      return s;
    }
    if (this.kernel === "poly") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
      return Math.pow(gamma * s + 1, this.degree);
    }
    // RBF
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.exp(-gamma * s);
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;

    // Compute variance for gamma=scale
    let variance = 0;
    if (this.gamma === "scale") {
      let mean = 0;
      let count = 0;
      for (const row of X) {
        for (const v of row) {
          mean += v;
          count++;
        }
      }
      mean /= count || 1;
      for (const row of X) {
        for (const v of row) variance += (v - mean) ** 2;
      }
      variance /= count || 1;
    }

    const gamma = this._resolveGamma(nFeatures, variance);
    const nu = Math.min(this.nu, 1);
    const maxSV = Math.ceil(nu * n);

    // Simplified: use k-center approach as approximation
    // Select support vectors as most "central" or spread out points
    const supportIndices: number[] = [];
    const remaining = new Set(Array.from({ length: n }, (_, i) => i));

    // Pick first point randomly (index 0)
    supportIndices.push(0);
    remaining.delete(0);

    while (supportIndices.length < maxSV && remaining.size > 0) {
      let farthest = -1;
      let maxDist = -1;
      for (const idx of remaining) {
        let minDistToSV = Number.POSITIVE_INFINITY;
        for (const sv of supportIndices) {
          let d = 0;
          for (let j = 0; j < nFeatures; j++) {
            d += ((X[idx]?.[j] ?? 0) - (X[sv]?.[j] ?? 0)) ** 2;
          }
          if (d < minDistToSV) minDistToSV = d;
        }
        if (minDistToSV > maxDist) {
          maxDist = minDistToSV;
          farthest = idx;
        }
      }
      if (farthest >= 0) {
        supportIndices.push(farthest);
        remaining.delete(farthest);
      }
    }

    this.supportVectors_ = supportIndices.map((i) => X[i] ?? new Float64Array(nFeatures));
    this.dualCoef_ = new Float64Array(this.supportVectors_.length).fill(1 / this.supportVectors_.length);

    // Compute offset
    let sumK = 0;
    for (const sv of this.supportVectors_) {
      sumK += this._kernel(sv, sv, gamma);
    }
    this.offset_ = -sumK / this.supportVectors_.length;

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.supportVectors_ || !this.dualCoef_) throw new NotFittedError("OneClassSVM is not fitted");
    const nFeatures = this.supportVectors_[0]?.length ?? 0;
    const gamma = this._resolveGamma(nFeatures, 1);
    const out = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let score = this.offset_;
      for (let s = 0; s < this.supportVectors_.length; s++) {
        score += (this.dualCoef_[s] ?? 0) * this._kernel(X[i] ?? new Float64Array(0), this.supportVectors_[s] ?? new Float64Array(0), gamma);
      }
      out[i] = score >= 0 ? 1 : -1;
    }
    return out;
  }

  scoreFunction(X: Float64Array[]): Float64Array {
    if (!this.supportVectors_ || !this.dualCoef_) throw new NotFittedError("OneClassSVM is not fitted");
    const nFeatures = this.supportVectors_[0]?.length ?? 0;
    const gamma = this._resolveGamma(nFeatures, 1);
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let score = this.offset_;
      for (let s = 0; s < this.supportVectors_.length; s++) {
        score += (this.dualCoef_[s] ?? 0) * this._kernel(X[i] ?? new Float64Array(0), this.supportVectors_[s] ?? new Float64Array(0), gamma);
      }
      out[i] = score;
    }
    return out;
  }
}
