/**
 * Non-negative Matrix Factorization (NMF).
 * Mirrors sklearn.decomposition.NMF.
 * Uses multiplicative update rules.
 */

import { NotFittedError } from "../exceptions.js";

function mulUpdate(
  X: Float64Array[],
  W: Float64Array[],
  H: Float64Array[],
  alpha: number,
  maxIter: number,
): void {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const k = H.length;
  const eps = 1e-10;

  for (let iter = 0; iter < maxIter; iter++) {
    // Update H
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < p; j++) {
        let num = 0;
        let den = 0;
        for (let i = 0; i < n; i++) {
          const wic = (W[i] ?? new Float64Array(k))[c] ?? 0;
          const xij = (X[i] ?? new Float64Array(p))[j] ?? 0;
          num += wic * xij;
          let whij = 0;
          for (let l = 0; l < k; l++) {
            whij +=
              ((W[i] ?? new Float64Array(k))[l] ?? 0) *
              ((H[l] ?? new Float64Array(p))[j] ?? 0);
          }
          den += wic * whij;
        }
        const hjc = (H[c] ?? new Float64Array(p))[j] ?? 0;
        (H[c] ?? new Float64Array(p))[j] =
          (hjc * (num + eps)) / (den + alpha + eps);
      }
    }

    // Update W
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        let num = 0;
        let den = 0;
        for (let j = 0; j < p; j++) {
          const hjc = (H[c] ?? new Float64Array(p))[j] ?? 0;
          const xij = (X[i] ?? new Float64Array(p))[j] ?? 0;
          num += xij * hjc;
          let whij = 0;
          for (let l = 0; l < k; l++) {
            whij +=
              ((W[i] ?? new Float64Array(k))[l] ?? 0) *
              ((H[l] ?? new Float64Array(p))[j] ?? 0);
          }
          den += whij * hjc;
        }
        const wic = (W[i] ?? new Float64Array(k))[c] ?? 0;
        (W[i] ?? new Float64Array(k))[c] =
          (wic * (num + eps)) / (den + alpha + eps);
      }
    }
  }
}

export class NMF {
  nComponents: number;
  maxIter: number;
  tol: number;
  alpha: number;

  components_: Float64Array[] | null = null;
  reconstructionErr_: number = 0;

  constructor(
    options: {
      nComponents?: number;
      maxIter?: number;
      tol?: number;
      alpha?: number;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.alpha = options.alpha ?? 0.0;
  }

  fit(X: Float64Array[]): this {
    this._fitTransform(X);
    return this;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this._fitTransform(X);
  }

  private _fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = Math.min(this.nComponents, n, p);

    const W: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = Math.random() * 0.1 + 0.01;
      return row;
    });
    const H: Float64Array[] = Array.from({ length: k }, () => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j] = Math.random() * 0.1 + 0.01;
      return row;
    });

    mulUpdate(X, W, H, this.alpha, this.maxIter);

    // Compute reconstruction error
    let err = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        let approx = 0;
        for (let c = 0; c < k; c++) {
          approx +=
            ((W[i] ?? new Float64Array(k))[c] ?? 0) *
            ((H[c] ?? new Float64Array(p))[j] ?? 0);
        }
        const diff = ((X[i] ?? new Float64Array(p))[j] ?? 0) - approx;
        err += diff * diff;
      }
    }
    this.reconstructionErr_ = Math.sqrt(err);
    this.components_ = H;
    return W;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null) throw new NotFittedError("NMF");
    const n = X.length;
    const k = this.components_.length;

    const W: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = Math.random() * 0.1 + 0.01;
      return row;
    });
    const H = this.components_;

    mulUpdate(X, W, H, this.alpha, this.maxIter);
    return W;
  }
}
