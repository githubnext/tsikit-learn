/**
 * SVM extensions: Structured SVM, Multi-class SVM variants, SVM calibration.
 * Mirrors sklearn.svm additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Margin-based loss functions for SVM training. */
export function hingeLoss(decision: Float64Array, y: Int32Array): number {
  let loss = 0;
  for (let i = 0; i < y.length; i++) {
    const yi = (y[i] ?? 0) === 1 ? 1 : -1;
    loss += Math.max(0, 1 - yi * (decision[i] ?? 0));
  }
  return loss / y.length;
}

export function squaredHingeLoss(decision: Float64Array, y: Int32Array): number {
  let loss = 0;
  for (let i = 0; i < y.length; i++) {
    const yi = (y[i] ?? 0) === 1 ? 1 : -1;
    const margin = 1 - yi * (decision[i] ?? 0);
    loss += Math.max(0, margin) ** 2;
  }
  return loss / y.length;
}

/** Platt scaling for SVM probability calibration. */
export class PlattScaling extends BaseEstimator {
  A = 0;
  B = 0;
  n_iter: number;

  constructor(n_iter = 100) {
    super();
    this.n_iter = n_iter;
  }

  fit(decision: Float64Array, y: Int32Array): this {
    const n = decision.length;
    const prior1 = Array.from(y).filter(v => v === 1).length;
    const prior0 = n - prior1;
    const T = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      T[i] = (y[i] ?? 0) === 1 ? (prior1 + 1) / (prior1 + 2) : 1 / (prior0 + 2);
    }

    let A = 0;
    let B = Math.log((prior0 + 1) / (prior1 + 1));
    const lr = 1e-3;
    for (let iter = 0; iter < this.n_iter; iter++) {
      let gA = 0;
      let gB = 0;
      for (let i = 0; i < n; i++) {
        const fApB = (decision[i] ?? 0) * A + B;
        const p = fApB >= 0 ? Math.exp(-fApB) / (1 + Math.exp(-fApB)) : 1 / (1 + Math.exp(fApB));
        const d = (T[i] ?? 0) - p;
        gA += (decision[i] ?? 0) * d;
        gB += d;
      }
      A += lr * gA;
      B += lr * gB;
    }
    this.A = A;
    this.B = B;
    return this;
  }

  predict_proba(decision: Float64Array): Float64Array {
    return decision.map(f => {
      const fApB = f * this.A + this.B;
      return fApB >= 0 ? Math.exp(-fApB) / (1 + Math.exp(-fApB)) : 1 / (1 + Math.exp(fApB));
    });
  }
}

export interface SVMRBFParams {
  C?: number;
  gamma?: number;
  max_iter?: number;
  tol?: number;
}

/** SVM with RBF kernel using SMO-lite algorithm. */
export class SVMRBF extends BaseEstimator {
  C: number;
  gamma: number;
  max_iter: number;
  tol: number;
  alphas_: Float64Array = new Float64Array(0);
  b_ = 0;
  X_train_: Float64Array[] = [];
  y_train_: Int32Array = new Int32Array(0);
  support_: Int32Array = new Int32Array(0);

  constructor(params: SVMRBFParams = {}) {
    super();
    this.C = params.C ?? 1.0;
    this.gamma = params.gamma ?? 0.1;
    this.max_iter = params.max_iter ?? 100;
    this.tol = params.tol ?? 1e-3;
  }

  private kernel(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let f = 0; f < a.length; f++) s += ((a[f] ?? 0) - (b[f] ?? 0)) ** 2;
    return Math.exp(-this.gamma * s);
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    this.X_train_ = X;
    const yb = new Float64Array(n).map((_, i) => (y[i] ?? 0) === 1 ? 1 : -1);
    this.y_train_ = y;
    const alpha = new Float64Array(n);
    let b = 0;

    for (let iter = 0; iter < this.max_iter; iter++) {
      let nChanged = 0;
      for (let i = 0; i < n; i++) {
        const Ei = this._decision(X, alpha, yb, b, X[i]!) - (yb[i] ?? 0);
        if (((yb[i] ?? 0) * Ei < -this.tol && (alpha[i] ?? 0) < this.C) ||
          ((yb[i] ?? 0) * Ei > this.tol && (alpha[i] ?? 0) > 0)) {
          // Pick j randomly
          let j = Math.floor(Math.random() * n);
          while (j === i) j = Math.floor(Math.random() * n);
          const Ej = this._decision(X, alpha, yb, b, X[j]!) - (yb[j] ?? 0);
          const alphaIOld = alpha[i] ?? 0;
          const alphaJOld = alpha[j] ?? 0;
          let L: number, H: number;
          if ((yb[i] ?? 0) !== (yb[j] ?? 0)) {
            L = Math.max(0, alphaJOld - alphaIOld);
            H = Math.min(this.C, this.C + alphaJOld - alphaIOld);
          } else {
            L = Math.max(0, alphaIOld + alphaJOld - this.C);
            H = Math.min(this.C, alphaIOld + alphaJOld);
          }
          if (L >= H) continue;
          const kii = this.kernel(X[i]!, X[i]!);
          const kij = this.kernel(X[i]!, X[j]!);
          const kjj = this.kernel(X[j]!, X[j]!);
          const eta = 2 * kij - kii - kjj;
          if (eta >= 0) continue;
          alpha[j] = Math.max(L, Math.min(H, alphaJOld - (yb[j] ?? 0) * (Ei - Ej) / eta));
          if (Math.abs((alpha[j] ?? 0) - alphaJOld) < 1e-5) continue;
          alpha[i] = alphaIOld + (yb[i] ?? 0) * (yb[j] ?? 0) * (alphaJOld - (alpha[j] ?? 0));
          const b1 = b - Ei - (yb[i] ?? 0) * ((alpha[i] ?? 0) - alphaIOld) * kii - (yb[j] ?? 0) * ((alpha[j] ?? 0) - alphaJOld) * kij;
          const b2 = b - Ej - (yb[i] ?? 0) * ((alpha[i] ?? 0) - alphaIOld) * kij - (yb[j] ?? 0) * ((alpha[j] ?? 0) - alphaJOld) * kjj;
          b = (b1 + b2) / 2;
          nChanged++;
        }
      }
      if (nChanged === 0) break;
    }
    this.alphas_ = alpha;
    this.b_ = b;
    this.support_ = new Int32Array(Array.from({ length: n }, (_, i) => i).filter(i => (alpha[i] ?? 0) > 1e-5));
    return this;
  }

  private _decision(X: Float64Array[], alpha: Float64Array, y: Float64Array, b: number, x: Float64Array): number {
    let s = b;
    for (let i = 0; i < X.length; i++) {
      if ((alpha[i] ?? 0) > 1e-10) s += (alpha[i] ?? 0) * (y[i] ?? 0) * this.kernel(X[i]!, x);
    }
    return s;
  }

  predict(X: Float64Array[]): Int32Array {
    const yb = new Float64Array(this.X_train_.length).map((_, i) => (this.y_train_[i] ?? 0) === 1 ? 1 : -1);
    return new Int32Array(X.map(x => {
      const d = this._decision(this.X_train_, this.alphas_, yb, this.b_, x);
      return d >= 0 ? 1 : 0;
    }));
  }
}
