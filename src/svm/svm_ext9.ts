/**
 * SVM extensions: OneClassSVM, SVR with custom kernels.
 * Mirrors sklearn.svm advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface OneClassSVMParams {
  kernel?: "rbf" | "linear" | "poly";
  nu?: number;
  gamma?: number | "scale" | "auto";
  degree?: number;
  tol?: number;
  max_iter?: number;
}

/** OneClassSVM: unsupervised outlier detection. */
export class OneClassSVM extends BaseEstimator {
  kernel: "rbf" | "linear" | "poly";
  nu: number;
  gamma: number | "scale" | "auto";
  degree: number;
  tol: number;
  max_iter: number;
  support_vectors_: Float64Array[] = [];
  dual_coef_: Float64Array = new Float64Array(0);
  offset_: number = 0;
  X_: Float64Array[] = [];

  constructor(params: OneClassSVMParams = {}) {
    super();
    this.kernel = params.kernel ?? "rbf";
    this.nu = params.nu ?? 0.5;
    this.gamma = params.gamma ?? "scale";
    this.degree = params.degree ?? 3;
    this.tol = params.tol ?? 1e-3;
    this.max_iter = params.max_iter ?? -1;
  }

  private _gamma(X: Float64Array[]): number {
    if (typeof this.gamma === "number") return this.gamma;
    const nf = X[0]?.length ?? 1;
    if (this.gamma === "auto") return 1 / nf;
    // "scale": 1 / (n_features * var(X))
    let variance = 0;
    let mean = 0;
    let cnt = 0;
    for (const xi of X) for (const v of xi) { mean += v; cnt++; }
    mean /= Math.max(cnt, 1);
    for (const xi of X) for (const v of xi) variance += (v - mean) ** 2;
    variance /= Math.max(cnt, 1);
    return 1 / (nf * Math.max(variance, 1e-10));
  }

  private _kernel(a: Float64Array, b: Float64Array, gamma: number): number {
    if (this.kernel === "linear") {
      let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0); return s;
    }
    if (this.kernel === "rbf") {
      let d = 0; for (let k = 0; k < a.length; k++) d += ((a[k] ?? 0) - (b[k] ?? 0)) ** 2; return Math.exp(-gamma * d);
    }
    // poly
    let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0);
    return (gamma * s + 1) ** this.degree;
  }

  fit(X: Float64Array[]): this {
    this.X_ = X;
    const n = X.length;
    const g = this._gamma(X);
    const nu = this.nu;
    // SMO-like simplified training: initialize alpha uniformly
    const alpha = new Float64Array(n).fill(nu / n * 2);
    const maxIter = this.max_iter > 0 ? this.max_iter : 200;
    // Kernel matrix diagonal
    for (let iter = 0; iter < maxIter; iter++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        let fi = 0;
        for (let j = 0; j < n; j++) fi += (alpha[j] ?? 0) * this._kernel(X[j]!, X[i]!, g);
        // Simplified update
        const newA = Math.max(0, Math.min(1 / (n * nu), (alpha[i] ?? 0) + 0.01 * (1 - fi)));
        if (Math.abs(newA - (alpha[i] ?? 0)) > 1e-5) { alpha[i] = newA; changed++; }
      }
      if (changed === 0) break;
    }
    this.dual_coef_ = alpha;
    this.support_vectors_ = X.filter((_, i) => (alpha[i] ?? 0) > 1e-5);
    // Compute offset
    let rhoSum = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      if ((alpha[i] ?? 0) > 1e-5) {
        let k = 0;
        for (let j = 0; j < n; j++) k += (alpha[j] ?? 0) * this._kernel(X[j]!, X[i]!, g);
        rhoSum += k; cnt++;
      }
    }
    this.offset_ = cnt > 0 ? rhoSum / cnt : 0;
    return this;
  }

  decision_function(X: Float64Array[]): Float64Array {
    const g = this._gamma(this.X_);
    return new Float64Array(X.map((xi) => {
      let s = 0;
      for (let j = 0; j < this.X_.length; j++) s += (this.dual_coef_[j] ?? 0) * this._kernel(this.X_[j]!, xi, g);
      return s - this.offset_;
    }));
  }

  predict(X: Float64Array[]): Int32Array {
    const df = this.decision_function(X);
    return new Int32Array(df.map((v) => v >= 0 ? 1 : -1));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

export interface SVCExtParams {
  C?: number;
  kernel?: "rbf" | "linear" | "poly";
  gamma?: number | "scale" | "auto";
  degree?: number;
  tol?: number;
  max_iter?: number;
  probability?: boolean;
}

/** SVC: C-Support Vector Classification. */
export class SVCExt extends BaseEstimator {
  C: number;
  kernel: "rbf" | "linear" | "poly";
  gamma: number | "scale" | "auto";
  degree: number;
  tol: number;
  max_iter: number;
  probability: boolean;
  X_: Float64Array[] = [];
  y_: Int32Array = new Int32Array(0);
  alpha_: Float64Array = new Float64Array(0);
  b_ = 0;
  classes_: Int32Array = new Int32Array(0);

  constructor(params: SVCExtParams = {}) {
    super();
    this.C = params.C ?? 1.0;
    this.kernel = params.kernel ?? "rbf";
    this.gamma = params.gamma ?? "scale";
    this.degree = params.degree ?? 3;
    this.tol = params.tol ?? 1e-3;
    this.max_iter = params.max_iter ?? 200;
    this.probability = params.probability ?? false;
  }

  private _gamma(X: Float64Array[]): number {
    if (typeof this.gamma === "number") return this.gamma;
    const nf = X[0]?.length ?? 1;
    if (this.gamma === "auto") return 1 / nf;
    let v = 0, m = 0, cnt = 0;
    for (const xi of X) for (const val of xi) { m += val; cnt++; }
    m /= Math.max(cnt, 1);
    for (const xi of X) for (const val of xi) v += (val - m) ** 2;
    return 1 / (nf * Math.max(v / cnt, 1e-10));
  }

  private _k(a: Float64Array, b: Float64Array, g: number): number {
    if (this.kernel === "linear") {
      let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0); return s;
    }
    if (this.kernel === "rbf") {
      let d = 0; for (let k = 0; k < a.length; k++) d += ((a[k] ?? 0) - (b[k] ?? 0)) ** 2; return Math.exp(-g * d);
    }
    let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0);
    return (g * s + 1) ** this.degree;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.X_ = X; this.y_ = y;
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const n = X.length;
    const g = this._gamma(X);
    const yBin = new Float64Array(n).map((_, i) => (y[i] ?? 0) === (classes[0] ?? 0) ? -1 : 1);
    const alpha = new Float64Array(n);
    const C = this.C;
    for (let iter = 0; iter < this.max_iter; iter++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        let fi = this.b_;
        for (let j = 0; j < n; j++) fi += (alpha[j] ?? 0) * (yBin[j] ?? 0) * this._k(X[j]!, X[i]!, g);
        const Ei = fi - (yBin[i] ?? 0);
        if ((yBin[i] ?? 0) * Ei < -this.tol && (alpha[i] ?? 0) < C || (yBin[i] ?? 0) * Ei > this.tol && (alpha[i] ?? 0) > 0) {
          const j = (i + 1) % n;
          let fj = this.b_;
          for (let k = 0; k < n; k++) fj += (alpha[k] ?? 0) * (yBin[k] ?? 0) * this._k(X[k]!, X[j]!, g);
          const Ej = fj - (yBin[j] ?? 0);
          const alphaIOld = alpha[i] ?? 0;
          const alphaJOld = alpha[j] ?? 0;
          const L = (yBin[i] ?? 0) === (yBin[j] ?? 0) ? Math.max(0, alphaIOld + alphaJOld - C) : Math.max(0, alphaJOld - alphaIOld);
          const H = (yBin[i] ?? 0) === (yBin[j] ?? 0) ? Math.min(C, alphaIOld + alphaJOld) : Math.min(C, C + alphaJOld - alphaIOld);
          if (L >= H) continue;
          const eta = 2 * this._k(X[i]!, X[j]!, g) - this._k(X[i]!, X[i]!, g) - this._k(X[j]!, X[j]!, g);
          if (eta >= 0) continue;
          let newJ = alphaJOld - (yBin[j] ?? 0) * (Ei - Ej) / eta;
          newJ = Math.max(L, Math.min(H, newJ));
          if (Math.abs(newJ - alphaJOld) < 1e-5) continue;
          alpha[j] = newJ;
          alpha[i] = alphaIOld + (yBin[i] ?? 0) * (yBin[j] ?? 0) * (alphaJOld - newJ);
          changed++;
        }
      }
      if (changed === 0) break;
    }
    this.alpha_ = alpha;
    let bSum = 0, bCnt = 0;
    for (let i = 0; i < n; i++) {
      if ((alpha[i] ?? 0) > 0 && (alpha[i] ?? 0) < C) {
        let f = 0;
        for (let j = 0; j < n; j++) f += (alpha[j] ?? 0) * (yBin[j] ?? 0) * this._k(X[j]!, X[i]!, g);
        bSum += (yBin[i] ?? 0) - f; bCnt++;
      }
    }
    this.b_ = bCnt > 0 ? bSum / bCnt : 0;
    return this;
  }

  decision_function(X: Float64Array[]): Float64Array {
    const g = this._gamma(this.X_);
    const n = this.X_.length;
    const yBin = new Float64Array(n).map((_, i) => (this.y_[i] ?? 0) === (this.classes_[0] ?? 0) ? -1 : 1);
    return new Float64Array(X.map((xi) => {
      let s = this.b_;
      for (let j = 0; j < n; j++) s += (this.alpha_[j] ?? 0) * (yBin[j] ?? 0) * this._k(this.X_[j]!, xi, g);
      return s;
    }));
  }

  predict(X: Float64Array[]): Int32Array {
    const df = this.decision_function(X);
    return new Int32Array(df.map((v) => v >= 0 ? (this.classes_[1] ?? 1) : (this.classes_[0] ?? 0)));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}
