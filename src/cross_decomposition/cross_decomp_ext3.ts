/**
 * Cross-decomposition extensions: NIPALS, PLS2.
 * Mirrors sklearn.cross_decomposition advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface NIPALSParams {
  n_components?: number;
  max_iter?: number;
  tol?: number;
}

/** NIPALS: Nonlinear Iterative Partial Least Squares algorithm. */
export class NIPALS extends BaseEstimator {
  n_components: number;
  max_iter: number;
  tol: number;
  x_weights_: Float64Array[] = [];
  y_weights_: Float64Array[] = [];
  x_loadings_: Float64Array[] = [];
  y_loadings_: Float64Array[] = [];
  x_scores_: Float64Array[] = [];
  y_scores_: Float64Array[] = [];
  x_mean_: Float64Array = new Float64Array(0);
  y_mean_: Float64Array = new Float64Array(0);
  n_features_in_ = 0;

  constructor(params: NIPALSParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.max_iter = params.max_iter ?? 500;
    this.tol = params.tol ?? 1e-6;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const px = X[0]?.length ?? 0, py = Y[0]?.length ?? 0;
    this.n_features_in_ = px;
    this.x_mean_ = new Float64Array(px);
    this.y_mean_ = new Float64Array(py);
    for (let k = 0; k < px; k++) for (const xi of X) this.x_mean_[k] = (this.x_mean_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < py; k++) for (const yi of Y) this.y_mean_[k] = (this.y_mean_[k] ?? 0) + (yi[k] ?? 0);
    for (let k = 0; k < px; k++) this.x_mean_[k] = (this.x_mean_[k] ?? 0) / n;
    for (let k = 0; k < py; k++) this.y_mean_[k] = (this.y_mean_[k] ?? 0) / n;
    let Xr = X.map((xi) => new Float64Array(px).map((_, k) => (xi[k] ?? 0) - (this.x_mean_[k] ?? 0)));
    let Yr = Y.map((yi) => new Float64Array(py).map((_, k) => (yi[k] ?? 0) - (this.y_mean_[k] ?? 0)));
    for (let c = 0; c < this.n_components; c++) {
      // NIPALS iteration
      let u = Yr.map((yi) => yi[0] ?? 0);
      let w = new Float64Array(px), q = new Float64Array(py), t = new Float64Array(n);
      for (let iter = 0; iter < this.max_iter; iter++) {
        // w = X'u / ||X'u||
        for (let j = 0; j < px; j++) { let s = 0; for (let i = 0; i < n; i++) s += (Xr[i]?.[j] ?? 0) * (u[i] ?? 0); w[j] = s; }
        let wn = 0; for (const v of w) wn += v * v; wn = Math.sqrt(wn); if (wn > 1e-10) for (let j = 0; j < px; j++) w[j] = (w[j] ?? 0) / wn;
        // t = Xw
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < px; j++) s += (Xr[i]?.[j] ?? 0) * (w[j] ?? 0); t[i] = s; }
        // q = Y't / ||Y't||
        for (let j = 0; j < py; j++) { let s = 0; for (let i = 0; i < n; i++) s += (Yr[i]?.[j] ?? 0) * (t[i] ?? 0); q[j] = s; }
        let qn = 0; for (const v of q) qn += v * v; qn = Math.sqrt(qn); if (qn > 1e-10) for (let j = 0; j < py; j++) q[j] = (q[j] ?? 0) / qn;
        // u = Yq
        const uNew = new Float64Array(n);
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < py; j++) s += (Yr[i]?.[j] ?? 0) * (q[j] ?? 0); uNew[i] = s; }
        let diff = 0; for (let i = 0; i < n; i++) diff += (uNew[i] ?? 0 - (u[i] ?? 0)) ** 2;
        u = Array.from(uNew);
        if (Math.sqrt(diff) < this.tol) break;
      }
      // Deflate
      const pLoading = new Float64Array(px);
      const tn2 = t.reduce((s, v) => s + v * v, 0);
      if (tn2 > 1e-10) {
        for (let j = 0; j < px; j++) { let s = 0; for (let i = 0; i < n; i++) s += (Xr[i]?.[j] ?? 0) * (t[i] ?? 0); pLoading[j] = s / tn2; }
      }
      Xr = Xr.map((xi, i) => new Float64Array(px).map((_, j) => (xi[j] ?? 0) - (t[i] ?? 0) * (pLoading[j] ?? 0)));
      Yr = Yr.map((yi, i) => new Float64Array(py).map((_, j) => (yi[j] ?? 0) - (q[j] ?? 0) * u[i]!));
      this.x_weights_.push(w);
      this.y_weights_.push(q);
      this.x_loadings_.push(pLoading);
      this.y_loadings_.push(q);
      this.x_scores_.push(t);
      this.y_scores_.push(new Float64Array(u));
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const nc = this.n_components;
    const px = this.n_features_in_;
    const Xc = X.map((xi) => new Float64Array(px).map((_, k) => (xi[k] ?? 0) - (this.x_mean_[k] ?? 0)));
    return Xc.map((xi) => new Float64Array(nc).map((_, c) => {
      let s = 0;
      const w = this.x_weights_[c];
      if (w) for (let k = 0; k < px; k++) s += (w[k] ?? 0) * (xi[k] ?? 0);
      return s;
    }));
  }

  fit_transform(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
    return this.fit(X, Y).transform(X);
  }
}

export interface CanonicalCorrelationExtParams {
  n_components?: number;
  regularization?: number;
}

/** Canonical Correlation Analysis (CCA) extension. */
export class CCAExt extends BaseEstimator {
  n_components: number;
  regularization: number;
  x_weights_: Float64Array[] = [];
  y_weights_: Float64Array[] = [];
  x_mean_: Float64Array = new Float64Array(0);
  y_mean_: Float64Array = new Float64Array(0);
  n_features_in_ = 0;

  constructor(params: CanonicalCorrelationExtParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.regularization = params.regularization ?? 1e-4;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const px = X[0]?.length ?? 0, py = Y[0]?.length ?? 0;
    this.n_features_in_ = px;
    this.x_mean_ = new Float64Array(px);
    this.y_mean_ = new Float64Array(py);
    for (let k = 0; k < px; k++) for (const xi of X) this.x_mean_[k] = (this.x_mean_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < py; k++) for (const yi of Y) this.y_mean_[k] = (this.y_mean_[k] ?? 0) + (yi[k] ?? 0);
    for (let k = 0; k < px; k++) this.x_mean_[k] = (this.x_mean_[k] ?? 0) / n;
    for (let k = 0; k < py; k++) this.y_mean_[k] = (this.y_mean_[k] ?? 0) / n;
    const Xc = X.map((xi) => new Float64Array(px).map((_, k) => (xi[k] ?? 0) - (this.x_mean_[k] ?? 0)));
    const Yc = Y.map((yi) => new Float64Array(py).map((_, k) => (yi[k] ?? 0) - (this.y_mean_[k] ?? 0)));
    // Covariance matrices
    const cov = (A: Float64Array[], B: Float64Array[], pa: number, pb: number): Float64Array[] => {
      const C = Array.from({ length: pa }, () => new Float64Array(pb));
      for (let i = 0; i < n; i++) for (let a = 0; a < pa; a++) for (let b = 0; b < pb; b++) C[a]![b] = (C[a]![b] ?? 0) + (A[i]?.[a] ?? 0) * (B[i]?.[b] ?? 0);
      for (let a = 0; a < pa; a++) for (let b = 0; b < pb; b++) C[a]![b] = (C[a]![b] ?? 0) / n;
      return C;
    };
    const Sxx = cov(Xc, Xc, px, px);
    const Syy = cov(Yc, Yc, py, py);
    const Sxy = cov(Xc, Yc, px, py);
    // Regularize diagonals
    for (let i = 0; i < px; i++) Sxx[i]![i] = (Sxx[i]![i] ?? 0) + this.regularization;
    for (let i = 0; i < py; i++) Syy[i]![i] = (Syy[i]![i] ?? 0) + this.regularization;
    // Power iteration for canonical directions
    const nc = Math.min(this.n_components, px, py);
    for (let c = 0; c < nc; c++) {
      let wx = new Float64Array(px).map((_, i) => i === c ? 1 : 0.01);
      for (let iter = 0; iter < 50; iter++) {
        // wx = Sxx^-1 * Sxy * Syy^-1 * Sxy' * wx (power iteration approximation)
        const Sxy_wx = new Float64Array(py).map((_, j) => { let s = 0; for (let k = 0; k < px; k++) s += (Sxy[k]?.[j] ?? 0) * (wx[k] ?? 0); return s; });
        const Syy_inv_v = new Float64Array(py).map((_, j) => (Sxy_wx[j] ?? 0) / (Syy[j]?.[j] ?? 1));
        const Sxyt_v = new Float64Array(px).map((_, i) => { let s = 0; for (let j = 0; j < py; j++) s += (Sxy[i]?.[j] ?? 0) * (Syy_inv_v[j] ?? 0); return s; });
        const newWx = new Float64Array(px).map((_, i) => (Sxyt_v[i] ?? 0) / (Sxx[i]?.[i] ?? 1));
        let norm = 0; for (const v of newWx) norm += v * v; norm = Math.sqrt(norm);
        if (norm > 1e-10) for (let i = 0; i < px; i++) newWx[i] = (newWx[i] ?? 0) / norm;
        let diff = 0; for (let i = 0; i < px; i++) diff += ((newWx[i] ?? 0) - (wx[i] ?? 0)) ** 2;
        wx = newWx;
        if (Math.sqrt(diff) < 1e-8) break;
      }
      const wy = new Float64Array(py).map((_, j) => { let s = 0; for (let i = 0; i < px; i++) s += (Sxy[i]?.[j] ?? 0) * (wx[i] ?? 0); return s; });
      let wyn = 0; for (const v of wy) wyn += v * v; wyn = Math.sqrt(wyn);
      if (wyn > 1e-10) for (let j = 0; j < py; j++) wy[j] = (wy[j] ?? 0) / wyn;
      this.x_weights_.push(wx);
      this.y_weights_.push(wy);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const px = this.n_features_in_;
    const Xc = X.map((xi) => new Float64Array(px).map((_, k) => (xi[k] ?? 0) - (this.x_mean_[k] ?? 0)));
    return Xc.map((xi) => new Float64Array(this.n_components).map((_, c) => {
      let s = 0;
      const w = this.x_weights_[c];
      if (w) for (let k = 0; k < px; k++) s += (w[k] ?? 0) * (xi[k] ?? 0);
      return s;
    }));
  }

  fit_transform(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
    return this.fit(X, Y).transform(X);
  }
}
