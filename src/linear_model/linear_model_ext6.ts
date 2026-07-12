/**
 * Linear model extensions: TheilSenRegressor, RANSACRegressor.
 * Mirrors sklearn.linear_model robust estimators.
 */

import { BaseEstimator, RegressorMixin } from "../base.js";

export interface TheilSenRegressorParams {
  fit_intercept?: boolean;
  max_subpopulation?: number;
  n_subsamples?: number | null;
  max_iter?: number;
  tol?: number;
}

/** Theil-Sen Estimator: robust regression using median of pairwise slopes. */
export class TheilSenRegressor extends BaseEstimator implements RegressorMixin {
  readonly _estimator_type = "regressor" as const;
  readonly __type = "regressor" as const;
  fit_intercept: boolean;
  max_subpopulation: number;
  n_subsamples: number | null;
  max_iter: number;
  tol: number;
  coef_: Float64Array = new Float64Array(0);
  intercept_ = 0;

  constructor(params: TheilSenRegressorParams = {}) {
    super();
    this.fit_intercept = params.fit_intercept ?? true;
    this.max_subpopulation = params.max_subpopulation ?? 1e4;
    this.n_subsamples = params.n_subsamples ?? null;
    this.max_iter = params.max_iter ?? 300;
    this.tol = params.tol ?? 1e-3;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    const slopes: Float64Array[] = [];
    const cap = Math.min(this.max_subpopulation, (n * (n - 1)) / 2);
    let c = 0;
    for (let i = 0; i < n && c < cap; i++) {
      for (let j = i + 1; j < n && c < cap; j++) {
        const dy = (y[j] ?? 0) - (y[i] ?? 0);
        const sl = new Float64Array(nf);
        for (let k = 0; k < nf; k++) {
          const dx = (X[j]?.[k] ?? 0) - (X[i]?.[k] ?? 0);
          sl[k] = dx !== 0 ? dy / dx : 0;
        }
        slopes.push(sl);
        c++;
      }
    }
    this.coef_ = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      const v = slopes.map((s) => s[k] ?? 0).sort((a, b) => a - b);
      const m = Math.floor(v.length / 2);
      this.coef_[k] = v.length % 2 === 0 ? ((v[m - 1] ?? 0) + (v[m] ?? 0)) / 2 : (v[m] ?? 0);
    }
    if (this.fit_intercept) {
      let ym = 0;
      for (let i = 0; i < n; i++) ym += y[i] ?? 0;
      ym /= n;
      this.intercept_ = ym;
      for (let k = 0; k < nf; k++) {
        let xm = 0;
        for (let i = 0; i < n; i++) xm += X[i]?.[k] ?? 0;
        this.intercept_ -= (this.coef_[k] ?? 0) * (xm / n);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => {
      let v = this.intercept_;
      for (let k = 0; k < xi.length; k++) v += (this.coef_[k] ?? 0) * (xi[k] ?? 0);
      return v;
    }));
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

export interface RANSACRegressorParams {
  max_trials?: number;
  min_samples?: number;
  residual_threshold?: number;
}

/** RANSAC: RANdom SAmple Consensus regressor. */
export class RANSACRegressor extends BaseEstimator implements RegressorMixin {
  readonly _estimator_type = "regressor" as const;
  readonly __type = "regressor" as const;
  max_trials: number;
  min_samples: number;
  residual_threshold: number;
  estimator_: TheilSenRegressor | null = null;
  inlier_mask_: boolean[] = [];

  constructor(params: RANSACRegressorParams = {}) {
    super();
    this.max_trials = params.max_trials ?? 100;
    this.min_samples = params.min_samples ?? 5;
    this.residual_threshold = params.residual_threshold ?? 1.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    let bestMask: boolean[] = new Array(n).fill(false);
    let bestCnt = 0;
    for (let t = 0; t < this.max_trials; t++) {
      const idx: number[] = [];
      for (let i = 0; i < this.min_samples; i++) idx.push((t * 37 + i * 13) % n);
      const Xs = idx.map((i) => X[i]).filter(Boolean) as Float64Array[];
      const ys = new Float64Array(idx.map((i) => y[i] ?? 0));
      const est = new TheilSenRegressor();
      try { est.fit(Xs, ys); } catch { continue; }
      const pred = est.predict(X);
      const mask = Array.from(y).map((yi, i) => Math.abs((yi ?? 0) - (pred[i] ?? 0)) <= this.residual_threshold);
      const cnt = mask.filter(Boolean).length;
      if (cnt > bestCnt) { bestCnt = cnt; bestMask = mask; }
    }
    this.inlier_mask_ = bestMask;
    const Xi = X.filter((_, i) => bestMask[i]);
    const yi = new Float64Array(Array.from(y).filter((_, i) => bestMask[i]));
    this.estimator_ = new TheilSenRegressor();
    this.estimator_.fit(Xi, yi);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimator_) throw new Error("Not fitted");
    return this.estimator_.predict(X);
  }

  score(X: Float64Array[], y: Float64Array): number {
    if (!this.estimator_) throw new Error("Not fitted");
    return this.estimator_.score(X, y);
  }
}
