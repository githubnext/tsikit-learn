/**
 * Preprocessing extensions: QuantileTransformer, PowerTransformer (Yeo-Johnson), KernelPCA preprocessing.
 * Mirrors sklearn.preprocessing advanced transformers.
 */

import { BaseEstimator } from "../base.js";

export interface QuantileTransformerParams {
  n_quantiles?: number;
  output_distribution?: "uniform" | "normal";
  random_state?: number | null;
}

/** QuantileTransformer: maps data to a uniform or normal distribution. */
export class QuantileTransformerExt extends BaseEstimator {
  n_quantiles: number;
  output_distribution: "uniform" | "normal";
  random_state: number | null;
  quantiles_: Float64Array[] = [];
  references_: Float64Array = new Float64Array(0);
  n_features_in_ = 0;

  constructor(params: QuantileTransformerParams = {}) {
    super();
    this.n_quantiles = params.n_quantiles ?? 1000;
    this.output_distribution = params.output_distribution ?? "uniform";
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    this.n_features_in_ = nf;
    const q = Math.min(this.n_quantiles, n);
    this.references_ = new Float64Array(q);
    for (let i = 0; i < q; i++) this.references_[i] = i / (q - 1);
    this.quantiles_ = [];
    for (let k = 0; k < nf; k++) {
      const col = Array.from({ length: n }, (_, i) => X[i]?.[k] ?? 0).sort((a, b) => a - b);
      const qVals = new Float64Array(q);
      for (let i = 0; i < q; i++) {
        const pos = (i / (q - 1)) * (col.length - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, col.length - 1);
        qVals[i] = (col[lo] ?? 0) + (pos - lo) * ((col[hi] ?? 0) - (col[lo] ?? 0));
      }
      this.quantiles_.push(qVals);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) {
        const qk = this.quantiles_[k];
        if (!qk) { out[k] = 0; continue; }
        const v = xi[k] ?? 0;
        let lo = 0, hi = qk.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if ((qk[mid] ?? 0) < v) lo = mid + 1; else hi = mid;
        }
        let p = lo / (qk.length - 1);
        if (this.output_distribution === "normal") {
          p = Math.max(1e-7, Math.min(1 - 1e-7, p));
          p = this._normalPPF(p);
        }
        out[k] = p;
      }
      return out;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  private _normalPPF(p: number): number {
    // Rational approximation for normal quantile
    const a = [2.515517, 0.802853, 0.010328];
    const b = [1.432788, 0.189269, 0.001308];
    const sign = p < 0.5 ? -1 : 1;
    const t = Math.sqrt(-2 * Math.log(Math.min(p, 1 - p)));
    const num = a[0]! + a[1]! * t + a[2]! * t * t;
    const den = 1 + b[0]! * t + b[1]! * t * t + b[2]! * t * t * t;
    return sign * (t - num / den);
  }
}

export interface RobustScalerExtParams {
  quantile_range?: [number, number];
  with_centering?: boolean;
  with_scaling?: boolean;
}

/** RobustScaler: scale features using statistics robust to outliers. */
export class RobustScalerExt extends BaseEstimator {
  quantile_range: [number, number];
  with_centering: boolean;
  with_scaling: boolean;
  center_: Float64Array = new Float64Array(0);
  scale_: Float64Array = new Float64Array(0);

  constructor(params: RobustScalerExtParams = {}) {
    super();
    this.quantile_range = params.quantile_range ?? [25, 75];
    this.with_centering = params.with_centering ?? true;
    this.with_scaling = params.with_scaling ?? true;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    this.center_ = new Float64Array(nf);
    this.scale_ = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      const col = Array.from({ length: n }, (_, i) => X[i]?.[k] ?? 0).sort((a, b) => a - b);
      const q1 = this._percentile(col, this.quantile_range[0]);
      const q3 = this._percentile(col, this.quantile_range[1]);
      this.center_[k] = this._percentile(col, 50);
      this.scale_[k] = Math.max(q3 - q1, 1e-10);
    }
    return this;
  }

  private _percentile(sorted: number[], p: number): number {
    const pos = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return (sorted[lo] ?? 0) + (pos - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0));
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) {
        let v = xi[k] ?? 0;
        if (this.with_centering) v -= this.center_[k] ?? 0;
        if (this.with_scaling) v /= this.scale_[k] ?? 1;
        out[k] = v;
      }
      return out;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverse_transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) {
        let v = xi[k] ?? 0;
        if (this.with_scaling) v *= this.scale_[k] ?? 1;
        if (this.with_centering) v += this.center_[k] ?? 0;
        out[k] = v;
      }
      return out;
    });
  }
}

export interface MaxAbsScalerParams {
  copy?: boolean;
}

/** MaxAbsScaler: scale each feature by its maximum absolute value. */
export class MaxAbsScaler extends BaseEstimator {
  copy: boolean;
  max_abs_: Float64Array = new Float64Array(0);
  scale_: Float64Array = new Float64Array(0);
  n_samples_seen_ = 0;

  constructor(params: MaxAbsScalerParams = {}) {
    super();
    this.copy = params.copy ?? true;
  }

  fit(X: Float64Array[]): this {
    const nf = X[0]?.length ?? 0;
    this.max_abs_ = new Float64Array(nf);
    for (const xi of X) {
      for (let k = 0; k < nf; k++) {
        const v = Math.abs(xi[k] ?? 0);
        if (v > (this.max_abs_[k] ?? 0)) this.max_abs_![k] = v;
      }
    }
    this.scale_ = new Float64Array(nf);
    for (let k = 0; k < nf; k++) this.scale_[k] = Math.max(this.max_abs_[k] ?? 0, 1e-10);
    this.n_samples_seen_ = X.length;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) out[k] = (xi[k] ?? 0) / (this.scale_[k] ?? 1);
      return out;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverse_transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) out[k] = (xi[k] ?? 0) * (this.scale_[k] ?? 1);
      return out;
    });
  }
}
