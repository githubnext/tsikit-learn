/**
 * Preprocessing extensions: QuantileTransformer, PowerTransformer, KernelCenterer extensions.
 * Mirrors sklearn.preprocessing additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Map each feature to a uniform or normal distribution using quantiles. */
export class QuantileTransformerExt extends BaseEstimator {
  n_quantiles: number;
  output_distribution: "uniform" | "normal";
  quantiles_: Float64Array[] = [];
  references_: Float64Array = new Float64Array(0);

  constructor(params: { n_quantiles?: number; output_distribution?: "uniform" | "normal" } = {}) {
    super();
    this.n_quantiles = params.n_quantiles ?? 1000;
    this.output_distribution = params.output_distribution ?? "uniform";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const nQ = Math.min(this.n_quantiles, n);
    this.references_ = new Float64Array(nQ).map((_, i) => i / (nQ - 1 || 1));
    this.quantiles_ = Array.from({ length: d }, (_, f) => {
      const vals = [...X.map(row => row[f] ?? 0)].sort((a, b) => a - b);
      return new Float64Array(nQ).map((_, i) => {
        const idx = (i / (nQ - 1 || 1)) * (n - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, n - 1);
        return (vals[lo] ?? 0) + (idx - lo) * ((vals[hi] ?? 0) - (vals[lo] ?? 0));
      });
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => {
      const out = new Float64Array(row.length);
      for (let f = 0; f < row.length; f++) {
        const q = this.quantiles_[f]!;
        const v = row[f] ?? 0;
        // Interpolate to find quantile
        let lo = 0;
        let hi = q.length - 1;
        while (lo < hi) {
          const mid = Math.floor((lo + hi) / 2);
          if ((q[mid] ?? 0) < v) lo = mid + 1;
          else hi = mid;
        }
        let quantileVal = lo / (q.length - 1 || 1);
        if (lo > 0 && (q[lo] ?? 0) > v) {
          const frac = (v - (q[lo - 1] ?? 0)) / ((q[lo] ?? 1) - (q[lo - 1] ?? 0) || 1);
          quantileVal = (lo - 1 + frac) / (q.length - 1 || 1);
        }
        out[f] = this.output_distribution === "uniform" ? quantileVal : normalPPF(quantileVal);
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

function normalPPF(p: number): number {
  p = Math.max(1e-7, Math.min(1 - 1e-7, p));
  // Rational approximation for inverse normal CDF
  const c = [2.515517, 0.802853, 0.010328];
  const d = [1.432788, 0.189269, 0.001308];
  const sign = p < 0.5 ? -1 : 1;
  const q = p < 0.5 ? p : 1 - p;
  const t = Math.sqrt(-2 * Math.log(q));
  const num = c[0]! + c[1]! * t + c[2]! * t * t;
  const den = 1 + d[0]! * t + d[1]! * t * t + d[2]! * t * t * t;
  return sign * (t - num / den);
}

/** Robust polynomial features with degree control. */
export class RobustPolynomialFeatures extends BaseEstimator {
  degree: number;
  include_bias: boolean;
  interaction_only: boolean;
  n_output_features_ = 0;

  constructor(params: { degree?: number; include_bias?: boolean; interaction_only?: boolean } = {}) {
    super();
    this.degree = params.degree ?? 2;
    this.include_bias = params.include_bias ?? true;
    this.interaction_only = params.interaction_only ?? false;
  }

  private _getFeatureIndices(d: number): Array<number[]> {
    const combos: Array<number[]> = [];
    if (this.include_bias) combos.push([]);
    const addCombos = (start: number, current: number[], deg: number) => {
      if (deg === 0) { combos.push([...current]); return; }
      for (let i = start; i < d; i++) {
        if (this.interaction_only && current.includes(i)) continue;
        addCombos(i, [...current, i], deg - 1);
      }
    };
    for (let deg = 1; deg <= this.degree; deg++) addCombos(0, [], deg);
    return combos;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const d = X[0]?.length ?? 0;
    const combos = this._getFeatureIndices(d);
    this.n_output_features_ = combos.length;
    return X.map(row => new Float64Array(combos.map(c => c.reduce((p, i) => p * (row[i] ?? 0), 1))));
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.fitTransform(X);
  }
}

/** Clipping transformer: clip feature values to [min, max] range. */
export class ClippingTransformer extends BaseEstimator {
  feature_range: [number, number];
  data_min_: Float64Array = new Float64Array(0);
  data_max_: Float64Array = new Float64Array(0);

  constructor(feature_range: [number, number] = [0, 1]) {
    super();
    this.feature_range = feature_range;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    this.data_min_ = new Float64Array(d).fill(Number.POSITIVE_INFINITY);
    this.data_max_ = new Float64Array(d).fill(Number.NEGATIVE_INFINITY);
    for (const row of X) {
      for (let f = 0; f < d; f++) {
        const v = row[f] ?? 0;
        if (v < (this.data_min_[f] ?? Number.POSITIVE_INFINITY)) this.data_min_[f] = v;
        if (v > (this.data_max_[f] ?? Number.NEGATIVE_INFINITY)) this.data_max_[f] = v;
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const [lo, hi] = this.feature_range;
    return X.map(row => new Float64Array(row).map((v, f) => {
      const mn = this.data_min_[f] ?? 0;
      const mx = this.data_max_[f] ?? 1;
      const scaled = (v - mn) / (mx - mn || 1);
      return lo + Math.max(0, Math.min(1, scaled)) * (hi - lo);
    }));
  }
}

/** Interaction-only feature generator. */
export class InteractionFeatures extends BaseEstimator {
  include_bias: boolean;

  constructor(include_bias = false) {
    super();
    this.include_bias = include_bias;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const d = X[0]?.length ?? 0;
    return X.map(row => {
      const features: number[] = [];
      if (this.include_bias) features.push(1);
      for (let i = 0; i < d; i++) features.push(row[i] ?? 0);
      for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) {
        features.push((row[i] ?? 0) * (row[j] ?? 0));
      }
      return new Float64Array(features);
    });
  }
}
