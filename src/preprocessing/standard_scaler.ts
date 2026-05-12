/**
 * StandardScaler — zero-mean, unit-variance normalization.
 * Mirrors sklearn.preprocessing.StandardScaler.
 */

import { BaseEstimator, TransformerMixin } from "../base.js";
import { ValueError } from "../exceptions.js";
import { checkArray, checkFeaturesConsistency } from "../utils/validation.js";

export interface StandardScalerParams {
  copy?: boolean;
  with_mean?: boolean;
  with_std?: boolean;
}

export class StandardScaler extends BaseEstimator {
  copy: boolean;
  with_mean: boolean;
  with_std: boolean;

  mean_?: Float64Array;
  scale_?: Float64Array;
  var_?: Float64Array;
  n_features_in_?: number;
  n_samples_seen_?: number;

  constructor(params: StandardScalerParams = {}) {
    super();
    this.copy = params.copy ?? true;
    this.with_mean = params.with_mean ?? true;
    this.with_std = params.with_std ?? true;
  }

  fit(X: Float64Array[], _y?: Float64Array | Int32Array): this {
    checkArray(X);
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.n_samples_seen_ = n;
    this.n_features_in_ = p;

    const mean = new Float64Array(p);
    const M2 = new Float64Array(p);

    // Welford's online algorithm for mean and variance
    for (let i = 0; i < n; i++) {
      const row = X[i] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const x = row[j] ?? 0;
        const delta = x - (mean[j] ?? 0);
        mean[j] = (mean[j] ?? 0) + delta / (i + 1);
        M2[j] = (M2[j] ?? 0) + delta * (x - (mean[j] ?? 0));
      }
    }

    this.mean_ = mean;
    const variance =
      n > 1 ? Float64Array.from(M2, (v) => v / (n - 1)) : new Float64Array(p);
    this.var_ = variance;
    this.scale_ = Float64Array.from(variance, (v) => Math.sqrt(v) || 1.0);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    this._check_is_fitted(["mean_", "scale_"]);
    checkFeaturesConsistency(X, X); // just shape check
    const mean = this.mean_ as Float64Array;
    const scale = this.scale_ as Float64Array;
    return X.map((row) => {
      const out = this.copy ? new Float64Array(row) : row;
      for (let j = 0; j < out.length; j++) {
        if (this.with_mean) out[j] = (out[j] ?? 0) - (mean[j] ?? 0);
        if (this.with_std) out[j] = (out[j] ?? 0) / (scale[j] ?? 1);
      }
      return out;
    });
  }

  inverse_transform(X: Float64Array[]): Float64Array[] {
    this._check_is_fitted(["mean_", "scale_"]);
    const mean = this.mean_ as Float64Array;
    const scale = this.scale_ as Float64Array;
    return X.map((row) => {
      const out = new Float64Array(row);
      for (let j = 0; j < out.length; j++) {
        if (this.with_std) out[j] = (out[j] ?? 0) * (scale[j] ?? 1);
        if (this.with_mean) out[j] = (out[j] ?? 0) + (mean[j] ?? 0);
      }
      return out;
    });
  }

  fit_transform(
    X: Float64Array[],
    y?: Float64Array | Int32Array,
  ): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
