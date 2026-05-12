/**
 * MinMaxScaler — scales features to a given range.
 * Mirrors sklearn.preprocessing.MinMaxScaler.
 */

import { BaseEstimator } from "../base.js";
import { checkArray } from "../utils/validation.js";
import { ValueError } from "../exceptions.js";

export interface MinMaxScalerParams {
  feature_range?: [number, number];
  copy?: boolean;
  clip?: boolean;
}

export class MinMaxScaler extends BaseEstimator {
  feature_range: [number, number];
  copy: boolean;
  clip: boolean;

  data_min_?: Float64Array;
  data_max_?: Float64Array;
  data_range_?: Float64Array;
  scale_?: Float64Array;
  min_?: Float64Array;
  n_features_in_?: number;
  n_samples_seen_?: number;

  constructor(params: MinMaxScalerParams = {}) {
    super();
    this.feature_range = params.feature_range ?? [0, 1];
    this.copy = params.copy ?? true;
    this.clip = params.clip ?? false;
  }

  fit(X: Float64Array[], _y?: Float64Array | Int32Array): this {
    checkArray(X);
    const [rMin, rMax] = this.feature_range;
    if (rMin >= rMax) {
      throw new ValueError(
        `Minimum of desired feature range must be smaller than maximum. Got ${String(this.feature_range)}.`,
      );
    }
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.n_samples_seen_ = n;
    this.n_features_in_ = p;

    const dataMin = new Float64Array(p).fill(Infinity);
    const dataMax = new Float64Array(p).fill(-Infinity);
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        const v = row[j] ?? 0;
        if (v < (dataMin[j] ?? Infinity)) dataMin[j] = v;
        if (v > (dataMax[j] ?? -Infinity)) dataMax[j] = v;
      }
    }
    this.data_min_ = dataMin;
    this.data_max_ = dataMax;
    this.data_range_ = Float64Array.from(dataMax, (v, i) => v - (dataMin[i] ?? 0));
    const rangeScale = rMax - rMin;
    this.scale_ = Float64Array.from(this.data_range_, (v) =>
      v === 0 ? 0 : rangeScale / v,
    );
    this.min_ = Float64Array.from(this.scale_, (v, i) =>
      rMin - v * (dataMin[i] ?? 0),
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    this._check_is_fitted(["scale_", "min_"]);
    const scale = this.scale_ as Float64Array;
    const min = this.min_ as Float64Array;
    const [rMin, rMax] = this.feature_range;
    return X.map((row) => {
      const out = this.copy ? new Float64Array(row) : row;
      for (let j = 0; j < out.length; j++) {
        out[j] = (out[j] ?? 0) * (scale[j] ?? 1) + (min[j] ?? 0);
        if (this.clip) {
          out[j] = Math.max(rMin, Math.min(rMax, out[j] ?? 0));
        }
      }
      return out;
    });
  }

  inverse_transform(X: Float64Array[]): Float64Array[] {
    this._check_is_fitted(["scale_", "min_"]);
    const scale = this.scale_ as Float64Array;
    const min = this.min_ as Float64Array;
    return X.map((row) => {
      const out = new Float64Array(row);
      for (let j = 0; j < out.length; j++) {
        const s = scale[j] ?? 0;
        out[j] = s !== 0 ? ((out[j] ?? 0) - (min[j] ?? 0)) / s : 0;
      }
      return out;
    });
  }

  fit_transform(X: Float64Array[], y?: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
