/**
 * Normalizer — normalize samples individually to unit norm.
 * Mirrors sklearn.preprocessing.Normalizer.
 */

import { BaseEstimator } from "../base.js";
import { ValueError } from "../exceptions.js";

export type NormType = "l1" | "l2" | "max";

export interface NormalizerParams {
  norm?: NormType;
  copy?: boolean;
}

export class Normalizer extends BaseEstimator {
  norm: NormType;
  copy: boolean;

  constructor(params: NormalizerParams = {}) {
    super();
    this.norm = params.norm ?? "l2";
    this.copy = params.copy ?? true;
  }

  fit(_X: Float64Array[], _y?: Float64Array | Int32Array): this {
    // Normalizer is stateless — nothing to fit
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => {
      const out = this.copy ? new Float64Array(row) : row;
      const norm = this._computeNorm(out);
      if (norm === 0) return out;
      for (let j = 0; j < out.length; j++) {
        out[j] = (out[j] ?? 0) / norm;
      }
      return out;
    });
  }

  fit_transform(
    X: Float64Array[],
    _y?: Float64Array | Int32Array,
  ): Float64Array[] {
    return this.transform(X);
  }

  private _computeNorm(row: Float64Array): number {
    switch (this.norm) {
      case "l1": {
        let sum = 0;
        for (const v of row) sum += Math.abs(v);
        return sum;
      }
      case "l2": {
        let sum = 0;
        for (const v of row) sum += v * v;
        return Math.sqrt(sum);
      }
      case "max": {
        let max = 0;
        for (const v of row) max = Math.max(max, Math.abs(v));
        return max;
      }
      default:
        throw new ValueError(`Unknown norm: ${String(this.norm)}`);
    }
  }
}
