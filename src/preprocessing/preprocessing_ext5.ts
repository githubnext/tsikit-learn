/**
 * Additional preprocessing transformers.
 * MaxAbsScaler, RobustScaler extensions.
 * Mirrors sklearn.preprocessing extras.
 */

import { NotFittedError } from "../exceptions.js";

export class MaxAbsScaler {
  maxAbsValues_: Float64Array | null = null;
  scale_: Float64Array | null = null;

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    const maxAbs = new Float64Array(nFeatures);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        const v = Math.abs(row[j] ?? 0);
        if (v > (maxAbs[j] ?? 0)) maxAbs[j] = v;
      }
    }
    this.maxAbsValues_ = maxAbs;
    this.scale_ = maxAbs.slice();
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbsValues_) throw new NotFittedError("MaxAbsScaler is not fitted");
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const s = this.maxAbsValues_![j] ?? 0;
        out[j] = s !== 0 ? (row[j] ?? 0) / s : 0;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbsValues_) throw new NotFittedError("MaxAbsScaler is not fitted");
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        out[j] = (row[j] ?? 0) * (this.maxAbsValues_![j] ?? 0);
      }
      return out;
    });
  }
}

export class RobustScalerExt {
  center_: Float64Array | null = null;
  scale_: Float64Array | null = null;
  withCentering: boolean;
  withScaling: boolean;
  quantileRange: [number, number];

  constructor(
    options: {
      withCentering?: boolean;
      withScaling?: boolean;
      quantileRange?: [number, number];
    } = {},
  ) {
    this.withCentering = options.withCentering ?? true;
    this.withScaling = options.withScaling ?? true;
    this.quantileRange = options.quantileRange ?? [25, 75];
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    const n = X.length;
    this.center_ = new Float64Array(nFeatures);
    this.scale_ = new Float64Array(nFeatures);

    for (let j = 0; j < nFeatures; j++) {
      const col = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0).sort((a, b) => a - b);
      if (this.withCentering) {
        this.center_[j] = this._quantile(col, 50);
      }
      if (this.withScaling) {
        const q1 = this._quantile(col, this.quantileRange[0]);
        const q3 = this._quantile(col, this.quantileRange[1]);
        this.scale_[j] = q3 - q1 !== 0 ? q3 - q1 : 1;
      } else {
        this.scale_[j] = 1;
      }
    }
    return this;
  }

  private _quantile(sorted: number[], q: number): number {
    const idx = (q / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const frac = idx - lo;
    return (sorted[lo] ?? 0) * (1 - frac) + (sorted[hi] ?? 0) * frac;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.center_ || !this.scale_) throw new NotFittedError("RobustScalerExt is not fitted");
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const v = (row[j] ?? 0) - (this.withCentering ? (this.center_![j] ?? 0) : 0);
        out[j] = v / (this.scale_![j] ?? 1);
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.center_ || !this.scale_) throw new NotFittedError("RobustScalerExt is not fitted");
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        out[j] = (row[j] ?? 0) * (this.scale_![j] ?? 1) + (this.withCentering ? (this.center_![j] ?? 0) : 0);
      }
      return out;
    });
  }
}
