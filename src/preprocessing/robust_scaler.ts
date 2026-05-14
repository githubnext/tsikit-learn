/**
 * RobustScaler and MaxAbsScaler.
 * Mirrors sklearn.preprocessing.RobustScaler and MaxAbsScaler.
 */

import { NotFittedError } from "../exceptions.js";

export interface RobustScalerOptions {
  withCentering?: boolean;
  withScaling?: boolean;
  quantileRange?: [number, number];
}

export class RobustScaler {
  withCentering: boolean;
  withScaling: boolean;
  quantileRange: [number, number];

  center_: Float64Array | null = null;
  scale_: Float64Array | null = null;

  constructor(options: RobustScalerOptions = {}) {
    this.withCentering = options.withCentering ?? true;
    this.withScaling = options.withScaling ?? true;
    this.quantileRange = options.quantileRange ?? [25, 75];
  }

  private _percentile(sorted: number[], q: number): number {
    const n = sorted.length;
    const idx = (q / 100) * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const frac = idx - lo;
    return (sorted[lo] ?? 0) * (1 - frac) + (sorted[hi] ?? 0) * frac;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const [qLow, qHigh] = this.quantileRange;

    this.center_ = new Float64Array(p);
    this.scale_ = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const col = Array.from({ length: n }, (_, i) => (X[i] as Float64Array)[j] ?? 0).sort((a, b) => a - b);
      this.center_[j] = this._percentile(col, 50);
      const iqr = this._percentile(col, qHigh) - this._percentile(col, qLow);
      this.scale_[j] = iqr === 0 ? 1 : iqr;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.center_ || !this.scale_) throw new NotFittedError("RobustScaler is not fitted.");
    return X.map(xi => {
      const out = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        let v = xi[j] ?? 0;
        if (this.withCentering) v -= this.center_![j] ?? 0;
        if (this.withScaling) v /= this.scale_![j] ?? 1;
        out[j] = v;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.center_ || !this.scale_) throw new NotFittedError("RobustScaler is not fitted.");
    return X.map(xi => {
      const out = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        let v = xi[j] ?? 0;
        if (this.withScaling) v *= this.scale_![j] ?? 1;
        if (this.withCentering) v += this.center_![j] ?? 0;
        out[j] = v;
      }
      return out;
    });
  }
}

export class MaxAbsScaler {
  maxAbsVals_: Float64Array | null = null;

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    this.maxAbsVals_ = new Float64Array(p);
    for (const xi of X) {
      for (let j = 0; j < p; j++) {
        const abs = Math.abs(xi[j] ?? 0);
        if (abs > (this.maxAbsVals_[j] ?? 0)) this.maxAbsVals_[j] = abs;
      }
    }
    for (let j = 0; j < p; j++) {
      if ((this.maxAbsVals_[j] ?? 0) === 0) this.maxAbsVals_[j] = 1;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbsVals_) throw new NotFittedError("MaxAbsScaler is not fitted.");
    return X.map(xi => Float64Array.from(xi.map((v, j) => v / (this.maxAbsVals_![j] ?? 1))));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbsVals_) throw new NotFittedError("MaxAbsScaler is not fitted.");
    return X.map(xi => Float64Array.from(xi.map((v, j) => v * (this.maxAbsVals_![j] ?? 1))));
  }
}
