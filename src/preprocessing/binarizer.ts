/**
 * Binarizer, FunctionTransformer, and QuantileTransformer.
 * Mirrors sklearn.preprocessing.Binarizer, FunctionTransformer, QuantileTransformer.
 */

import { NotFittedError } from "../exceptions.js";

export interface BinarizerOptions {
  threshold?: number;
}

/**
 * Binarizer — thresholds numerical features to get boolean values.
 */
export class Binarizer {
  threshold: number;
  nFeatureIn_: number = 0;

  constructor(options: BinarizerOptions = {}) {
    this.threshold = options.threshold ?? 0.0;
  }

  fit(X: Float64Array[]): this {
    this.nFeatureIn_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    return X.map((xi) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++)
        row[j]! = (xi[j] ?? 0) > this.threshold ? 1 : 0;
      return row;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface FunctionTransformerOptions {
  func?: ((X: Float64Array[]) => Float64Array[]) | null;
  inverseFunc?: ((X: Float64Array[]) => Float64Array[]) | null;
  validate?: boolean;
  acceptSparse?: boolean;
  checkInverse?: boolean;
  featureNamesOut?: string | null;
}

/**
 * FunctionTransformer — constructs a transformer from an arbitrary callable.
 */
export class FunctionTransformer {
  func: ((X: Float64Array[]) => Float64Array[]) | null;
  inverseFunc: ((X: Float64Array[]) => Float64Array[]) | null;
  validate: boolean;
  nFeatureIn_: number = 0;

  constructor(options: FunctionTransformerOptions = {}) {
    this.func = options.func ?? null;
    this.inverseFunc = options.inverseFunc ?? null;
    this.validate = options.validate ?? false;
  }

  fit(X: Float64Array[]): this {
    this.nFeatureIn_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.func === null) return X.map((xi) => xi.slice());
    return this.func(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.inverseFunc === null) return X.map((xi) => xi.slice());
    return this.inverseFunc(X);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface QuantileTransformerOptions {
  nQuantiles?: number;
  outputDistribution?: "uniform" | "normal";
  ignoreImplicitZeros?: boolean;
  subsample?: number;
  randomState?: number;
  copy?: boolean;
}

/**
 * QuantileTransformer — transforms features to follow a uniform or normal distribution.
 */
export class QuantileTransformer {
  nQuantiles: number;
  outputDistribution: "uniform" | "normal";
  subsample: number;
  randomState: number;
  nFeatureIn_: number = 0;
  nQuantiles_: number = 0;

  quantiles_: Float64Array[] | null = null;
  references_: Float64Array | null = null;

  constructor(options: QuantileTransformerOptions = {}) {
    this.nQuantiles = options.nQuantiles ?? 1000;
    this.outputDistribution = options.outputDistribution ?? "uniform";
    this.subsample = options.subsample ?? 100000;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeatureIn_ = p;
    this.nQuantiles_ = Math.min(this.nQuantiles, n);

    // Compute quantiles for each feature
    this.quantiles_ = Array.from({ length: p }, (_, j) => {
      const vals = Array.from({ length: n }, (_, i) => X[i]![j] ?? 0).sort(
        (a, b) => a - b,
      );
      const qs = new Float64Array(this.nQuantiles_);
      for (let q = 0; q < this.nQuantiles_; q++) {
        const pos = (q / (this.nQuantiles_ - 1)) * (vals.length - 1);
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        const frac = pos - lo;
        qs[q]! = (vals[lo] ?? 0) * (1 - frac) + (vals[hi] ?? 0) * frac;
      }
      return qs;
    });

    // Reference quantiles (uniform [0,1] grid)
    this.references_ = new Float64Array(this.nQuantiles_);
    for (let q = 0; q < this.nQuantiles_; q++)
      this.references_[q]! = q / (this.nQuantiles_ - 1);

    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.quantiles_ || !this.references_)
      throw new NotFittedError("QuantileTransformer is not fitted");
    const p = this.nFeatureIn_;
    const nQ = this.nQuantiles_;

    return X.map((xi) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const v = xi[j] ?? 0;
        const qs = this.quantiles_![j]!;

        // Find position via binary search
        let lo = 0;
        let hi = nQ - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if ((qs[mid] ?? 0) < v) lo = mid + 1;
          else hi = mid;
        }

        let quantile: number;
        if (lo === 0) {
          quantile = 0;
        } else if (lo >= nQ) {
          quantile = 1;
        } else {
          const q0 = qs[lo - 1] ?? 0;
          const q1 = qs[lo] ?? 0;
          const r0 = this.references_![lo - 1] ?? 0;
          const r1 = this.references_![lo] ?? 0;
          if (q1 - q0 < 1e-12) {
            quantile = r0;
          } else {
            quantile = r0 + ((v - q0) / (q1 - q0)) * (r1 - r0);
          }
        }
        quantile = Math.max(0, Math.min(1, quantile));

        if (this.outputDistribution === "normal") {
          // Approximate probit (inverse normal CDF)
          quantile = Math.max(1e-7, Math.min(1 - 1e-7, quantile));
          row[j]! = this._probit(quantile);
        } else {
          row[j]! = quantile;
        }
      }
      return row;
    });
  }

  private _probit(p: number): number {
    // Rational approximation for the inverse normal CDF (Beasley-Springer-Moro)
    const a = [2.515517, 0.802853, 0.010328];
    const b = [1.432788, 0.189269, 0.001308];
    const sign = p < 0.5 ? -1 : 1;
    const q = p < 0.5 ? p : 1 - p;
    const t = Math.sqrt(-2 * Math.log(q));
    const num = a[0]! + t * (a[1]! + t * a[2]!);
    const den = 1 + t * (b[0]! + t * (b[1]! + t * b[2]!));
    return sign * (t - num / den);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.quantiles_ || !this.references_)
      throw new NotFittedError("QuantileTransformer is not fitted");
    const p = this.nFeatureIn_;
    const nQ = this.nQuantiles_;

    return X.map((xi) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let q = xi[j] ?? 0;
        if (this.outputDistribution === "normal") {
          // CDF of standard normal
          q = this._normCDF(q);
        }
        q = Math.max(0, Math.min(1, q));

        const qs = this.quantiles_![j]!;
        const refs = this.references_!;

        // Find position in references
        let lo = 0;
        let hi = nQ - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if ((refs[mid] ?? 0) < q) lo = mid + 1;
          else hi = mid;
        }

        if (lo === 0) {
          row[j]! = qs[0] ?? 0;
        } else if (lo >= nQ) {
          row[j]! = qs[nQ - 1] ?? 0;
        } else {
          const r0 = refs[lo - 1] ?? 0;
          const r1 = refs[lo] ?? 0;
          const q0 = qs[lo - 1] ?? 0;
          const q1 = qs[lo] ?? 0;
          if (r1 - r0 < 1e-12) {
            row[j]! = q0;
          } else {
            row[j]! = q0 + ((q - r0) / (r1 - r0)) * (q1 - q0);
          }
        }
      }
      return row;
    });
  }

  private _normCDF(x: number): number {
    return (
      0.5 * (1 + Math.sign(x) * Math.sqrt(1 - Math.exp((-2 * x * x) / Math.PI)))
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
