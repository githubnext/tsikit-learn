/**
 * Additional preprocessing transformers: PowerTransformer, QuantileTransformer,
 * Binarizer, FunctionTransformer, KBinsDiscretizer.
 * Mirrors sklearn.preprocessing.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Apply a power transform to make data more Gaussian-like.
 * Supports Box-Cox and Yeo-Johnson methods.
 * Mirrors sklearn.preprocessing.PowerTransformer.
 */
export class PowerTransformer {
  method: "yeo-johnson" | "box-cox";
  standardize: boolean;

  lambdas_: Float64Array | null = null;
  means_: Float64Array | null = null;
  stds_: Float64Array | null = null;

  constructor(
    options: { method?: "yeo-johnson" | "box-cox"; standardize?: boolean } = {},
  ) {
    this.method = options.method ?? "yeo-johnson";
    this.standardize = options.standardize ?? true;
  }

  private _yeojohnson(x: number, lam: number): number {
    if (x >= 0) {
      if (Math.abs(lam) < 1e-10) return Math.log(x + 1);
      return ((x + 1) ** lam - 1) / lam;
    }
    if (Math.abs(lam - 2) < 1e-10) return -Math.log(-x + 1);
    return -((-x + 1) ** (2 - lam) - 1) / (2 - lam);
  }

  private _boxcox(x: number, lam: number): number {
    if (x <= 0) throw new Error("Box-Cox requires positive data");
    if (Math.abs(lam) < 1e-10) return Math.log(x);
    return (x ** lam - 1) / lam;
  }

  private _optimalLambda(col: Float64Array): number {
    // Grid search for lambda that maximizes log-likelihood (simplified)
    const lambdas = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    let bestLam = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const lam of lambdas) {
      try {
        const transformed = Float64Array.from(col, (x) =>
          this.method === "box-cox"
            ? this._boxcox(x, lam)
            : this._yeojohnson(x, lam),
        );
        let mean = 0;
        for (let i = 0; i < transformed.length; i++)
          mean += transformed[i] ?? 0;
        mean /= transformed.length;
        let variance = 0;
        for (let i = 0; i < transformed.length; i++) {
          variance += ((transformed[i] ?? 0) - mean) ** 2;
        }
        variance /= transformed.length;
        // Log-likelihood proxy: -variance
        const score = -(variance || 1e-15);
        if (score > bestScore) {
          bestScore = score;
          bestLam = lam;
        }
      } catch {
        /* skip */
      }
    }
    return bestLam;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.lambdas_ = new Float64Array(p);
    this.means_ = new Float64Array(p);
    this.stds_ = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const col = Float64Array.from({ length: n }, (_, i) => X[i]![j] ?? 0);
      this.lambdas_[j] = this._optimalLambda(col);
      if (this.standardize) {
        const lam = this.lambdas_[j] ?? 0;
        const t = Float64Array.from(col, (x) =>
          this.method === "box-cox"
            ? this._boxcox(x, lam)
            : this._yeojohnson(x, lam),
        );
        let mean = 0;
        for (let i = 0; i < n; i++) mean += t[i] ?? 0;
        mean /= n;
        let variance = 0;
        for (let i = 0; i < n; i++) variance += ((t[i] ?? 0) - mean) ** 2;
        variance /= n;
        this.means_[j] = mean;
        this.stds_[j] = Math.sqrt(variance) || 1;
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.lambdas_ === null) throw new NotFittedError();
    const p = this.lambdas_.length;
    return X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const lam = this.lambdas_![j] ?? 0;
        let val =
          this.method === "box-cox"
            ? this._boxcox(xi[j] ?? 0, lam)
            : this._yeojohnson(xi[j] ?? 0, lam);
        if (this.standardize) {
          val = (val - (this.means_![j] ?? 0)) / ((this.stds_![j] ?? 1) || 1);
        }
        out[j] = val;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.lambdas_ === null) throw new NotFittedError();
    const p = this.lambdas_.length;
    return X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let val = xi[j] ?? 0;
        if (this.standardize) {
          val = val * ((this.stds_![j] ?? 1) || 1) + (this.means_![j] ?? 0);
        }
        const lam = this.lambdas_![j] ?? 0;
        if (this.method === "yeo-johnson") {
          out[j] = this._invYeoJohnson(val, lam);
        } else {
          out[j] = this._invBoxCox(val, lam);
        }
      }
      return out;
    });
  }

  private _invYeoJohnson(y: number, lam: number): number {
    if (y >= 0) {
      if (Math.abs(lam) < 1e-10) return Math.exp(y) - 1;
      return (y * lam + 1) ** (1 / lam) - 1;
    }
    if (Math.abs(lam - 2) < 1e-10) return 1 - Math.exp(-y);
    return 1 - (-(2 - lam) * y + 1) ** (1 / (2 - lam));
  }

  private _invBoxCox(y: number, lam: number): number {
    if (Math.abs(lam) < 1e-10) return Math.exp(y);
    return (y * lam + 1) ** (1 / lam);
  }
}

/**
 * Transform features using quantile information (maps to uniform or normal distribution).
 * Mirrors sklearn.preprocessing.QuantileTransformer.
 */
export class QuantileTransformer {
  nQuantiles: number;
  outputDistribution: "uniform" | "normal";
  subsample: number;

  quantiles_: Float64Array[] | null = null;
  referenceQuantiles_: Float64Array | null = null;

  constructor(
    options: {
      nQuantiles?: number;
      outputDistribution?: "uniform" | "normal";
      subsample?: number;
    } = {},
  ) {
    this.nQuantiles = options.nQuantiles ?? 1000;
    this.outputDistribution = options.outputDistribution ?? "uniform";
    this.subsample = options.subsample ?? 100000;
  }

  private _normalPPF(p: number): number {
    // Approximation of normal PPF (probit)
    const a = [
      -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
      1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
    ];
    const b = [
      -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
      6.680131188771972e1, -1.328068155288572e1,
    ];
    const c = [
      -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
      -2.549732539343734, 4.374664141464968, 2.938163982698783,
    ];
    const d = [
      7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
      3.754408661907416,
    ];
    const pLow = 0.02425;
    const pHigh = 1 - pLow;
    if (p < pLow) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (
        (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
          c[5]!) /
        ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
      );
    }
    if (p <= pHigh) {
      const q = p - 0.5;
      const r = q * q;
      return (
        ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r +
          a[5]!) *
          q) /
        (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
      );
    }
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(
        ((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
        c[5]!
      ) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
    );
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const nQ = Math.min(this.nQuantiles, n);
    this.referenceQuantiles_ = Float64Array.from(
      { length: nQ },
      (_, i) => i / (nQ - 1),
    );
    this.quantiles_ = [];
    for (let j = 0; j < p; j++) {
      const col = Array.from({ length: n }, (_, i) => X[i]![j] ?? 0).sort(
        (a, b) => a - b,
      );
      const quants = new Float64Array(nQ);
      for (let q = 0; q < nQ; q++) {
        const pos = (q / (nQ - 1)) * (n - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, n - 1);
        const frac = pos - lo;
        quants[q] = (col[lo] ?? 0) * (1 - frac) + (col[hi] ?? 0) * frac;
      }
      this.quantiles_[j] = quants;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.quantiles_ === null || this.referenceQuantiles_ === null) {
      throw new NotFittedError();
    }
    const p = this.quantiles_.length;
    const nQ = this.referenceQuantiles_.length;
    return X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const val = xi[j] ?? 0;
        const quants = this.quantiles_![j] ?? new Float64Array(0);
        // Binary search for val in quants
        let lo = 0;
        let hi = nQ - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if ((quants[mid] ?? 0) < val) lo = mid + 1;
          else hi = mid;
        }
        let u = lo / (nQ - 1);
        if (lo > 0 && lo < nQ) {
          const qlo = quants[lo - 1] ?? 0;
          const qhi = quants[lo] ?? 0;
          const range = qhi - qlo;
          if (range > 1e-15) u = (lo - 1 + (val - qlo) / range) / (nQ - 1);
        }
        u = Math.max(1e-7, Math.min(1 - 1e-7, u));
        out[j] = this.outputDistribution === "normal" ? this._normalPPF(u) : u;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Binarize data (set feature values to 0 or 1) according to a threshold.
 * Mirrors sklearn.preprocessing.Binarizer.
 */
export class Binarizer {
  threshold: number;

  constructor(options: { threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0.0;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = (X[0] ?? new Float64Array(0)).length;
    return X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++)
        out[j] = (xi[j] ?? 0) > this.threshold ? 1 : 0;
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Constructs a transformer from an arbitrary callable.
 * Mirrors sklearn.preprocessing.FunctionTransformer.
 */
export class FunctionTransformer {
  func: ((X: Float64Array[]) => Float64Array[]) | null;
  inverseFunc: ((X: Float64Array[]) => Float64Array[]) | null;
  validate: boolean;

  constructor(
    options: {
      func?: ((X: Float64Array[]) => Float64Array[]) | null;
      inverseFunc?: ((X: Float64Array[]) => Float64Array[]) | null;
      validate?: boolean;
    } = {},
  ) {
    this.func = options.func ?? null;
    this.inverseFunc = options.inverseFunc ?? null;
    this.validate = options.validate ?? false;
  }

  fit(_X: Float64Array[]): this {
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
