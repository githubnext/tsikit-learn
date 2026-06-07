/**
 * Winsorizer and QuantileNormalizer — preprocessing transformers.
 */

export class Winsorizer {
  lowerQuantile: number;
  upperQuantile: number;
  lower_: Float64Array | null = null;
  upper_: Float64Array | null = null;

  constructor(lowerQuantile = 0.05, upperQuantile = 0.95) {
    this.lowerQuantile = lowerQuantile;
    this.upperQuantile = upperQuantile;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.lower_ = new Float64Array(p);
    this.upper_ = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const col = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0).sort((a, b) => a - b);
      const loIdx = Math.floor(this.lowerQuantile * n);
      const hiIdx = Math.min(Math.ceil(this.upperQuantile * n), n - 1);
      this.lower_[j] = col[loIdx] ?? 0;
      this.upper_[j] = col[hiIdx] ?? 0;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const lower = this.lower_ as Float64Array;
    const upper = this.upper_ as Float64Array;
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        out[j] = Math.min(Math.max(row[j] ?? 0, lower[j] ?? 0), upper[j] ?? 0);
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => new Float64Array(row));
  }
}

export class QuantileNormalizer {
  nQuantiles: number;
  output_distribution: "normal" | "uniform";
  private references_: Float64Array | null = null;
  private quantiles_: Float64Array[] | null = null;

  constructor(nQuantiles = 1000, output_distribution: "normal" | "uniform" = "normal") {
    this.nQuantiles = nQuantiles;
    this.output_distribution = output_distribution;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.references_ = Float64Array.from(
      { length: this.nQuantiles },
      (_, i) => (i + 0.5) / this.nQuantiles
    );
    this.quantiles_ = [];
    for (let j = 0; j < p; j++) {
      const col = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0).sort((a, b) => a - b);
      const quantiles = this.references_!.map((q) => {
        const idx = q * (n - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return (col[lo] ?? 0) * (hi - idx) + (col[hi] ?? 0) * (idx - lo);
      });
      this.quantiles_.push(quantiles);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const refs = this.references_ as Float64Array;
    const quants = this.quantiles_ as Float64Array[];
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const v = row[j] ?? 0;
        const q = quants[j] ?? new Float64Array(0);
        // Binary search for quantile
        let lo = 0, hi = q.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if ((q[mid] ?? 0) < v) lo = mid + 1; else hi = mid;
        }
        const pct = refs[lo] ?? 0;
        if (this.output_distribution === "normal") {
          out[j] = normalQuantile(pct);
        } else {
          out[j] = pct;
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

function normalQuantile(p: number): number {
  // Rational approximation for normal quantile
  p = Math.max(1e-9, Math.min(1 - 1e-9, p));
  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
  ];
  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
  ];
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
  ];
  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
  ];
  const pLow = 0.02425, pHigh = 1 - pLow;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
           ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  if (p <= pHigh) {
    const q = p - 0.5, r = q * q;
    return (((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q /
           (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
          ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
}

export class VarianceThresholdExt {
  threshold: number;
  private variances_: Float64Array | null = null;
  private supportMask_: Uint8Array | null = null;

  constructor(threshold = 0.0) {
    this.threshold = threshold;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.variances_ = new Float64Array(p);
    const means = new Float64Array(p);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) means[j] += (X[i]?.[j] ?? 0) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) this.variances_[j] += ((X[i]?.[j] ?? 0) - (means[j] ?? 0)) ** 2 / n;
    this.supportMask_ = new Uint8Array(p).map((_, j) => (this.variances_?.[j] ?? 0) > this.threshold ? 1 : 0);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const mask = this.supportMask_ as Uint8Array;
    const cols = Array.from({ length: mask.length }, (_, j) => j).filter((j) => mask[j] === 1);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  get_support(): Uint8Array {
    return this.supportMask_ ?? new Uint8Array(0);
  }
}

export class ClipTransformer {
  min: number;
  max: number;

  constructor(min = 0.0, max = 1.0) {
    this.min = min;
    this.max = max;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => row.map((v) => Math.min(Math.max(v, this.min), this.max)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }
}
