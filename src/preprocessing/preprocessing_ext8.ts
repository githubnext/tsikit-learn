/**
 * Preprocessing extensions: MaxAbsScaler, VarianceThreshold, PowerTransformerExt
 * Port of sklearn.preprocessing
 */

import { NotFittedError } from "../exceptions.js";

export class MaxAbsScaler {
  private maxAbs_: Float64Array | null = null;
  private nFeatures_ = 0;

  fit(X: Float64Array[]): this {
    if (X.length === 0) return this;
    this.nFeatures_ = X[0]?.length ?? 0;
    this.maxAbs_ = new Float64Array(this.nFeatures_);
    for (const xi of X) {
      for (let j = 0; j < this.nFeatures_; j++) {
        const absVal = Math.abs(xi[j] ?? 0);
        if (absVal > (this.maxAbs_[j] ?? 0)) this.maxAbs_[j] = absVal;
      }
    }
    for (let j = 0; j < this.nFeatures_; j++) {
      if ((this.maxAbs_[j] ?? 0) === 0) this.maxAbs_[j] = 1;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbs_) throw new NotFittedError("MaxAbsScaler not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.nFeatures_);
      for (let j = 0; j < this.nFeatures_; j++) r[j] = (xi[j] ?? 0) / (this.maxAbs_![j] ?? 1);
      return r;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.maxAbs_) throw new NotFittedError("MaxAbsScaler not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.nFeatures_);
      for (let j = 0; j < this.nFeatures_; j++) r[j] = (xi[j] ?? 0) * (this.maxAbs_![j] ?? 1);
      return r;
    });
  }
}

export class VarianceThresholdExt {
  threshold: number;
  private variance_: Float64Array | null = null;
  private mask_: boolean[] | null = null;

  constructor(opts: { threshold?: number } = {}) {
    this.threshold = opts.threshold ?? 0.0;
  }

  fit(X: Float64Array[]): this {
    if (X.length === 0) return this;
    const p = X[0]?.length ?? 0;
    const mean = new Float64Array(p);
    for (const xi of X) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0) / X.length;
    this.variance_ = new Float64Array(p);
    for (const xi of X) for (let j = 0; j < p; j++) this.variance_[j] = (this.variance_[j] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) ** 2 / X.length;
    this.mask_ = Array.from({ length: p }, (_, j) => (this.variance_![j] ?? 0) > this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.mask_) throw new NotFittedError("VarianceThresholdExt not fitted.");
    const selectedCols = this.mask_.reduce<number[]>((acc, v, i) => { if (v) acc.push(i); return acc; }, []);
    return X.map(xi => {
      const r = new Float64Array(selectedCols.length);
      for (let j = 0; j < selectedCols.length; j++) r[j] = xi[selectedCols[j]!] ?? 0;
      return r;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  getSupport(): boolean[] {
    if (!this.mask_) throw new NotFittedError("VarianceThresholdExt not fitted.");
    return [...this.mask_];
  }
}

export class KBinsDiscretizerExt {
  nBins: number;
  strategy: "uniform" | "quantile" | "kmeans";
  encode: "ordinal" | "onehot-dense";

  private binEdges_: Float64Array[] | null = null;
  private nFeatures_ = 0;

  constructor(opts: {
    nBins?: number;
    strategy?: "uniform" | "quantile" | "kmeans";
    encode?: "ordinal" | "onehot-dense";
  } = {}) {
    this.nBins = opts.nBins ?? 5;
    this.strategy = opts.strategy ?? "quantile";
    this.encode = opts.encode ?? "ordinal";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;
    this.binEdges_ = [];
    for (let j = 0; j < this.nFeatures_; j++) {
      const vals = X.map(xi => xi[j] ?? 0).sort((a, b) => a - b);
      const edges = new Float64Array(this.nBins + 1);
      if (this.strategy === "uniform") {
        const min = vals[0] ?? 0;
        const max = vals[vals.length - 1] ?? 0;
        for (let b = 0; b <= this.nBins; b++) edges[b] = min + (b / this.nBins) * (max - min);
      } else {
        for (let b = 0; b <= this.nBins; b++) {
          const idx = Math.floor((b / this.nBins) * (n - 1));
          edges[b] = vals[idx] ?? 0;
        }
      }
      edges[0] = -Number.POSITIVE_INFINITY;
      edges[this.nBins] = Number.POSITIVE_INFINITY;
      this.binEdges_.push(edges);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.binEdges_) throw new NotFittedError("KBinsDiscretizerExt not fitted.");
    if (this.encode === "ordinal") {
      return X.map(xi => {
        const r = new Float64Array(this.nFeatures_);
        for (let j = 0; j < this.nFeatures_; j++) {
          const edges = this.binEdges_![j]!;
          let bin = this.nBins - 1;
          for (let b = 1; b < edges.length; b++) {
            if ((xi[j] ?? 0) < (edges[b] ?? Number.POSITIVE_INFINITY)) { bin = b - 1; break; }
          }
          r[j] = bin;
        }
        return r;
      });
    }
    return X.map(xi => {
      const r = new Float64Array(this.nFeatures_ * this.nBins);
      for (let j = 0; j < this.nFeatures_; j++) {
        const edges = this.binEdges_![j]!;
        let bin = this.nBins - 1;
        for (let b = 1; b < edges.length; b++) {
          if ((xi[j] ?? 0) < (edges[b] ?? Number.POSITIVE_INFINITY)) { bin = b - 1; break; }
        }
        r[j * this.nBins + bin] = 1;
      }
      return r;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class AdditiveChi2SamplerExt {
  sampleSteps: number;
  sampleInterval: number;

  constructor(opts: { sampleSteps?: number; sampleInterval?: number } = {}) {
    this.sampleSteps = opts.sampleSteps ?? 2;
    this.sampleInterval = opts.sampleInterval ?? 0.4;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    const nOutputFeats = p * (2 * this.sampleSteps + 1);
    return X.map(xi => {
      const r = new Float64Array(nOutputFeats);
      for (let j = 0; j < p; j++) {
        const xj = Math.max(xi[j] ?? 0, 0);
        r[j * (2 * this.sampleSteps + 1)] = Math.sqrt(xj * this.sampleInterval);
        for (let s = 1; s <= this.sampleSteps; s++) {
          const c = Math.sqrt(2 * xj * this.sampleInterval);
          const angle = s * this.sampleInterval * Math.log(xj + 1e-15);
          r[j * (2 * this.sampleSteps + 1) + 2 * s - 1] = c * Math.cos(angle);
          r[j * (2 * this.sampleSteps + 1) + 2 * s] = c * Math.sin(angle);
        }
      }
      return r;
    });
  }
}
