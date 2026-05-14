/**
 * KBinsDiscretizer for preprocessing.
 * Mirrors sklearn.preprocessing.KBinsDiscretizer.
 */

import { NotFittedError } from "../exceptions.js";

export type KBinsStrategy = "uniform" | "quantile" | "kmeans";
export type KBinsEncode = "onehot" | "ordinal" | "onehot-dense";

export interface KBinsDiscretizerOptions {
  nBins?: number | number[];
  encode?: KBinsEncode;
  strategy?: KBinsStrategy;
  dtype?: "float32" | "float64";
}

export class KBinsDiscretizer {
  nBins: number | number[];
  encode: KBinsEncode;
  strategy: KBinsStrategy;

  binEdges_: Float64Array[] | null = null;
  nBins_: Int32Array | null = null;

  constructor(opts: KBinsDiscretizerOptions = {}) {
    this.nBins = opts.nBins ?? 5;
    this.encode = opts.encode ?? "onehot-dense";
    this.strategy = opts.strategy ?? "quantile";
  }

  private getNBinsForFeature(f: number): number {
    if (Array.isArray(this.nBins)) return this.nBins[f] ?? 5;
    return this.nBins as number;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    const n = X.length;
    this.nBins_ = new Int32Array(nFeatures);
    this.binEdges_ = [];

    for (let f = 0; f < nFeatures; f++) {
      const values = X.map((xi) => xi[f] ?? 0).sort((a, b) => a - b);
      const nBins = this.getNBinsForFeature(f);
      this.nBins_[f]! = nBins;

      let edges: number[];
      if (this.strategy === "quantile") {
        edges = Array.from({ length: nBins + 1 }, (_, i) => {
          const pos = (i / nBins) * (n - 1);
          const lo = Math.floor(pos);
          const hi = Math.ceil(pos);
          const frac = pos - lo;
          return (values[lo] ?? 0) * (1 - frac) + (values[hi] ?? 0) * frac;
        });
      } else if (this.strategy === "uniform") {
        const min = values[0] ?? 0;
        const max = values[n - 1] ?? 0;
        const step = (max - min) / nBins;
        edges = Array.from({ length: nBins + 1 }, (_, i) => min + i * step);
      } else {
        // kmeans: use quantile as approximation
        edges = Array.from({ length: nBins + 1 }, (_, i) => {
          const pos = (i / nBins) * (n - 1);
          const lo = Math.floor(pos);
          const hi = Math.ceil(pos);
          const frac = pos - lo;
          return (values[lo] ?? 0) * (1 - frac) + (values[hi] ?? 0) * frac;
        });
      }

      // Remove duplicate edges
      const unique = [...new Set(edges)];
      if (unique.length < 2) unique.push((unique[0] ?? 0) + 1);
      this.binEdges_.push(Float64Array.from(unique));
    }
    return this;
  }

  private binFeature(value: number, edges: Float64Array): number {
    const nBins = edges.length - 1;
    if (value <= (edges[0] ?? 0)) return 0;
    if (value >= (edges[nBins] ?? 0)) return nBins - 1;
    let lo = 0;
    let hi = nBins;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((edges[mid] ?? 0) <= value) lo = mid + 1;
      else hi = mid;
    }
    return Math.min(lo - 1, nBins - 1);
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.binEdges_) throw new NotFittedError("KBinsDiscretizer");
    const nFeatures = X[0]?.length ?? 0;

    if (this.encode === "ordinal") {
      return X.map((xi) =>
        Float64Array.from({ length: nFeatures }, (_, f) =>
          this.binFeature(xi[f] ?? 0, this.binEdges_![f] as Float64Array),
        ),
      );
    }

    // onehot-dense (and onehot)
    const totalCols = (this.nBins_ as Int32Array).reduce((s, b) => s + b, 0);
    return X.map((xi) => {
      const row = new Float64Array(totalCols);
      let offset = 0;
      for (let f = 0; f < nFeatures; f++) {
        const edges = this.binEdges_![f] as Float64Array;
        const binIdx = this.binFeature(xi[f] ?? 0, edges);
        row[offset + binIdx]! = 1;
        offset += (this.nBins_![f] ?? 0);
      }
      return row;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }

  inverseTransform(Xt: Float64Array[]): Float64Array[] {
    if (!this.binEdges_) throw new NotFittedError("KBinsDiscretizer");
    const nFeatures = this.binEdges_.length;
    return Xt.map((xi) =>
      Float64Array.from({ length: nFeatures }, (_, f) => {
        const edges = this.binEdges_![f] as Float64Array;
        const binIdx = Math.round(xi[f] ?? 0);
        const lo = edges[binIdx] ?? 0;
        const hi = edges[binIdx + 1] ?? lo;
        return (lo + hi) / 2;
      }),
    );
  }
}
