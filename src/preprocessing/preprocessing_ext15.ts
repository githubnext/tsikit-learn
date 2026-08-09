/**
 * Polynomial interaction features, SplineTransformer, and PowerTransformer extensions.
 */

export class SplineTransformerExt {
  private knots_!: Float64Array;
  private degree: number;
  private fitted_ = false;

  constructor(private nKnots = 5, degree = 3, private includeIntercept = false) {
    this.degree = degree;
  }

  fit(X: Float64Array[]): this {
    const values = X.map(row => row[0] ?? 0).sort((a, b) => a - b);
    const n = values.length;
    this.knots_ = new Float64Array(this.nKnots).map((_, i) => {
      const q = (i + 1) / (this.nKnots + 1);
      return values[Math.floor(q * n)] ?? 0;
    });
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => {
      const x = row[0] ?? 0;
      const features: number[] = this.includeIntercept ? [1] : [];
      // B-spline basis functions approximation
      for (let k = 0; k < this.knots_.length + this.degree - 1; k++) {
        features.push(this._bsplineBasis(x, k, this.degree));
      }
      return new Float64Array(features);
    });
  }

  private _bsplineBasis(x: number, k: number, d: number): number {
    // Truncated power basis: (x - t)_+^d
    const t = this.knots_[k - Math.floor(d / 2)] ?? 0;
    return d === 0 ? (x >= t ? 1 : 0) : Math.max(0, x - t) ** d / (this.knots_.length ** d + 1e-10);
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get nOutputsPerFeature(): number { return this.knots_.length + this.degree - 1 + (this.includeIntercept ? 1 : 0); }
}

export class KBinsDiscretizer {
  private binEdges_!: Float64Array[];
  private fitted_ = false;

  constructor(private nBins = 5, private encode: 'onehot' | 'ordinal' = 'ordinal', private strategy: 'uniform' | 'quantile' | 'kmeans' = 'quantile') {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.binEdges_ = Array.from({ length: p }, (_, j) => {
      const col = X.map(row => row[j] ?? 0).sort((a, b) => a - b);
      const edges = new Float64Array(this.nBins + 1);
      if (this.strategy === 'uniform') {
        const min = col[0] ?? 0, max = col[n - 1] ?? 1;
        for (let b = 0; b <= this.nBins; b++) edges[b] = min + b * (max - min) / this.nBins;
      } else {
        for (let b = 0; b <= this.nBins; b++) {
          const q = b / this.nBins;
          edges[b] = col[Math.min(Math.floor(q * n), n - 1)] ?? 0;
        }
      }
      return edges;
    });
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0;
    return X.map(row => {
      if (this.encode === 'ordinal') {
        return new Float64Array(p).map((_, j) => {
          const x = row[j] ?? 0;
          const edges = this.binEdges_[j]!;
          let bin = 0;
          for (let b = 1; b < edges.length; b++) if (x >= (edges[b] ?? 0)) bin = b;
          return Math.min(bin, this.nBins - 1);
        });
      }
      // One-hot encoding
      const features: number[] = [];
      for (let j = 0; j < p; j++) {
        const x = row[j] ?? 0;
        const edges = this.binEdges_[j]!;
        let bin = 0;
        for (let b = 1; b < edges.length; b++) if (x >= (edges[b] ?? 0)) bin = b;
        bin = Math.min(bin, this.nBins - 1);
        for (let b = 0; b < this.nBins; b++) features.push(b === bin ? 1 : 0);
      }
      return new Float64Array(features);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
}
