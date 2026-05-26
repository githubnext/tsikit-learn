/**
 * Random projection extensions: SparseRandomProjection, GaussianRandomProjection,
 * JohnsonLindenstrauss lemma utilities.
 */

export function johnsonLindenstraussMinDim(nSamples: number, eps = 0.1): number {
  return Math.ceil(4 * Math.log(nSamples) / (eps ** 2 / 2 - eps ** 3 / 3));
}

export class GaussianRandomProjectionExt {
  private components_: Float64Array[] = [];
  private nFeaturesIn_ = 0;

  constructor(
    private readonly nComponents: number | "auto" = "auto",
    private readonly eps = 0.1,
    private readonly seed = 42
  ) {}

  fit(X: Float64Array[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    const n = X.length;
    const k = this.nComponents === "auto" ? johnsonLindenstraussMinDim(n, this.eps) : this.nComponents;
    const rng = this._seededRng(this.seed);
    this.components_ = Array.from({ length: k }, () => {
      const row = new Float64Array(this.nFeaturesIn_);
      for (let f = 0; f < this.nFeaturesIn_; f++) {
        // Box-Muller
        const u1 = Math.max(rng(), 1e-10);
        const u2 = rng();
        row[f] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) / Math.sqrt(k);
      }
      return row;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(this.components_.map((comp) => {
      let dot = 0;
      for (let f = 0; f < x.length; f++) dot += (x[f] ?? 0) * (comp[f] ?? 0);
      return dot;
    })));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class SparseRandomProjectionExt {
  private components_: Float64Array[] = [];
  private nFeaturesIn_ = 0;

  constructor(
    private readonly nComponents: number | "auto" = "auto",
    private readonly density: number | "auto" = "auto",
    private readonly eps = 0.1,
    private readonly seed = 42
  ) {}

  fit(X: Float64Array[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    const n = X.length;
    const k = this.nComponents === "auto" ? johnsonLindenstraussMinDim(n, this.eps) : this.nComponents;
    const d = this.density === "auto" ? 1 / Math.sqrt(this.nFeaturesIn_) : this.density;
    const rng = this._seededRng(this.seed);
    const scale = Math.sqrt(1 / (d * k));
    this.components_ = Array.from({ length: k }, () => {
      const row = new Float64Array(this.nFeaturesIn_);
      for (let f = 0; f < this.nFeaturesIn_; f++) {
        const u = rng();
        if (u < d / 2) row[f] = scale;
        else if (u < d) row[f] = -scale;
        // else 0 (sparse)
      }
      return row;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(this.components_.map((comp) => {
      let dot = 0;
      for (let f = 0; f < x.length; f++) dot += (x[f] ?? 0) * (comp[f] ?? 0);
      return dot;
    })));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export function estimateJLTransformDimension(nSamples: number, eps: number): number {
  return johnsonLindenstraussMinDim(nSamples, eps);
}
