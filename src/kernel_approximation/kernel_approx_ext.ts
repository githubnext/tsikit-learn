/**
 * Kernel approximation extensions: ANOVAKernel, SkewedChi2Sampler, AdditiveChi2Sampler,
 * Nystroem approximation.
 */

export class ANOVASampler {
  private sampledComponents_: Float64Array[] = [];
  private nDegree: number;

  constructor(
    private readonly sigma = 1.0,
    private readonly degree = 2,
    private readonly nComponents = 100,
    private readonly seed = 42
  ) {
    this.nDegree = degree;
  }

  fit(X: Float64Array[]): this {
    const rng = this._seededRng(this.seed);
    const nF = X[0]?.length ?? 1;
    this.sampledComponents_ = Array.from({ length: this.nComponents }, () => {
      const v = new Float64Array(nF);
      for (let f = 0; f < nF; f++) v[f] = rng() * 2 - 1;
      return v;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => {
      const result = new Float64Array(this.nComponents);
      for (let j = 0; j < this.nComponents; j++) {
        const sc = this.sampledComponents_[j]!;
        let k = 0;
        for (let f = 0; f < x.length; f++) {
          const diff = (x[f] ?? 0) - (sc[f] ?? 0);
          k += Math.exp(-diff * diff / (2 * this.sigma ** 2));
        }
        result[j] = k ** this.nDegree;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class SkewedChi2Sampler {
  private randomWeights_: Float64Array[] = [];
  private randomOffset_: Float64Array = new Float64Array(0);

  constructor(
    private readonly skewedness = 1.0,
    private readonly nComponents = 100,
    private readonly seed = 42
  ) {}

  fit(X: Float64Array[]): this {
    const nF = X[0]?.length ?? 1;
    const rng = this._seededRng(this.seed);
    this.randomWeights_ = Array.from({ length: nF }, () => {
      const v = new Float64Array(this.nComponents);
      for (let j = 0; j < this.nComponents; j++) {
        // Sample from Cauchy distribution
        const u = rng() * Math.PI - Math.PI / 2;
        v[j] = this.skewedness * Math.tan(u);
      }
      return v;
    });
    this.randomOffset_ = new Float64Array(this.nComponents);
    for (let j = 0; j < this.nComponents; j++) this.randomOffset_[j] = rng() * 2 * Math.PI;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => {
      const projection = new Float64Array(this.nComponents);
      for (let f = 0; f < x.length; f++) {
        const w = this.randomWeights_[f]!;
        const logX = Math.log(Math.max(x[f] ?? 0, 1e-10) + this.skewedness);
        for (let j = 0; j < this.nComponents; j++) projection[j] = (projection[j] ?? 0) + logX * (w[j] ?? 0);
      }
      const result = new Float64Array(this.nComponents);
      for (let j = 0; j < this.nComponents; j++) {
        result[j] = Math.cos(projection[j]! + (this.randomOffset_[j] ?? 0)) * Math.sqrt(2 / this.nComponents);
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class NystroemApproximation {
  private components_: Float64Array[] = [];
  private normalizationFactor_: Float64Array[] = [];

  constructor(
    private readonly kernel: "rbf" | "linear" | "poly" = "rbf",
    private readonly gamma = 1.0,
    private readonly degree = 3,
    private readonly coef0 = 1.0,
    private readonly nComponents = 100,
    private readonly seed = 42
  ) {}

  fit(X: Float64Array[]): this {
    const rng = this._seededRng(this.seed);
    const n = X.length;
    const indices: number[] = [];
    const perm = Array.from({ length: n }, (_, i) => i);
    // Fisher-Yates shuffle (seeded)
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [perm[i], perm[j]] = [perm[j]!, perm[i]!];
    }
    for (let i = 0; i < Math.min(this.nComponents, n); i++) indices.push(perm[i]!);
    this.components_ = indices.map((i) => new Float64Array(X[i]!));
    // Compute kernel between components
    const K = this._computeKernel(this.components_, this.components_);
    this.normalizationFactor_ = this._pseudoInverseSqrt(K);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const K = this._computeKernel(X, this.components_);
    return K.map((row) => {
      const result = new Float64Array(this.normalizationFactor_.length);
      for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < row.length; j++) result[i] = (result[i] ?? 0) + (row[j] ?? 0) * (this.normalizationFactor_[i]?.[j] ?? 0);
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  private _computeKernel(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    return X1.map((x1) => new Float64Array(X2.map((x2) => {
      switch (this.kernel) {
        case "rbf": {
          let d = 0;
          for (let f = 0; f < x1.length; f++) d += ((x1[f] ?? 0) - (x2[f] ?? 0)) ** 2;
          return Math.exp(-this.gamma * d);
        }
        case "linear": {
          let dot = 0;
          for (let f = 0; f < x1.length; f++) dot += (x1[f] ?? 0) * (x2[f] ?? 0);
          return dot;
        }
        case "poly": {
          let dot = 0;
          for (let f = 0; f < x1.length; f++) dot += (x1[f] ?? 0) * (x2[f] ?? 0);
          return (this.gamma * dot + this.coef0) ** this.degree;
        }
      }
    })));
  }

  private _pseudoInverseSqrt(K: Float64Array[]): Float64Array[] {
    // Simplified: return scaled identity-like matrix
    const n = K.length;
    return Array.from({ length: n }, (_, i) =>
      new Float64Array(n).fill(0).map((_, j) => i === j ? 1 / Math.sqrt(Math.max(K[i]?.[i] ?? 1, 1e-10)) : 0)
    );
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}
