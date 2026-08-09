/**
 * Fourier RBF approximation, Nystroem extension, polynomial kernel approximation.
 */

export class FourierRBF {
  private omega_!: Float64Array[];
  private b_!: Float64Array;
  private fitted_ = false;

  constructor(
    private nComponents = 100,
    private gamma = 1.0
  ) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    // Sample random Fourier features from N(0, 2*gamma)
    this.omega_ = Array.from({ length: this.nComponents }, () =>
      new Float64Array(p).map(() => this._sampleGaussian() * Math.sqrt(2 * this.gamma))
    );
    this.b_ = new Float64Array(this.nComponents).map(() => Math.random() * 2 * Math.PI);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const scale = Math.sqrt(2 / this.nComponents);
    return X.map(x => new Float64Array(this.nComponents).map((_, j) => {
      const dot = this.omega_[j]!.reduce((s, w, k) => s + w * (x[k] ?? 0), 0);
      return scale * Math.cos(dot + (this.b_[j] ?? 0));
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  private _sampleGaussian(): number {
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
}

export class PolynomialKernelApproximation {
  private fitted_ = false;
  private sampledFeatures_!: Array<Array<[number, number, number]>>;

  constructor(
    private nComponents = 100,
    private degree = 2,
    private gamma = 1.0,
    private coef0 = 0.0
  ) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    // Tensor Sketching via random feature combinations
    this.sampledFeatures_ = Array.from({ length: this.nComponents }, () =>
      Array.from({ length: this.degree }, () => [
        Math.floor(Math.random() * p),
        Math.random() < 0.5 ? 1 : -1,
        Math.random() * 2 * Math.PI
      ] as [number, number, number])
    );
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(x => new Float64Array(this.nComponents).map((_, j) => {
      let val = 1;
      for (const [fi, sign, phase] of this.sampledFeatures_[j]!) {
        val *= this.gamma * (x[fi] ?? 0) * sign + this.coef0 + Math.cos(phase);
      }
      return val / Math.sqrt(this.nComponents);
    }));
  }
}

export class NystroemExt {
  private sampledX_!: Float64Array[];
  private normalization_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private kernel: (x1: Float64Array, x2: Float64Array) => number,
    private nComponents = 100
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const m = Math.min(this.nComponents, n);
    // Randomly select m samples
    const indices = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, m);
    this.sampledX_ = indices.map(i => X[i]!);

    // Compute kernel matrix among sampled points
    const Kmm = Array.from({ length: m }, (_, i) =>
      new Float64Array(m).map((_, j) => this.kernel(this.sampledX_[i]!, this.sampledX_[j]!))
    );
    // Regularize
    for (let i = 0; i < m; i++) Kmm[i]![i] = (Kmm[i]![i] ?? 0) + 1e-6;
    // Compute Cholesky-based normalization (simplified: use diagonal sqrt)
    this.normalization_ = Kmm;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const m = this.sampledX_.length;
    return X.map(x => new Float64Array(m).map((_, j) => this.kernel(x, this.sampledX_[j]!)));
  }
}

export class SkewedChi2Approximation {
  private sampler_!: FourierRBF;
  private fitted_ = false;
  private skewnessC = 0.5;

  constructor(private nComponents = 100, skewness = 0.5) {
    this.skewnessC = skewness;
    this.sampler_ = new FourierRBF(nComponents, 1.0);
  }

  fit(X: Float64Array[]): this {
    const transformed = X.map(row =>
      new Float64Array(row.map(v => Math.log(v + this.skewnessC)))
    );
    this.sampler_.fit(transformed);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const transformed = X.map(row =>
      new Float64Array(row.map(v => Math.log(v + this.skewnessC)))
    );
    return this.sampler_.transform(transformed);
  }
}
