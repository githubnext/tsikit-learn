/**
 * Sparse random projection with Achlioptas distribution and very sparse transforms.
 */

export class SparseRandomProjection {
  private components_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nComponents: number | 'auto' = 'auto',
    private density: number | 'auto' = 'auto',
    private eps = 0.1
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const nComp = this.nComponents === 'auto'
      ? Math.max(1, Math.ceil(Math.log(n) / this.eps ** 2))
      : this.nComponents;
    const d = this.density === 'auto'
      ? Math.min(1, Math.sqrt(p) / p)
      : this.density;

    // Achlioptas distribution: {+1, 0, -1} with probabilities {d/2, 1-d, d/2}
    this.components_ = Array.from({ length: nComp }, () =>
      new Float64Array(p).map(() => {
        const u = Math.random();
        if (u < d / 2) return Math.sqrt(1 / d);
        if (u < d) return -Math.sqrt(1 / d);
        return 0;
      })
    );
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const nComp = this.components_.length;
    const scale = 1 / Math.sqrt(nComp);
    return X.map(x => new Float64Array(nComp).map((_, j) =>
      scale * this.components_[j]!.reduce((s, v, k) => s + v * (x[k] ?? 0), 0)
    ));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  get components(): Float64Array[] { return this.components_; }
}

export class VerySparseRandomProjection extends SparseRandomProjection {
  constructor(nComponents: number | 'auto' = 'auto', eps = 0.1) {
    // Very sparse: density = 1/sqrt(p)
    super(nComponents, 'auto', eps);
  }
}

export class GaussianRandomProjection {
  private components_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nComponents: number | 'auto' = 'auto',
    private eps = 0.1
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const nComp = this.nComponents === 'auto'
      ? Math.max(1, Math.ceil(Math.log(n) / this.eps ** 2))
      : this.nComponents;
    this.components_ = Array.from({ length: nComp }, () =>
      new Float64Array(p).map(() => this._sampleGaussian() / Math.sqrt(nComp))
    );
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(x => new Float64Array(this.components_.length).map((_, j) =>
      this.components_[j]!.reduce((s, v, k) => s + v * (x[k] ?? 0), 0)
    ));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  private _sampleGaussian(): number {
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
}

export function johnsonLindenstraussMinDim(nSamples: number, eps = 0.1): number {
  return Math.ceil(4 * Math.log(nSamples) / (eps ** 2 / 2 - eps ** 3 / 3));
}

export class SRHTTransform {
  private perm_!: Int32Array;
  private signs_!: Float64Array;
  private fitted_ = false;

  constructor(private nComponents: number) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this.perm_ = new Int32Array(p).map((_, i) => i).sort(() => Math.random() - 0.5) as Int32Array;
    this.signs_ = new Float64Array(p).map(() => Math.random() < 0.5 ? 1 : -1);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = this.perm_.length;
    const nComp = Math.min(this.nComponents, p);
    return X.map(x => {
      // Sign flip + subsample (simplified SRHT without full Hadamard)
      const flipped = new Float64Array(p).map((_, k) => (x[k] ?? 0) * (this.signs_[k] ?? 1));
      return new Float64Array(nComp).map((_, j) => (flipped[this.perm_[j] ?? 0] ?? 0) / Math.sqrt(nComp));
    });
  }
}
