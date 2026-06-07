/**
 * Extended kernel approximations — Fastfood, Nystroem extension, SkewedChi2Sampler.
 */

export class FastfoodSampler {
  nComponents: number;
  gamma: number;
  randomState: number;
  private _omega: Float64Array[] | null = null;
  private _bias: Float64Array | null = null;
  private _d: Float64Array | null = null;
  private _B: Int32Array | null = null;
  private _nFeaturesIn: number = 0;

  constructor(nComponents = 100, gamma = 1.0, randomState = 42) {
    this.nComponents = nComponents;
    this.gamma = gamma;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this._nFeaturesIn = p;
    let seed = this.randomState;
    const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };
    const randn = () => {
      const u1 = rand(), u2 = rand();
      return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
    };

    this._omega = Array.from({ length: this.nComponents }, () =>
      Float64Array.from({ length: p }, () => Math.sqrt(2 * this.gamma) * randn())
    );
    this._bias = Float64Array.from({ length: this.nComponents }, () => rand() * 2 * Math.PI);
    this._d = Float64Array.from({ length: p }, () => randn());
    this._B = Int32Array.from({ length: p }, () => rand() > 0.5 ? 1 : -1);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._omega || !this._bias) throw new Error("Not fitted");
    const k = this.nComponents;
    return X.map((x) => Float64Array.from({ length: k }, (_, j) => {
      const dotProduct = (this._omega![j] as Float64Array).reduce((s, w, d) => s + w * (x[d] ?? 0), 0);
      return Math.sqrt(2 / k) * Math.cos(dotProduct + (this._bias![j] ?? 0));
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class NystroemExt {
  nComponents: number;
  kernel: "rbf" | "poly" | "linear";
  gamma: number;
  degree: number;
  coef0: number;
  randomState: number;
  private _components: Float64Array[] | null = null;
  private _normalization: Float64Array[] | null = null;
  nFeaturesOut_: number = 0;

  constructor(
    nComponents = 100,
    kernel: "rbf" | "poly" | "linear" = "rbf",
    gamma = 1.0,
    degree = 3,
    coef0 = 1.0,
    randomState = 42,
  ) {
    this.nComponents = nComponents;
    this.kernel = kernel;
    this.gamma = gamma;
    this.degree = degree;
    this.coef0 = coef0;
    this.randomState = randomState;
  }

  private _kernelFunc(a: Float64Array, b: Float64Array): number {
    if (this.kernel === "linear") return a.reduce((s, v, d) => s + v * (b[d] ?? 0), 0);
    if (this.kernel === "poly") return (this.gamma * a.reduce((s, v, d) => s + v * (b[d] ?? 0), 0) + this.coef0) ** this.degree;
    const dist2 = a.reduce((s, v, d) => s + (v - (b[d] ?? 0)) ** 2, 0);
    return Math.exp(-this.gamma * dist2);
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const m = Math.min(this.nComponents, n);

    // Random subset
    const perm = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
    this._components = perm.slice(0, m).map((i) => new Float64Array(X[i] as Float64Array));

    // Compute kernel matrix for components
    const K: Float64Array[] = Array.from({ length: m }, (_, i) =>
      Float64Array.from({ length: m }, (_, j) => this._kernelFunc(this._components![i] as Float64Array, this._components![j] as Float64Array))
    );

    // Eigendecomposition (power iteration for top eigenvectors — simplified)
    // Use diagonal approximation
    this._normalization = K.map((row, i) => {
      const kii = (row as Float64Array)[i] ?? 1;
      const res = new Float64Array(m);
      res[i] = 1 / Math.sqrt(Math.max(kii, 1e-12));
      return res;
    });
    this.nFeaturesOut_ = m;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._components || !this._normalization) throw new Error("Not fitted");
    const m = this._components.length;
    return X.map((x) =>
      Float64Array.from({ length: m }, (_, j) => {
        const k = this._kernelFunc(x, this._components![j] as Float64Array);
        return k * ((this._normalization![j] as Float64Array)[j] ?? 1);
      })
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class SkewedChi2Sampler {
  skewedness: number;
  nComponents: number;
  randomState: number;
  private _randomWeights: Float64Array[] | null = null;
  private _randomOffset: Float64Array | null = null;

  constructor(skewedness = 1.0, nComponents = 100, randomState = 42) {
    this.skewedness = skewedness;
    this.nComponents = nComponents;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    let seed = this.randomState;
    const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };
    const randn = () => {
      const u1 = rand(), u2 = rand();
      return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
    };

    this._randomWeights = Array.from({ length: this.nComponents }, () =>
      Float64Array.from({ length: p }, () => randn())
    );
    this._randomOffset = Float64Array.from({ length: this.nComponents }, () => rand() * 2 * Math.PI);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._randomWeights || !this._randomOffset) throw new Error("Not fitted");
    return X.map((x) => {
      // Skewed chi2 feature: log(x + skewedness)
      const logX = x.map((v) => Math.log(Math.max(v + this.skewedness, 1e-12)));
      return Float64Array.from({ length: this.nComponents }, (_, j) => {
        const dot = (this._randomWeights![j] as Float64Array).reduce((s, w, d) => s + w * (logX[d] ?? 0), 0);
        return Math.sqrt(2 / this.nComponents) * Math.cos(dot + (this._randomOffset![j] ?? 0));
      });
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
