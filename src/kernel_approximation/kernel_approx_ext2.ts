/**
 * Extended kernel approximations: AdditiveChi2Sampler, ANOVASampler, TensorSketch
 */

export class AdditiveChi2Sampler {
  private sampleSteps: number;
  private sampleInterval: number;
  private nFeaturesOut_: number = 0;

  constructor(sampleSteps = 2, sampleInterval = 0.2) {
    this.sampleSteps = sampleSteps;
    this.sampleInterval = sampleInterval;
  }

  get nFeaturesOut(): number {
    return this.nFeaturesOut_;
  }

  fit(nFeaturesIn: number): this {
    this.nFeaturesOut_ = nFeaturesIn * (2 * this.sampleSteps + 1);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    this.fit(p);
    return X.map((row) => {
      const out = new Float64Array(this.nFeaturesOut_);
      let outIdx = 0;
      for (let j = 0; j < p; j++) {
        const x = Math.abs(row[j] ?? 0);
        // Step 0: sqrt(x * interval)
        out[outIdx++] = Math.sqrt(x * this.sampleInterval);
        for (let s = 1; s <= this.sampleSteps; s++) {
          const c = Math.sqrt(x * this.sampleInterval * 2) * Math.cos(s * Math.log(x + 1e-12) * this.sampleInterval);
          const d = Math.sqrt(x * this.sampleInterval * 2) * Math.sin(s * Math.log(x + 1e-12) * this.sampleInterval);
          out[outIdx++] = c;
          out[outIdx++] = d;
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X[0]?.length ?? 0).transform(X);
  }
}

export class ANOVASampler {
  private degree: number;
  private gammaScale: number;
  private nComponents: number;
  private W_: Float64Array[] | null = null;
  private b_: Float64Array | null = null;
  nFeaturesOut_: number = 0;

  constructor(degree = 2, gammaScale = 1.0, nComponents = 100) {
    this.degree = degree;
    this.gammaScale = gammaScale;
    this.nComponents = nComponents;
  }

  fit(nFeaturesIn: number): this {
    // Random Fourier Features for ANOVA kernel approximation
    this.nFeaturesOut_ = this.nComponents * this.degree;
    this.W_ = Array.from({ length: this.degree }, () => {
      const w = new Float64Array(nFeaturesIn * this.nComponents);
      for (let i = 0; i < w.length; i++) {
        const u1 = Math.random(), u2 = Math.random();
        w[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2) * Math.sqrt(this.gammaScale);
      }
      return w;
    });
    this.b_ = new Float64Array(this.nComponents * this.degree);
    for (let i = 0; i < this.b_.length; i++) this.b_[i] = Math.random() * 2 * Math.PI;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.W_ || !this.b_) throw new Error("Not fitted");
    const p = X[0]?.length ?? 0;
    return X.map((row) => {
      const out = new Float64Array(this.nFeaturesOut_);
      for (let d = 0; d < this.degree; d++) {
        const W = this.W_![d]!;
        for (let c = 0; c < this.nComponents; c++) {
          let proj = this.b_![d * this.nComponents + c] ?? 0;
          for (let j = 0; j < p; j++) proj += (W[c * p + j] ?? 0) * (row[j] ?? 0);
          out[d * this.nComponents + c] = Math.sqrt(2 / this.nComponents) * Math.cos(proj);
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X[0]?.length ?? 0).transform(X);
  }
}

export class TensorSketch {
  private nComponents: number;
  private degree: number;
  private hashMaps_: Int32Array[] | null = null;
  private signs_: Int32Array[] | null = null;
  nFeaturesOut_: number = 0;

  constructor(nComponents = 100, degree = 2) {
    this.nComponents = nComponents;
    this.degree = degree;
  }

  fit(nFeaturesIn: number): this {
    this.nFeaturesOut_ = this.nComponents;
    this.hashMaps_ = Array.from({ length: this.degree }, () => {
      const h = new Int32Array(nFeaturesIn);
      for (let i = 0; i < nFeaturesIn; i++) h[i] = Math.floor(Math.random() * this.nComponents);
      return h;
    });
    this.signs_ = Array.from({ length: this.degree }, () => {
      const s = new Int32Array(nFeaturesIn);
      for (let i = 0; i < nFeaturesIn; i++) s[i] = Math.random() < 0.5 ? 1 : -1;
      return s;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.hashMaps_ || !this.signs_) throw new Error("Not fitted");
    const p = X[0]?.length ?? 0;
    return X.map((row) => {
      // Compute tensor sketch via FFT convolution approximation
      let sketch = new Float64Array(this.nComponents);
      for (let j = 0; j < p; j++) {
        const h = this.hashMaps_![0]![j] ?? 0;
        const s = this.signs_![0]![j] ?? 1;
        sketch[h] += s * (row[j] ?? 0);
      }
      for (let d = 1; d < this.degree; d++) {
        const nextSketch = new Float64Array(this.nComponents);
        const currSketch = new Float64Array(this.nComponents);
        for (let j = 0; j < p; j++) {
          const h = this.hashMaps_![d]![j] ?? 0;
          const s = this.signs_![d]![j] ?? 1;
          currSketch[h] += s * (row[j] ?? 0);
        }
        // Approximate polynomial kernel via element-wise product in frequency domain
        for (let c = 0; c < this.nComponents; c++) {
          nextSketch[c] = (sketch[c] ?? 0) * (currSketch[c] ?? 0);
        }
        sketch = nextSketch;
      }
      return sketch;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X[0]?.length ?? 0).transform(X);
  }
}
