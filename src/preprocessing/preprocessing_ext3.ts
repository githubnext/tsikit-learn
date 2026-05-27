/**
 * Extended preprocessing: AdditiveChi2Sampler-style feature maps,
 * interaction features, and additional data transformations.
 */

/** Nystroem approximation kernel feature map (simplified). */
export class NystroemApprox {
  private components_?: Float64Array[];
  private normalization_?: Float64Array[];
  nComponents: number;
  gamma: number;

  constructor(nComponents = 100, gamma = 1.0) {
    this.nComponents = nComponents;
    this.gamma = gamma;
  }

  fit(X: Float64Array[]): this {
    // Sample nComponents training points as components
    const step = Math.max(1, Math.floor(X.length / this.nComponents));
    this.components_ = [];
    for (let i = 0; i < X.length && this.components_.length < this.nComponents; i += step) {
      const xi = X[i];
      if (xi !== undefined) this.components_.push(new Float64Array(xi));
    }
    this.normalization_ = this.components_.map(() => new Float64Array([1.0]));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    return X.map((xi) => {
      const feats = new Float64Array(this.components_!.length);
      for (let k = 0; k < this.components_!.length; k++) {
        const comp = this.components_![k];
        if (comp === undefined) continue;
        let dist2 = 0;
        for (let j = 0; j < xi.length; j++) dist2 += ((xi[j] ?? 0) - (comp[j] ?? 0)) ** 2;
        feats[k] = Math.exp(-this.gamma * dist2);
      }
      return feats;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/** Additive Chi2 Sampler — feature map for chi2 kernel. */
export class AdditiveChi2Sampler {
  private sampleSteps: number;
  private sampleInterval: number;
  nFeaturesIn_?: number;
  nFeaturesOut_?: number;

  constructor(sampleSteps = 2, sampleInterval = 1.0) {
    this.sampleSteps = sampleSteps;
    this.sampleInterval = sampleInterval;
  }

  fit(X: Float64Array[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    this.nFeaturesOut_ = this.nFeaturesIn_ * (2 * this.sampleSteps + 1);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const d = this.nFeaturesIn_ ?? (X[0]?.length ?? 0);
    return X.map((xi) => {
      const out: number[] = [];
      for (let j = 0; j < d; j++) {
        const xij = Math.max(xi[j] ?? 0, 0);
        out.push(Math.sqrt(xij * this.sampleInterval));
        for (let s = 1; s <= this.sampleSteps; s++) {
          const cos = Math.sqrt(2 * xij * this.sampleInterval) * Math.cos(Math.PI * s * this.sampleInterval * xij);
          const sin = Math.sqrt(2 * xij * this.sampleInterval) * Math.sin(Math.PI * s * this.sampleInterval * xij);
          out.push(cos, sin);
        }
      }
      return new Float64Array(out);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/** Skewed Chi2 Sampler — feature map for skewed chi2 kernel. */
export class SkewedChi2Sampler {
  private nComponents: number;
  private skewness: number;
  private randomWeights_?: Float64Array[];
  private randomOffset_?: Float64Array;

  constructor(skewness = 1.0, nComponents = 100) {
    this.skewness = skewness;
    this.nComponents = nComponents;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    // Random Fourier features weights
    this.randomWeights_ = Array.from({ length: d }, () => {
      const w = new Float64Array(this.nComponents);
      for (let i = 0; i < this.nComponents; i++) w[i] = (Math.random() - 0.5) * 2 * Math.sqrt(this.skewness);
      return w;
    });
    this.randomOffset_ = new Float64Array(this.nComponents).map(() => Math.random() * 2 * Math.PI);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const d = this.randomWeights_?.length ?? 0;
    return X.map((xi) => {
      const projection = new Float64Array(this.nComponents);
      for (let j = 0; j < d; j++) {
        const w = this.randomWeights_?.[j];
        const xij = Math.log(Math.max(xi[j] ?? 1e-10, 1e-10) + this.skewness);
        if (w === undefined) continue;
        for (let k = 0; k < this.nComponents; k++) {
          projection[k] = (projection[k] ?? 0) + (w[k] ?? 0) * xij;
        }
      }
      const out = new Float64Array(this.nComponents);
      for (let k = 0; k < this.nComponents; k++) {
        out[k] = Math.sqrt(2 / this.nComponents) * Math.cos((projection[k] ?? 0) + (this.randomOffset_?.[k] ?? 0));
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/** Max absolute scaler (scale each feature by its max absolute value). */
export class MaxAbsScaler {
  scale_?: Float64Array;
  maxAbs_?: Float64Array;
  nFeaturesIn_?: number;

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    this.nFeaturesIn_ = d;
    this.maxAbs_ = new Float64Array(d);
    for (const xi of X) {
      for (let j = 0; j < d; j++) {
        const abs = Math.abs(xi[j] ?? 0);
        if (abs > (this.maxAbs_[j] ?? 0)) this.maxAbs_[j] = abs;
      }
    }
    this.scale_ = this.maxAbs_.map((v) => (v === 0 ? 1 : v));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.scale_) throw new Error("Not fitted");
    return X.map((xi) => xi.map((v, j) => v / (this.scale_?.[j] ?? 1)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.scale_) throw new Error("Not fitted");
    return X.map((xi) => xi.map((v, j) => v * (this.scale_?.[j] ?? 1)));
  }
}
