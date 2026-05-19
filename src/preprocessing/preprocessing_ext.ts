/**
 * Additional preprocessing utilities: InteractionFeatures, MaxAbsScalerExt.
 * Mirrors sklearn.preprocessing interaction terms and scaler utilities.
 */

export interface InteractionFeaturesOptions {
  degree?: number;
  interactionOnly?: boolean;
  includeBias?: boolean;
}

/**
 * Generate interaction features between variables.
 * For degree=2, produces x_i * x_j for i <= j (or i < j with interactionOnly).
 */
export class InteractionFeatures {
  degree: number;
  interactionOnly: boolean;
  includeBias: boolean;

  private inputDim_: number = 0;
  private outputDim_: number = 0;
  private indices_: [number, number][] | null = null;

  constructor(options: InteractionFeaturesOptions = {}) {
    this.degree = options.degree ?? 2;
    this.interactionOnly = options.interactionOnly ?? false;
    this.includeBias = options.includeBias ?? false;
  }

  fit(X: Float64Array[]): this {
    const n = X[0]?.length ?? 0;
    this.inputDim_ = n;

    const pairs: [number, number][] = [];
    if (this.includeBias) {
      // Bias term — represented as index pair (-1, -1)
      pairs.push([-1, -1] as unknown as [number, number]);
    }
    for (let i = 0; i < n; i++) {
      if (!this.interactionOnly) pairs.push([i, i]);
      for (let j = i + 1; j < n; j++) {
        pairs.push([i, j]);
      }
    }
    this.indices_ = pairs;
    this.outputDim_ = pairs.length;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.indices_) throw new Error("InteractionFeatures not fitted");
    const pairs = this.indices_;
    return X.map(row => {
      const out = new Float64Array(this.outputDim_);
      for (let k = 0; k < pairs.length; k++) {
        const [i, j] = pairs[k]!;
        if (i === -1) {
          out[k] = 1; // bias
        } else {
          out[k] = (row[i] ?? 0) * (row[j] ?? 0);
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  getOutputDim(): number { return this.outputDim_; }
}

export interface MissingIndicatorExtOptions {
  missingValues?: number;
  features?: "missing-only" | "all";
  sparse?: boolean;
  errorOnNew?: boolean;
}

/**
 * Extended MissingIndicator that adds binary indicator features for missing values.
 */
export class MissingIndicatorExt {
  missingValues: number;
  features: "missing-only" | "all";
  errorOnNew: boolean;

  private indicatorFeatures_: number[] | null = null;
  private nFeatures_: number = 0;

  constructor(options: MissingIndicatorExtOptions = {}) {
    this.missingValues = options.missingValues ?? Number.NaN;
    this.features = options.features ?? "missing-only";
    this.errorOnNew = options.errorOnNew ?? true;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    const hasMissing = new Array(nFeatures).fill(false);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        const v = row[j] ?? 0;
        if (Number.isNaN(v) || v === this.missingValues) hasMissing[j] = true;
      }
    }

    if (this.features === "missing-only") {
      this.indicatorFeatures_ = hasMissing.reduce<number[]>((acc, v, i) => {
        if (v) acc.push(i);
        return acc;
      }, []);
    } else {
      this.indicatorFeatures_ = Array.from({ length: nFeatures }, (_, i) => i);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.indicatorFeatures_) throw new Error("MissingIndicatorExt not fitted");
    const features = this.indicatorFeatures_;
    return X.map(row => new Float64Array(features.map(j => {
      const v = row[j] ?? 0;
      return (Number.isNaN(v) || v === this.missingValues) ? 1 : 0;
    })));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Binarizer with extended threshold options.
 */
export class ThresholdBinarizer {
  threshold: number;
  copyData: boolean;

  constructor(options: { threshold?: number; copyData?: boolean } = {}) {
    this.threshold = options.threshold ?? 0.0;
    this.copyData = options.copyData ?? true;
  }

  fit(_X: Float64Array[]): this { return this; }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => new Float64Array(row.map(v => v > this.threshold ? 1 : 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }
}

/**
 * AdditiveChi2Sampler — approximates the additive chi2 kernel via feature map.
 * Mirrors sklearn.kernel_approximation.AdditiveChi2Sampler.
 */
export class AdditiveChi2SamplerExt {
  sampleSteps: number;
  sampleInterval: number;

  private nComponents_: number = 0;

  constructor(options: { sampleSteps?: number; sampleInterval?: number } = {}) {
    this.sampleSteps = options.sampleSteps ?? 2;
    this.sampleInterval = options.sampleInterval ?? 0.4;
  }

  fit(X: Float64Array[]): this {
    this.nComponents_ = (X[0]?.length ?? 0) * (2 * this.sampleSteps + 1);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const nFeatures = X[0]?.length ?? 0;
    const nOut = nFeatures * (2 * this.sampleSteps + 1);
    return X.map(row => {
      const out = new Float64Array(nOut);
      let offset = 0;
      for (let j = 0; j < nFeatures; j++) {
        const xj = Math.max(row[j] ?? 0, 1e-10);
        const sqrtX = Math.sqrt(xj);
        out[offset++] = sqrtX;
        for (let s = 1; s <= this.sampleSteps; s++) {
          const factor = Math.sqrt(2 * this.sampleInterval);
          const angle = s * this.sampleInterval * Math.log(xj);
          out[offset++] = factor * Math.cos(angle) * sqrtX;
          out[offset++] = factor * Math.sin(angle) * sqrtX;
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  get nComponentsOut(): number { return this.nComponents_; }
}
