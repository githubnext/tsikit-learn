/**
 * Extended random projection methods.
 * Port of sklearn.random_projection extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Very sparse random projection using Li et al. (2006) construction.
 */
export class VerySparsePureRandomProjection {
  private nComponents: number;
  private sparsity: number;
  private components_: Float64Array[] = [];
  private fitted = false;

  constructor(options: { nComponents?: number; sparsity?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
    this.sparsity = options.sparsity ?? 1 / 3;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    const scale = Math.sqrt(1 / (this.sparsity * this.nComponents));

    this.components_ = Array.from({ length: this.nComponents }, () => {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const r = Math.random();
        if (r < this.sparsity / 2) row[j] = -scale;
        else if (r < this.sparsity) row[j] = scale;
        // else 0 (very sparse)
      }
      return row;
    });

    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("VerySparsePureRandomProjection not fitted");
    return X.map(row => Float64Array.from(this.components_, comp => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) * (comp[j] ?? 0);
      return s;
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Johnson-Lindenstrauss lemma: estimate minimum number of components.
 */
export function johnsonLindenstraussMinDim(
  nSamples: number,
  eps: number,
): number {
  if (eps <= 0 || eps >= 1) throw new Error("eps must be in (0, 1)");
  const denominator = eps ** 2 / 2 - eps ** 3 / 3;
  return Math.ceil(4 * Math.log(nSamples) / denominator);
}

/**
 * Count sketch random projection (for feature hashing).
 */
export class CountSketchProjection {
  private nComponents: number;
  private hashIndices_: Int32Array = new Int32Array(0);
  private hashSigns_: Int32Array = new Int32Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.hashIndices_ = Int32Array.from({ length: nFeatures }, () => Math.floor(Math.random() * this.nComponents));
    this.hashSigns_ = Int32Array.from({ length: nFeatures }, () => Math.random() < 0.5 ? 1 : -1);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("CountSketchProjection not fitted");
    return X.map(row => {
      const out = new Float64Array(this.nComponents);
      for (let j = 0; j < row.length; j++) {
        const idx = this.hashIndices_[j] ?? 0;
        out[idx] = (out[idx] ?? 0) + (row[j] ?? 0) * (this.hashSigns_[j] ?? 1);
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Random Fourier Features (approximates RBF kernel via random projections).
 */
export class RandomFourierFeatures {
  private nComponents: number;
  private gamma: number;
  private omega_: Float64Array[] = [];
  private bias_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; gamma?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
    this.gamma = options.gamma ?? 1.0;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    const scale = Math.sqrt(2 * this.gamma);

    // Sample from N(0, 2*gamma)
    this.omega_ = Array.from({ length: this.nComponents }, () => {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        // Box-Muller transform
        const u1 = Math.random(); const u2 = Math.random();
        row[j] = scale * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
      }
      return row;
    });

    this.bias_ = Float64Array.from({ length: this.nComponents }, () => Math.random() * 2 * Math.PI);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("RandomFourierFeatures not fitted");
    const scale = Math.sqrt(2 / this.nComponents);
    return X.map(row => Float64Array.from({ length: this.nComponents }, (_, k) => {
      let dot = 0;
      for (let j = 0; j < row.length; j++) dot += (row[j] ?? 0) * (this.omega_[k]?.[j] ?? 0);
      return scale * Math.cos(dot + (this.bias_[k] ?? 0));
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Circulant binary embedding (faster than standard random projections).
 */
export class CirculantBinaryEmbedding {
  private nComponents: number;
  private randomVector_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    // Generate a random vector for the circulant matrix
    this.randomVector_ = Float64Array.from({ length: nFeatures }, () => (Math.random() - 0.5) * 2);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("CirculantBinaryEmbedding not fitted");
    const d = this.randomVector_.length;
    return X.map(row => {
      const out = new Float64Array(Math.min(this.nComponents, d));
      for (let k = 0; k < out.length; k++) {
        let s = 0;
        for (let j = 0; j < d; j++) {
          s += (row[j] ?? 0) * (this.randomVector_[(j + k) % d] ?? 0);
        }
        out[k] = s > 0 ? 1 : -1;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}
