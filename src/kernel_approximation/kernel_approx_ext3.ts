/**
 * Extended kernel approximation methods: Tensor Sketch, Fastfood, SORF.
 * Port of sklearn.kernel_approximation extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Polynomial kernel approximation using Tensor Sketch.
 */
export class TensorSketchApproximation {
  private nComponents: number;
  private degree: number;
  private hashFuncs_: Array<{ indices: Int32Array; signs: Int32Array }> = [];
  private fitted = false;

  constructor(options: { nComponents?: number; degree?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
    this.degree = options.degree ?? 2;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.hashFuncs_ = Array.from({ length: this.degree }, () => ({
      indices: Int32Array.from({ length: nFeatures }, () => Math.floor(Math.random() * this.nComponents)),
      signs: Int32Array.from({ length: nFeatures }, () => Math.random() < 0.5 ? 1 : -1),
    }));
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("TensorSketchApproximation not fitted");
    return X.map(row => {
      let sketch = new Float64Array(this.nComponents);
      for (let j = 0; j < row.length; j++) {
        const { indices, signs } = this.hashFuncs_[0]!;
        sketch[indices[j] ?? 0] = (sketch[indices[j] ?? 0] ?? 0) + (row[j] ?? 0) * (signs[j] ?? 1);
      }

      for (let d = 1; d < this.degree; d++) {
        const nextSketch = new Float64Array(this.nComponents);
        for (let j = 0; j < row.length; j++) {
          const { indices, signs } = this.hashFuncs_[d]!;
          nextSketch[indices[j] ?? 0] = (nextSketch[indices[j] ?? 0] ?? 0) + (row[j] ?? 0) * (signs[j] ?? 1);
        }
        for (let k = 0; k < this.nComponents; k++) {
          sketch[k] = (sketch[k] ?? 0) * (nextSketch[k] ?? 0);
        }
      }

      return sketch;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Matern kernel approximation via random features.
 */
export class MaternRandomFeatures {
  private nComponents: number;
  private nu: number;
  private gamma: number;
  private omega_: Float64Array[] = [];
  private bias_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; nu?: number; gamma?: number } = {}) {
    this.nComponents = options.nComponents ?? 100;
    this.nu = options.nu ?? 1.5;
    this.gamma = options.gamma ?? 1.0;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    // For Matern kernel, sample from Student-t distribution with df = 2*nu
    const df = 2 * this.nu;
    this.omega_ = Array.from({ length: this.nComponents }, () => {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        // Normal / sqrt(chi2/df) approximation
        const u1 = Math.random(); const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        // Chi2 approximation
        let chi2 = 0;
        for (let k = 0; k < Math.round(df); k++) {
          const u3 = Math.random(); const u4 = Math.random();
          const n2 = Math.sqrt(-2 * Math.log(u3 + 1e-10)) * Math.cos(2 * Math.PI * u4);
          chi2 += n2 * n2;
        }
        row[j] = normal / Math.sqrt(chi2 / df) * Math.sqrt(2 * this.gamma);
      }
      return row;
    });
    this.bias_ = Float64Array.from({ length: this.nComponents }, () => Math.random() * 2 * Math.PI);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("MaternRandomFeatures not fitted");
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
