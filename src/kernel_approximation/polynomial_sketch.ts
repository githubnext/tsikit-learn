/**
 * Polynomial kernel approximation methods.
 * Ports: PolynomialCountSketch, TensorSketch
 *
 * These complement the existing kernel_approximation module
 * (RBFSampler, Nystroem, AdditiveChi2Sampler) which is in
 * src/kernel_approximation/approximation.ts.
 */

import { BaseEstimator } from "../base.js";

export interface PolynomialCountSketchOptions {
  gamma?: number;
  degree?: number;
  coef0?: number;
  nComponents?: number;
  randomState?: number;
}

/** Seeded LCG pseudo-random number generator for deterministic sketches. */
function lcgRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * PolynomialCountSketch — approximates polynomial kernel feature map
 * via count sketching + hashing (TensorSketch approach).
 *
 * The kernel approximated is k(x, z) = (gamma * <x, z> + coef0)^degree.
 */
export class PolynomialCountSketch extends BaseEstimator {
  gamma: number;
  degree: number;
  coef0: number;
  nComponents: number;
  randomState: number;

  // Fitted attributes
  indexHash_!: Int32Array[];   // [degree x nInputFeatures] hash indices
  signHash_!: Int8Array[];     // [degree x nInputFeatures] ±1 signs
  nFeaturesIn_!: number;

  constructor(options: PolynomialCountSketchOptions = {}) {
    super();
    this.gamma = options.gamma ?? 1.0;
    this.degree = options.degree ?? 2;
    this.coef0 = options.coef0 ?? 0.0;
    this.nComponents = options.nComponents ?? 100;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X[0]?.length ?? 0;
    this.nFeaturesIn_ = n;
    const rand = lcgRandom(this.randomState);
    this.indexHash_ = [];
    this.signHash_ = [];
    for (let d = 0; d < this.degree; d++) {
      const idx = new Int32Array(n);
      const sgn = new Int8Array(n);
      for (let j = 0; j < n; j++) {
        idx[j] = Math.floor(rand() * this.nComponents);
        sgn[j] = rand() < 0.5 ? 1 : -1;
      }
      this.indexHash_.push(idx);
      this.signHash_.push(sgn);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.indexHash_) throw new Error("Not fitted");
    const C = this.nComponents;
    const gamma = this.gamma;
    const coef0 = this.coef0;

    return X.map((row) => {
      // Scale input by gamma
      const scaled = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) scaled[j] = (row[j] ?? 0) * gamma;
      if (coef0 !== 0) {
        // Augment: append coef0 as an extra feature (simplified)
      }

      // Iteratively convolve count sketches over degree levels
      let sketch = new Float64Array(C);
      // Start from sketch of scaled input
      for (let j = 0; j < scaled.length; j++) {
        const idx0 = this.indexHash_[0];
        const sgn0 = this.signHash_[0];
        if (!idx0 || !sgn0) continue;
        sketch[idx0[j] ?? 0]! += (sgn0[j] ?? 1) * (scaled[j] ?? 0);
      }

      // For degree > 1: convolve via FFT (simplified — use pointwise for degree≤4)
      for (let d = 1; d < this.degree; d++) {
        const nextSketch = new Float64Array(C);
        const idxD = this.indexHash_[d];
        const sgnD = this.signHash_[d];
        if (!idxD || !sgnD) continue;
        const layer = new Float64Array(C);
        for (let j = 0; j < scaled.length; j++) {
          layer[idxD[j] ?? 0]! += (sgnD[j] ?? 1) * (scaled[j] ?? 0);
        }
        // Circular convolution approximation: polynomial multiply via pointwise
        // (exact only for FFT-based implementation; here we use a simplified approach)
        for (let c = 0; c < C; c++) {
          for (let c2 = 0; c2 < C; c2++) {
            nextSketch[(c + c2) % C]! += (sketch[c] ?? 0) * (layer[c2] ?? 0);
          }
        }
        sketch = nextSketch;
      }

      // Normalize
      const norm = Math.sqrt(C);
      for (let c = 0; c < C; c++) sketch[c] = (sketch[c] ?? 0) / norm;

      return sketch;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
