/**
 * Random projection extensions: Count Sketch, SRHT, tensor sketch.
 * Mirrors sklearn.random_projection additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Subsampled Randomized Hadamard Transform (SRHT) for fast random projections. */
export class SRHTProjection extends BaseEstimator {
  n_components: number;
  random_state: number | null;
  components_: Float64Array[] = [];
  sampling_: Int32Array = new Int32Array(0);

  constructor(params: { n_components?: number; random_state?: number | null } = {}) {
    super();
    this.n_components = params.n_components ?? 10;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const k = this.n_components;
    // Random sign flip
    const signs = new Float64Array(d).map(() => Math.random() < 0.5 ? 1 : -1);
    // Random sampling indices
    this.sampling_ = new Int32Array(k).map(() => Math.floor(Math.random() * d));
    this.components_ = [signs];
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const d = X[0]?.length ?? 0;
    const k = this.n_components;
    const signs = this.components_[0] ?? new Float64Array(d).fill(1);
    return X.map(row => {
      // Apply Hadamard transform approximation
      const temp = new Float64Array(d).map((_, i) => (row[i] ?? 0) * (signs[i] ?? 1));
      // Apply fast Walsh-Hadamard transform
      const wht = fwht(temp);
      return new Float64Array(k).map((_, i) => (wht[this.sampling_[i] ?? 0] ?? 0) / Math.sqrt(k));
    });
  }
}

function fwht(x: Float64Array): Float64Array {
  const n = x.length;
  const result = new Float64Array(x);
  let h = 1;
  while (h < n) {
    for (let i = 0; i < n; i += h * 2) {
      for (let j = i; j < i + h; j++) {
        const u = result[j] ?? 0;
        const v = result[j + h] ?? 0;
        result[j] = u + v;
        result[j + h] = u - v;
      }
    }
    h *= 2;
  }
  return result;
}

/** Random kitchen sinks (random features for RBF kernel approximation). */
export class RandomKitchenSinks extends BaseEstimator {
  n_components: number;
  gamma: number;
  omegas_: Float64Array[] = [];
  biases_: Float64Array = new Float64Array(0);

  constructor(params: { n_components?: number; gamma?: number } = {}) {
    super();
    this.n_components = params.n_components ?? 100;
    this.gamma = params.gamma ?? 1.0;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const k = this.n_components;
    // Sample frequencies from Fourier transform of RBF kernel
    this.omegas_ = Array.from({ length: k }, () => {
      const omega = new Float64Array(d);
      let norm = 0;
      for (let f = 0; f < d; f++) {
        // Box-Muller transform for normal samples
        const u1 = Math.random();
        const u2 = Math.random();
        omega[f] = Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2) * Math.sqrt(2 * this.gamma);
        norm += omega[f]! ** 2;
      }
      return omega;
    });
    this.biases_ = new Float64Array(k).map(() => Math.random() * 2 * Math.PI);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const k = this.n_components;
    const scale = Math.sqrt(2 / k);
    return X.map(row => new Float64Array(k).map((_, i) => {
      let dot = 0;
      for (let f = 0; f < row.length; f++) dot += (row[f] ?? 0) * (this.omegas_[i]?.[f] ?? 0);
      return scale * Math.cos(dot + (this.biases_[i] ?? 0));
    }));
  }
}

/** Ternary sparse random projection (entries in {-1/sqrt(s), 0, 1/sqrt(s)}). */
export class TernarySparseProjection extends BaseEstimator {
  n_components: number;
  density: number;
  components_: Float64Array[] = [];

  constructor(params: { n_components?: number; density?: number } = {}) {
    super();
    this.n_components = params.n_components ?? 100;
    this.density = params.density ?? 1 / 3;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const k = this.n_components;
    const scale = Math.sqrt(1 / (d * this.density));
    this.components_ = Array.from({ length: k }, () => {
      const row = new Float64Array(d);
      for (let f = 0; f < d; f++) {
        const u = Math.random();
        if (u < this.density / 2) row[f] = scale;
        else if (u < this.density) row[f] = -scale;
        // else row[f] = 0 (already 0)
      }
      return row;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const k = this.n_components;
    return X.map(row => new Float64Array(k).map((_, i) => {
      let s = 0;
      const comp = this.components_[i]!;
      for (let f = 0; f < row.length; f++) s += (row[f] ?? 0) * (comp[f] ?? 0);
      return s;
    }));
  }
}

/** Johnson-Lindenstrauss lemma: estimate minimum n_components for target eps. */
export function johnsonLindenstraussMinComponents(
  n_samples: number,
  eps: number,
): number {
  return Math.ceil(4 * Math.log(n_samples) / (eps ** 2 / 2 - eps ** 3 / 3));
}
