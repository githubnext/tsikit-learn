/**
 * Random projection dimensionality reduction.
 * Mirrors sklearn.random_projection: GaussianRandomProjection,
 * SparseRandomProjection, johnson_lindenstrauss_min_dim.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Compute the minimum number of components needed to guarantee the
 * Johnson-Lindenstrauss lemma.
 * Mirrors sklearn.random_projection.johnson_lindenstrauss_min_dim.
 */
export function johnsonLindenstraussMinDim(
  nSamples: number,
  eps: number = 0.1,
): number {
  if (eps <= 0 || eps >= 1) throw new RangeError("eps must be in (0, 1)");
  const denominator = (eps ** 2 / 2) - (eps ** 3 / 3);
  return Math.ceil((4 * Math.log(nSamples)) / denominator);
}

/** Sample Gaussian random matrix. */
function gaussianMatrix(
  nComponents: number,
  nFeatures: number,
): Float64Array[] {
  const std = 1 / Math.sqrt(nComponents);
  return Array.from({ length: nFeatures }, () => {
    const row = new Float64Array(nComponents);
    for (let j = 0; j < nComponents; j++) {
      // Box-Muller transform
      const u1 = Math.random() + 1e-10;
      const u2 = Math.random();
      row[j] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return row;
  });
}

/** Sample sparse random matrix (Li et al. 2006). density = 1/sqrt(nFeatures). */
function sparseMatrix(
  nComponents: number,
  nFeatures: number,
  density: number,
): Float64Array[] {
  const s = 1 / density;
  const scale = Math.sqrt(s / nComponents);
  return Array.from({ length: nFeatures }, () => {
    const row = new Float64Array(nComponents);
    for (let j = 0; j < nComponents; j++) {
      const r = Math.random();
      if (r < density / 2) {
        row[j] = scale;
      } else if (r < density) {
        row[j] = -scale;
      }
      // else 0 (sparse)
    }
    return row;
  });
}

function project(X: Float64Array[], components: Float64Array[]): Float64Array[] {
  // X: n x nFeatures, components: nFeatures x nComponents -> n x nComponents
  const nComponents = (components[0] ?? new Float64Array(0)).length;
  return X.map(xi => {
    const out = new Float64Array(nComponents);
    for (let f = 0; f < xi.length; f++) {
      const comp = components[f] ?? new Float64Array(nComponents);
      for (let j = 0; j < nComponents; j++) {
        out[j]! += (xi[f] ?? 0) * (comp[j] ?? 0);
      }
    }
    return out;
  });
}

export interface GaussianRandomProjectionOptions {
  nComponents?: number | "auto";
  eps?: number;
  randomState?: number;
}

/**
 * Reduce dimensionality using Gaussian random projection.
 * Mirrors sklearn.random_projection.GaussianRandomProjection.
 */
export class GaussianRandomProjection {
  nComponents: number | "auto";
  eps: number;

  nComponents_: number | null = null;
  components_: Float64Array[] | null = null;

  constructor(options: GaussianRandomProjectionOptions = {}) {
    this.nComponents = options.nComponents ?? "auto";
    this.eps = options.eps ?? 0.1;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k =
      this.nComponents === "auto"
        ? Math.min(johnsonLindenstraussMinDim(n, this.eps), p)
        : this.nComponents;
    this.nComponents_ = k;
    this.components_ = gaussianMatrix(k, p);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || this.nComponents_ === null)
      throw new NotFittedError("GaussianRandomProjection");
    return project(X, this.components_);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface SparseRandomProjectionOptions {
  nComponents?: number | "auto";
  density?: number | "auto";
  eps?: number;
  randomState?: number;
}

/**
 * Reduce dimensionality using sparse random projection.
 * Mirrors sklearn.random_projection.SparseRandomProjection.
 */
export class SparseRandomProjection {
  nComponents: number | "auto";
  density: number | "auto";
  eps: number;

  nComponents_: number | null = null;
  components_: Float64Array[] | null = null;
  density_: number | null = null;

  constructor(options: SparseRandomProjectionOptions = {}) {
    this.nComponents = options.nComponents ?? "auto";
    this.density = options.density ?? "auto";
    this.eps = options.eps ?? 0.1;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k =
      this.nComponents === "auto"
        ? Math.min(johnsonLindenstraussMinDim(n, this.eps), p)
        : this.nComponents;
    const d = this.density === "auto" ? 1 / Math.sqrt(p) : this.density;
    this.nComponents_ = k;
    this.density_ = d;
    this.components_ = sparseMatrix(k, p, d);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || this.nComponents_ === null)
      throw new NotFittedError("SparseRandomProjection");
    return project(X, this.components_);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
