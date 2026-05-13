/**
 * PCA (Principal Component Analysis) and TruncatedSVD.
 * Mirrors sklearn.decomposition.PCA and TruncatedSVD.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute mean of each column. */
function colMeans(X: Float64Array[], p: number): Float64Array {
  const means = new Float64Array(p);
  for (const xi of X) {
    for (let j = 0; j < p; j++) {
      means[j] = (means[j] ?? 0) + (xi[j] ?? 0);
    }
  }
  for (let j = 0; j < p; j++) {
    means[j] = (means[j] ?? 0) / X.length;
  }
  return means;
}

/** Power iteration to find top-k eigenvectors (randomized SVD). */
function randomizedSVD(
  X: Float64Array[],
  nComponents: number,
  nIter = 5,
): { components: Float64Array[]; explainedVariance: Float64Array } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const k = Math.min(nComponents, n, p);

  // Build components via power iteration
  const components: Float64Array[] = [];
  const explainedVariance = new Float64Array(k);

  // Make a copy to deflate
  const Xwork: Float64Array[] = X.map((xi) => new Float64Array(xi));

  for (let c = 0; c < k; c++) {
    // Random init
    let v = new Float64Array(p);
    for (let j = 0; j < p; j++) v[j] = Math.random() - 0.5;

    // Normalize
    let norm = Math.sqrt(v.reduce((s, x) => s + x ** 2, 0));
    if (norm > 0) {
      for (let j = 0; j < p; j++) v[j] = (v[j] ?? 0) / norm;
    }

    for (let iter = 0; iter < nIter * 10; iter++) {
      // v = X^T X v
      const u = new Float64Array(p);
      // First compute Xv
      const Xv = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        const xi = Xwork[i] ?? new Float64Array(p);
        for (let j = 0; j < p; j++) {
          dot += (xi[j] ?? 0) * (v[j] ?? 0);
        }
        Xv[i] = dot;
      }
      // Then X^T (Xv)
      for (let i = 0; i < n; i++) {
        const xi = Xwork[i] ?? new Float64Array(p);
        for (let j = 0; j < p; j++) {
          u[j] = (u[j] ?? 0) + (Xv[i] ?? 0) * (xi[j] ?? 0);
        }
      }
      norm = Math.sqrt(u.reduce((s, x) => s + x ** 2, 0));
      if (norm === 0) break;
      for (let j = 0; j < p; j++) u[j] = (u[j] ?? 0) / norm;

      let diff = 0;
      for (let j = 0; j < p; j++) diff += (u[j] ?? 0 - (v[j] ?? 0)) ** 2;
      v = u;
      if (diff < 1e-10) break;
    }

    components.push(v);

    // Compute eigenvalue (variance along this component)
    let variance = 0;
    for (let i = 0; i < n; i++) {
      let dot = 0;
      const xi = Xwork[i] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        dot += (xi[j] ?? 0) * (v[j] ?? 0);
      }
      variance += dot ** 2;
    }
    explainedVariance[c] = variance / n;

    // Deflate X
    for (let i = 0; i < n; i++) {
      const xi = Xwork[i] ?? new Float64Array(p);
      let dot = 0;
      for (let j = 0; j < p; j++) dot += (xi[j] ?? 0) * (v[j] ?? 0);
      for (let j = 0; j < p; j++) {
        xi[j] = (xi[j] ?? 0) - dot * (v[j] ?? 0);
      }
    }
  }

  return { components, explainedVariance };
}

export class PCA {
  nComponents: number;
  whiten: boolean;

  components_: Float64Array[] | null = null;
  explainedVariance_: Float64Array | null = null;
  explainedVarianceRatio_: Float64Array | null = null;
  mean_: Float64Array | null = null;

  constructor(
    options: { nComponents?: number; whiten?: boolean } = {},
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.whiten = options.whiten ?? false;
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    this.mean_ = colMeans(X, p);
    const centered = X.map((xi) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        row[j] = (xi[j] ?? 0) - ((this.mean_ as Float64Array)[j] ?? 0);
      }
      return row;
    });

    const { components, explainedVariance } = randomizedSVD(centered, this.nComponents);
    this.components_ = components;
    this.explainedVariance_ = explainedVariance;
    const totalVar = Array.from(explainedVariance).reduce((a, b) => a + b, 0);
    this.explainedVarianceRatio_ = new Float64Array(
      explainedVariance.map((v) => (totalVar > 0 ? v / totalVar : 0)),
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.mean_ === null)
      throw new NotFittedError("PCA");

    const p = (X[0] ?? new Float64Array(0)).length;
    const k = this.components_.length;

    return X.map((xi) => {
      const result = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        const comp = (this.components_ as Float64Array[])[c] ?? new Float64Array(p);
        let dot = 0;
        for (let j = 0; j < p; j++) {
          dot += ((xi[j] ?? 0) - ((this.mean_ as Float64Array)[j] ?? 0)) * (comp[j] ?? 0);
        }
        if (this.whiten) {
          const ev = ((this.explainedVariance_ as Float64Array)[c] ?? 1);
          result[c] = ev > 0 ? dot / Math.sqrt(ev) : dot;
        } else {
          result[c] = dot;
        }
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.mean_ === null)
      throw new NotFittedError("PCA");
    const k = (X[0] ?? new Float64Array(0)).length;
    const p = (this.components_[0] ?? new Float64Array(0)).length;
    return X.map((xi) => {
      const result = new Float64Array(p);
      for (let c = 0; c < k; c++) {
        const comp = (this.components_ as Float64Array[])[c] ?? new Float64Array(p);
        const scale = this.whiten
          ? (xi[c] ?? 0) * Math.sqrt((this.explainedVariance_ as Float64Array)[c] ?? 1)
          : (xi[c] ?? 0);
        for (let j = 0; j < p; j++) {
          result[j] = (result[j] ?? 0) + scale * (comp[j] ?? 0);
        }
      }
      for (let j = 0; j < p; j++) {
        result[j] = (result[j] ?? 0) + ((this.mean_ as Float64Array)[j] ?? 0);
      }
      return result;
    });
  }
}

export class TruncatedSVD {
  nComponents: number;
  nIter: number;

  components_: Float64Array[] | null = null;
  explainedVariance_: Float64Array | null = null;
  explainedVarianceRatio_: Float64Array | null = null;

  constructor(
    options: { nComponents?: number; nIter?: number } = {},
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.nIter = options.nIter ?? 5;
  }

  fit(X: Float64Array[]): this {
    const { components, explainedVariance } = randomizedSVD(X, this.nComponents, this.nIter);
    this.components_ = components;
    this.explainedVariance_ = explainedVariance;
    const totalVar = Array.from(explainedVariance).reduce((a, b) => a + b, 0);
    this.explainedVarianceRatio_ = new Float64Array(
      explainedVariance.map((v) => (totalVar > 0 ? v / totalVar : 0)),
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null) throw new NotFittedError("TruncatedSVD");
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = this.components_.length;
    return X.map((xi) => {
      const result = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        const comp = (this.components_ as Float64Array[])[c] ?? new Float64Array(p);
        let dot = 0;
        for (let j = 0; j < p; j++) dot += (xi[j] ?? 0) * (comp[j] ?? 0);
        result[c] = dot;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
