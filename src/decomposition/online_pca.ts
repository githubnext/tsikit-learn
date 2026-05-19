/**
 * Online PCA (IncrementalPCA) and Randomized SVD extensions.
 * Mirrors sklearn.decomposition.IncrementalPCA.
 */

import { safeDot } from "../utils/extmath.js";

export interface IncrementalPCAOptions {
  nComponents?: number | null;
  whiten?: boolean;
  copyData?: boolean;
  batchSize?: number | null;
}

/**
 * Incremental principal components analysis (IPCA).
 * Processes data in batches for memory efficiency.
 */
export class IncrementalPCAOnline {
  nComponents: number | null;
  whiten: boolean;
  batchSize: number | null;

  components_: Float64Array[] | null = null;
  explainedVariance_: Float64Array | null = null;
  explainedVarianceRatio_: Float64Array | null = null;
  singularValues_: Float64Array | null = null;
  mean_: Float64Array | null = null;
  variances_: Float64Array | null = null;
  nSamplesSeen_: number = 0;
  noiseVariance_: number = 0;
  nFeatures_: number = 0;
  nBatches_: number = 0;

  constructor(options: IncrementalPCAOptions = {}) {
    this.nComponents = options.nComponents ?? null;
    this.whiten = options.whiten ?? false;
    this.batchSize = options.batchSize ?? null;
  }

  partialFit(X: Float64Array[], y?: unknown): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    const k = this.nComponents ?? Math.min(nSamples, nFeatures);

    // Update running mean
    if (this.mean_ === null) {
      this.mean_ = new Float64Array(nFeatures);
      this.variances_ = new Float64Array(nFeatures);
    }

    // Welford's algorithm for mean/variance
    for (const row of X) {
      this.nSamplesSeen_++;
      for (let j = 0; j < nFeatures; j++) {
        const delta = (row[j] ?? 0) - (this.mean_![j] ?? 0);
        this.mean_![j] = (this.mean_![j] ?? 0) + delta / this.nSamplesSeen_;
        const delta2 = (row[j] ?? 0) - (this.mean_![j] ?? 0);
        this.variances_![j] = (this.variances_![j] ?? 0) + delta * delta2;
      }
    }

    // Center data
    const Xc = X.map(row => {
      const r = new Float64Array(row);
      for (let j = 0; j < nFeatures; j++) r[j] = (r[j] ?? 0) - (this.mean_![j] ?? 0);
      return r;
    });

    // Augment with existing components
    const augmented = this.components_
      ? [...Xc, ...this.components_.map(c => {
          const sv = this.singularValues_?.[0] ?? 1;
          return new Float64Array(c.map(v => v * sv));
        })]
      : Xc;

    // Truncated SVD on augmented matrix (power iteration)
    const cols = nFeatures;
    const rows = augmented.length;
    const effectiveK = Math.min(k, cols, rows);

    // Initialize V via random
    let seed = this.nBatches_ * 1000;
    const V: Float64Array[] = [];
    for (let c = 0; c < effectiveK; c++) {
      const v = new Float64Array(cols);
      let norm = 0;
      for (let j = 0; j < cols; j++) {
        seed = (seed * 1664525 + 1013904223) & 0xffffffff;
        v[j] = ((seed >>> 0) / 0xffffffff) * 2 - 1;
        norm += v[j] ** 2;
      }
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < cols; j++) v[j] = (v[j] ?? 0) / norm;
      V.push(v);
    }

    // Power iteration for dominant singular vectors
    for (let iter = 0; iter < 3; iter++) {
      for (let c = 0; c < effectiveK; c++) {
        // A^T A v
        const u = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) u[i] += (augmented[i]?.[j] ?? 0) * (V[c]?.[j] ?? 0);
        }
        const vNew = new Float64Array(cols);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) vNew[j] += (augmented[i]?.[j] ?? 0) * (u[i] ?? 0);
        }
        // Orthogonalize against previous
        for (let p = 0; p < c; p++) {
          let dot = 0;
          for (let j = 0; j < cols; j++) dot += (vNew[j] ?? 0) * (V[p]?.[j] ?? 0);
          for (let j = 0; j < cols; j++) vNew[j] = (vNew[j] ?? 0) - dot * (V[p]?.[j] ?? 0);
        }
        let norm = 0;
        for (let j = 0; j < cols; j++) norm += (vNew[j] ?? 0) ** 2;
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < cols; j++) V[c]![j] = (vNew[j] ?? 0) / norm;
      }
    }

    this.components_ = V.slice(0, effectiveK);

    // Compute singular values
    this.singularValues_ = new Float64Array(effectiveK);
    this.explainedVariance_ = new Float64Array(effectiveK);
    for (let c = 0; c < effectiveK; c++) {
      let sv = 0;
      for (let i = 0; i < rows; i++) {
        let proj = 0;
        for (let j = 0; j < cols; j++) proj += (augmented[i]?.[j] ?? 0) * (V[c]?.[j] ?? 0);
        sv += proj ** 2;
      }
      this.singularValues_[c] = Math.sqrt(sv);
      this.explainedVariance_[c] = sv / Math.max(this.nSamplesSeen_ - 1, 1);
    }

    const totalVar = this.explainedVariance_.reduce((s, v) => s + v, 0);
    this.explainedVarianceRatio_ = new Float64Array(this.explainedVariance_.map(v => v / (totalVar || 1)));
    this.nBatches_++;
    return this;
  }

  fit(X: Float64Array[]): this {
    const batchSize = this.batchSize ?? Math.max(10, Math.min(100, X.length));
    this.nSamplesSeen_ = 0;
    this.mean_ = null;
    this.components_ = null;

    for (let start = 0; start < X.length; start += batchSize) {
      const batch = X.slice(start, start + batchSize);
      this.partialFit(batch);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || !this.mean_) throw new Error("IncrementalPCA not fitted");
    const k = this.components_.length;
    const nFeatures = this.nFeatures_;

    return X.map(row => {
      const xc = new Float64Array(row);
      for (let j = 0; j < nFeatures; j++) xc[j] = (xc[j] ?? 0) - (this.mean_![j] ?? 0);
      const out = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let j = 0; j < nFeatures; j++) dot += (xc[j] ?? 0) * (this.components_![c]?.[j] ?? 0);
        if (this.whiten) dot /= (this.singularValues_?.[c] ?? 1) + 1e-10;
        out[c] = dot;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  getCovariance(): Float64Array[] {
    if (!this.components_) throw new Error("IncrementalPCA not fitted");
    const nFeatures = this.nFeatures_;
    const k = this.components_.length;
    const cov: Float64Array[] = Array.from({ length: nFeatures }, () => new Float64Array(nFeatures));
    for (let c = 0; c < k; c++) {
      const sv2 = (this.explainedVariance_?.[c] ?? 0);
      for (let i = 0; i < nFeatures; i++) {
        for (let j = 0; j < nFeatures; j++) {
          cov[i]![j] = (cov[i]![j] ?? 0) + sv2 * (this.components_[c]?.[i] ?? 0) * (this.components_[c]?.[j] ?? 0);
        }
      }
    }
    return cov;
  }
}
