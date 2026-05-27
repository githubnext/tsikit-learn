/**
 * Additional decomposition: SparsePCA, DictionaryLearning extensions.
 * Mirrors sklearn.decomposition extras.
 */

import { NotFittedError } from "../exceptions.js";

export class MiniBatchSparsePCA {
  nComponents: number;
  alpha: number;
  batchSize: number;
  maxIter: number;
  randomState: number;

  components_: Float64Array[] | null = null;
  meanValues_: Float64Array | null = null;

  constructor(
    options: {
      nComponents?: number;
      alpha?: number;
      batchSize?: number;
      maxIter?: number;
      randomState?: number;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 10;
    this.alpha = options.alpha ?? 1.0;
    this.batchSize = options.batchSize ?? 200;
    this.maxIter = options.maxIter ?? 100;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const k = Math.min(this.nComponents, nFeatures);

    // Compute mean
    const mean = new Float64Array(nFeatures);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) mean[j] = (mean[j] ?? 0) + (row[j] ?? 0);
    }
    for (let j = 0; j < nFeatures; j++) mean[j] = (mean[j] ?? 0) / n;
    this.meanValues_ = mean;

    // Initialize dictionary randomly
    let rng = this.randomState;
    const nextRand = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return (rng / 4294967296) * 2 - 1;
    };

    const D: Float64Array[] = Array.from({ length: k }, () => {
      const v = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) v[j] = nextRand();
      let norm = 0;
      for (const vj of v) norm += vj ** 2;
      norm = Math.sqrt(norm);
      if (norm > 0) for (let j = 0; j < nFeatures; j++) v[j] = (v[j] ?? 0) / norm;
      return v;
    });

    // Mini-batch iterations
    for (let iter = 0; iter < this.maxIter; iter++) {
      const batchSize = Math.min(this.batchSize, n);
      const startIdx = (iter * batchSize) % n;
      const batch = X.slice(startIdx, startIdx + batchSize);

      // Update dictionary (simplified)
      for (const x of batch) {
        // Sparse code via lasso
        const codes = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          let dot = 0;
          for (let j = 0; j < nFeatures; j++) {
            dot += ((x[j] ?? 0) - (mean[j] ?? 0)) * (D[c]?.[j] ?? 0);
          }
          const threshold = this.alpha / batchSize;
          codes[c] = dot > threshold ? dot - threshold : dot < -threshold ? dot + threshold : 0;
        }

        // Update dictionary atoms
        for (let c = 0; c < k; c++) {
          if (Math.abs(codes[c] ?? 0) < 1e-10) continue;
          for (let j = 0; j < nFeatures; j++) {
            D[c]![j] = (D[c]?.[j] ?? 0) + (codes[c] ?? 0) * ((x[j] ?? 0) - (mean[j] ?? 0));
          }
          // Normalize
          let norm = 0;
          for (const dj of D[c]!) norm += dj ** 2;
          norm = Math.sqrt(norm);
          if (norm > 0) for (let j = 0; j < nFeatures; j++) D[c]![j] = (D[c]?.[j] ?? 0) / norm;
        }
      }
    }

    this.components_ = D;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || !this.meanValues_) throw new NotFittedError("MiniBatchSparsePCA is not fitted");
    const k = this.components_.length;
    const nFeatures = this.meanValues_.length;
    return X.map((x) => {
      const codes = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let j = 0; j < nFeatures; j++) {
          dot += ((x[j] ?? 0) - (this.meanValues_![j] ?? 0)) * (this.components_![c]?.[j] ?? 0);
        }
        codes[c] = dot;
      }
      return codes;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class MiniBatchDictionaryLearning {
  nComponents: number;
  alpha: number;
  batchSize: number;
  maxIter: number;
  randomState: number;

  components_: Float64Array[] | null = null;

  constructor(
    options: {
      nComponents?: number;
      alpha?: number;
      batchSize?: number;
      maxIter?: number;
      randomState?: number;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 10;
    this.alpha = options.alpha ?? 1.0;
    this.batchSize = options.batchSize ?? 200;
    this.maxIter = options.maxIter ?? 100;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const k = Math.min(this.nComponents, nFeatures);

    let rng = this.randomState;
    const nextRand = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return (rng / 4294967296) * 2 - 1;
    };

    const D: Float64Array[] = Array.from({ length: k }, () => {
      const v = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) v[j] = nextRand();
      let norm = 0;
      for (const vj of v) norm += vj ** 2;
      norm = Math.sqrt(norm);
      if (norm > 0) for (let j = 0; j < nFeatures; j++) v[j] = (v[j] ?? 0) / norm;
      return v;
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      const batchSize = Math.min(this.batchSize, n);
      const batch = X.slice((iter * batchSize) % n, (iter * batchSize) % n + batchSize);

      for (const x of batch) {
        const codes = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          let dot = 0;
          for (let j = 0; j < nFeatures; j++) dot += (x[j] ?? 0) * (D[c]?.[j] ?? 0);
          const thr = this.alpha / Math.max(batchSize, 1);
          codes[c] = dot > thr ? dot - thr : dot < -thr ? dot + thr : 0;
        }

        for (let c = 0; c < k; c++) {
          if (Math.abs(codes[c] ?? 0) < 1e-10) continue;
          for (let j = 0; j < nFeatures; j++) {
            D[c]![j] = (D[c]?.[j] ?? 0) + (codes[c] ?? 0) * (x[j] ?? 0);
          }
          let norm = 0;
          for (const dj of D[c]!) norm += dj ** 2;
          norm = Math.sqrt(norm);
          if (norm > 0) for (let j = 0; j < nFeatures; j++) D[c]![j] = (D[c]?.[j] ?? 0) / norm;
        }
      }
    }

    this.components_ = D;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new NotFittedError("MiniBatchDictionaryLearning is not fitted");
    const k = this.components_.length;
    const nFeatures = this.components_[0]?.length ?? 0;
    return X.map((x) => {
      const codes = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let j = 0; j < nFeatures; j++) dot += (x[j] ?? 0) * (this.components_![c]?.[j] ?? 0);
        codes[c] = dot;
      }
      return codes;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
