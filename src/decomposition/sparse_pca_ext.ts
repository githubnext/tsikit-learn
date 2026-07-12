/**
 * SparsePCA and MiniBatchSparsePCA.
 * Mirrors sklearn.decomposition.SparsePCA and MiniBatchSparsePCA.
 */

export interface SparsePCAOptions {
  nComponents?: number;
  alpha?: number;
  ridgeAlpha?: number;
  maxIter?: number;
  tol?: number;
  method?: "lars" | "cd";
  nJobs?: number | null;
  verbose?: boolean;
  randomState?: number | null;
}

/**
 * Sparse Principal Components Analysis (SparsePCA).
 * Finds sparse components that can optimally reconstruct data.
 */
export class SparsePCA {
  nComponents: number;
  alpha: number;
  ridgeAlpha: number;
  maxIter: number;
  tol: number;
  method: string;
  verbose: boolean;
  randomState: number | null;

  components_: Float64Array[] | null = null;
  errorReduction_: number[] | null = null;
  nIter_: number = 0;
  mean_: Float64Array | null = null;
  nFeatures_: number = 0;

  constructor(options: SparsePCAOptions = {}) {
    this.nComponents = options.nComponents ?? 10;
    this.alpha = options.alpha ?? 1.0;
    this.ridgeAlpha = options.ridgeAlpha ?? 0.01;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-8;
    this.method = options.method ?? "lars";
    this.verbose = options.verbose ?? false;
    this.randomState = options.randomState ?? null;
  }

  fit(X: Float64Array[]): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;
    const k = Math.min(this.nComponents, nFeatures, nSamples);

    // Compute mean
    this.mean_ = new Float64Array(nFeatures);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) this.mean_[j] = (this.mean_[j] ?? 0) + (row[j] ?? 0);
    }
    for (let j = 0; j < nFeatures; j++) this.mean_[j] = (this.mean_[j] ?? 0) / nSamples;

    // Initialize dictionary with random atoms
    let seed = this.randomState ?? 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }

    // Initialize components randomly and normalize
    this.components_ = [];
    for (let c = 0; c < k; c++) {
      const comp = new Float64Array(nFeatures);
      let norm = 0;
      for (let j = 0; j < nFeatures; j++) { comp[j] = rand(); norm += comp[j]! ** 2; }
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < nFeatures; j++) comp[j] = (comp[j] ?? 0) / norm;
      this.components_.push(comp);
    }

    // Dictionary learning via alternating minimization (simplified)
    const errors: number[] = [];
    const codes = Array.from({ length: nSamples }, () => new Float64Array(k));

    for (let iter = 0; iter < Math.min(this.maxIter, 100); iter++) {
      // Sparse coding step (lasso-like for each sample)
      for (let i = 0; i < nSamples; i++) {
        const xi = X[i]!;
        for (let c = 0; c < k; c++) {
          // Simple soft thresholding
          let dot = 0;
          for (let j = 0; j < nFeatures; j++) dot += (xi[j] ?? 0) * (this.components_![c]?.[j] ?? 0);
          const sign = dot > 0 ? 1 : -1;
          codes[i]![c] = sign * Math.max(Math.abs(dot) - this.alpha / nSamples, 0);
        }
      }

      // Dictionary update step
      for (let c = 0; c < k; c++) {
        const newComp = new Float64Array(nFeatures);
        let weight = 0;
        for (let i = 0; i < nSamples; i++) {
          const ci = codes[i]?.[c] ?? 0;
          if (Math.abs(ci) < 1e-10) continue;
          weight += ci ** 2;
          for (let j = 0; j < nFeatures; j++) {
            newComp[j] = (newComp[j] ?? 0) + ci * (X[i]?.[j] ?? 0);
          }
        }
        // Ridge regularization + normalize
        const denom = weight + this.ridgeAlpha;
        let norm = 0;
        for (let j = 0; j < nFeatures; j++) {
          newComp[j] = (newComp[j] ?? 0) / denom;
          norm += (newComp[j] ?? 0) ** 2;
        }
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < nFeatures; j++) newComp[j] = (newComp[j] ?? 0) / norm;
        this.components_[c] = newComp;
      }

      // Compute reconstruction error
      let totalError = 0;
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          let recon = 0;
          for (let c = 0; c < k; c++) recon += (codes[i]?.[c] ?? 0) * (this.components_![c]?.[j] ?? 0);
          totalError += ((X[i]?.[j] ?? 0) - recon) ** 2;
        }
      }
      errors.push(totalError);
      if (iter > 0 && Math.abs((errors[iter - 1] ?? 0) - totalError) < this.tol) break;
      this.nIter_ = iter + 1;
    }
    this.errorReduction_ = errors;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("SparsePCA not fitted");
    const k = this.components_.length;
    const nFeatures = this.nFeatures_;

    return X.map(xi => {
      const code = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let j = 0; j < nFeatures; j++) dot += (xi[j] ?? 0) * (this.components_![c]?.[j] ?? 0);
        const sign = dot > 0 ? 1 : -1;
        code[c] = sign * Math.max(Math.abs(dot) - this.alpha, 0);
      }
      return code;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface MiniBatchSparsePCAOptions extends SparsePCAOptions {
  batchSize?: number;
  shuffle?: boolean;
  nIter?: number;
}

/**
 * Mini-batch SparsePCA for large datasets.
 */
export class MiniBatchSparsePCA extends SparsePCA {
  batchSize: number;
  shuffle: boolean;
  nIterMB: number;

  constructor(options: MiniBatchSparsePCAOptions = {}) {
    super(options);
    this.batchSize = options.batchSize ?? 3;
    this.shuffle = options.shuffle ?? true;
    this.nIterMB = options.nIter ?? 1000;
  }

  override fit(X: Float64Array[]): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;
    const k = Math.min(this.nComponents, nFeatures, nSamples);

    let seed = this.randomState ?? 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }
    function randIdx(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) % nSamples;
    }

    this.components_ = [];
    for (let c = 0; c < k; c++) {
      const comp = new Float64Array(nFeatures);
      let norm = 0;
      for (let j = 0; j < nFeatures; j++) { comp[j] = rand(); norm += comp[j]! ** 2; }
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < nFeatures; j++) comp[j] = (comp[j] ?? 0) / norm;
      this.components_.push(comp);
    }

    const nBatches = Math.min(this.nIterMB, Math.ceil(nSamples / this.batchSize));
    for (let iter = 0; iter < nBatches; iter++) {
      // Sample batch
      const batch: Float64Array[] = [];
      for (let b = 0; b < this.batchSize; b++) {
        const idx = this.shuffle ? randIdx() : (iter * this.batchSize + b) % nSamples;
        batch.push(X[idx]!);
      }

      // Update components on this mini-batch
      for (let c = 0; c < k; c++) {
        const newComp = new Float64Array(nFeatures);
        let weight = 0;
        for (const xi of batch) {
          let dot = 0;
          for (let j = 0; j < nFeatures; j++) dot += (xi[j] ?? 0) * (this.components_![c]?.[j] ?? 0);
          const ci = Math.sign(dot) * Math.max(Math.abs(dot) - this.alpha, 0);
          if (Math.abs(ci) < 1e-10) continue;
          weight += ci ** 2;
          for (let j = 0; j < nFeatures; j++) newComp[j] = (newComp[j] ?? 0) + ci * (xi[j] ?? 0);
        }
        if (weight < 1e-10) continue;
        let norm = 0;
        for (let j = 0; j < nFeatures; j++) {
          newComp[j] = (newComp[j] ?? 0) / (weight + this.ridgeAlpha);
          norm += (newComp[j] ?? 0) ** 2;
        }
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < nFeatures; j++) newComp[j] = (newComp[j] ?? 0) / norm;
        this.components_[c] = newComp;
      }
    }
    this.nIter_ = nBatches;
    return this;
  }
}
