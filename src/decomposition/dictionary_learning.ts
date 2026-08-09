/**
 * DictionaryLearning and SparsePCA.
 * Mirrors sklearn.decomposition.DictionaryLearning and SparsePCA.
 */

import { NotFittedError } from "../exceptions.js";

export interface DictionaryLearningOptions {
  nComponents?: number;
  alpha?: number;
  maxIter?: number;
  tol?: number;
  fitAlgorithm?: "lars" | "cd";
  transformAlgorithm?: "lasso_lars" | "lasso_cd" | "lars" | "omp" | "threshold";
  splitSign?: boolean;
  nJobs?: number | null;
  codeTol?: number;
  randomState?: number;
  positiveCode?: boolean;
  positiveDict?: boolean;
  transformMaxIter?: number;
}

/**
 * DictionaryLearning — sparse coding dictionary learning.
 * Finds a dictionary D such that X ≈ code @ D with sparse code.
 */
export class DictionaryLearning {
  nComponents: number;
  alpha: number;
  maxIter: number;
  tol: number;
  randomState: number;
  nIter_: number = 0;

  components_: Float64Array[] | null = null;
  errorArray_: Float64Array | null = null;
  nFeatureIn_: number = 0;

  constructor(options: DictionaryLearningOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.alpha = options.alpha ?? 1.0;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-8;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeatureIn_ = p;
    const k = this.nComponents;

    let rng = this.randomState;
    const nextRng = () => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return (rng / 4294967296) * 2 - 1;
    };

    // Initialize dictionary as random rows from X
    const D: Float64Array[] = Array.from({ length: k }, () => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j]! = nextRng();
      return row;
    });

    // Normalize dictionary atoms
    const normD = (di: Float64Array) => {
      let norm = 0;
      for (let j = 0; j < p; j++) norm += (di[j] ?? 0) ** 2;
      norm = Math.sqrt(norm);
      if (norm > 1e-10) for (let j = 0; j < p; j++) di[j]! /= norm;
    };
    D.forEach(normD);

    const errors = new Float64Array(this.maxIter);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sparse coding step: for each x_i, find code c_i that minimizes ||x_i - c_i @ D||^2 + alpha*||c_i||_1
      const codes: Float64Array[] = X.map((xi) => this._lasso(D, xi, p, k));

      // Dictionary update step: for each atom d_j, update via ridge regression
      for (let j = 0; j < k; j++) {
        const cj = new Float64Array(n);
        for (let i = 0; i < n; i++) cj[i]! = codes[i]![j] ?? 0;

        let cNorm2 = 0;
        for (let i = 0; i < n; i++) cNorm2 += (cj[i] ?? 0) ** 2;
        if (cNorm2 < 1e-12) continue;

        // residual = X - sum_{l!=j} c_l * d_l
        for (let ff = 0; ff < p; ff++) {
          let r = 0;
          for (let i = 0; i < n; i++) {
            let xi_approx_no_j = 0;
            for (let l = 0; l < k; l++) {
              if (l === j) continue;
              xi_approx_no_j += (codes[i]![l] ?? 0) * (D[l]![ff] ?? 0);
            }
            r += (cj[i] ?? 0) * ((X[i]![ff] ?? 0) - xi_approx_no_j);
          }
          D[j]![ff]! = r / cNorm2;
        }
        normD(D[j]!);
      }

      // Compute reconstruction error
      let err = 0;
      for (let i = 0; i < n; i++) {
        for (let ff = 0; ff < p; ff++) {
          let approx = 0;
          for (let j = 0; j < k; j++)
            approx += (codes[i]![j] ?? 0) * (D[j]![ff] ?? 0);
          err += ((X[i]![ff] ?? 0) - approx) ** 2;
        }
      }
      errors[iter]! = err;
      this.nIter_ = iter + 1;
      if (iter > 0 && Math.abs((errors[iter - 1] ?? 0) - err) < this.tol) break;
    }

    this.components_ = D;
    this.errorArray_ = errors;
    return this;
  }

  private _lasso(
    D: Float64Array[],
    xi: Float64Array,
    p: number,
    k: number,
  ): Float64Array {
    // Simple proximal gradient for lasso: minimize 0.5||xi - c@D||^2 + alpha*||c||_1
    const c = new Float64Array(k);
    const lr = 0.01;
    const thresh = this.alpha * lr;
    for (let iter = 0; iter < 50; iter++) {
      // gradient of smooth part
      const grad = new Float64Array(k);
      for (let j = 0; j < k; j++) {
        let residj = 0;
        for (let ff = 0; ff < p; ff++) {
          let approx = 0;
          for (let l = 0; l < k; l++) approx += (c[l] ?? 0) * (D[l]![ff] ?? 0);
          residj += -((xi[ff] ?? 0) - approx) * (D[j]![ff] ?? 0);
        }
        grad[j]! = residj;
      }
      // proximal step
      for (let j = 0; j < k; j++) {
        const v = (c[j] ?? 0) - lr * (grad[j] ?? 0);
        c[j]! = Math.sign(v) * Math.max(0, Math.abs(v) - thresh);
      }
    }
    return c;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_)
      throw new NotFittedError("DictionaryLearning is not fitted");
    const k = this.nComponents;
    const p = this.nFeatureIn_;
    return X.map((xi) => this._lasso(this.components_!, xi, p, k));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface SparsePCAOptions {
  nComponents?: number;
  alpha?: number;
  ridge_alpha?: number;
  maxIter?: number;
  tol?: number;
  method?: "lars" | "cd";
  nJobs?: number | null;
  verbose?: boolean;
  randomState?: number;
}

/**
 * SparsePCA — sparse principal component analysis.
 * Finds sparse components via dictionary learning with L1 penalty on the codes.
 */
export class SparsePCA {
  nComponents: number;
  alpha: number;
  ridgeAlpha: number;
  maxIter: number;
  tol: number;
  randomState: number;
  nIter_: number = 0;

  components_: Float64Array[] | null = null;
  mean_: Float64Array | null = null;
  nFeatureIn_: number = 0;
  error_: Float64Array | null = null;

  constructor(options: SparsePCAOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.alpha = options.alpha ?? 1.0;
    this.ridgeAlpha = options.ridge_alpha ?? 0.01;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-8;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeatureIn_ = p;

    // Compute mean and center
    const mean = new Float64Array(p);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < p; j++) mean[j]! += (X[i]![j] ?? 0) / n;
    this.mean_ = mean;
    const Xc = X.map((xi) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j]! = (xi[j] ?? 0) - (mean[j] ?? 0);
      return r;
    });

    const dl = new DictionaryLearning({
      nComponents: this.nComponents,
      alpha: this.alpha,
      maxIter: this.maxIter,
      tol: this.tol,
      randomState: this.randomState,
    });
    dl.fit(Xc);
    this.components_ = dl.components_;
    this.nIter_ = dl.nIter_;
    this.error_ = dl.errorArray_;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || !this.mean_)
      throw new NotFittedError("SparsePCA is not fitted");
    const p = this.nFeatureIn_;
    const mean = this.mean_;
    const Xc = X.map((xi) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j]! = (xi[j] ?? 0) - (mean[j] ?? 0);
      return r;
    });
    const dl = new DictionaryLearning({
      nComponents: this.nComponents,
      alpha: this.alpha,
      maxIter: 50,
      randomState: this.randomState,
    });
    dl.components_ = this.components_;
    dl.nFeatureIn_ = p;
    return dl.transform(Xc);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
