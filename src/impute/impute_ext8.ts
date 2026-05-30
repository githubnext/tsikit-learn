/**
 * Imputation extensions: IterativeImputer extensions, gain imputer.
 * Mirrors sklearn.impute advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface ChainedEquationImputerParams {
  max_iter?: number;
  tol?: number;
  random_state?: number | null;
}

/** ChainedEquationImputer: MICE-style imputation. */
export class ChainedEquationImputer extends BaseEstimator {
  max_iter: number;
  tol: number;
  random_state: number | null;
  statistics_: Float64Array = new Float64Array(0);
  n_features_in_ = 0;

  constructor(params: ChainedEquationImputerParams = {}) {
    super();
    this.max_iter = params.max_iter ?? 10;
    this.tol = params.tol ?? 1e-3;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const nf = X[0]?.length ?? 0;
    this.n_features_in_ = nf;
    this.statistics_ = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      const vals = X.map((xi) => xi[k] ?? Number.NaN).filter((v) => !Number.isNaN(v));
      this.statistics_[k] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const nf = this.n_features_in_;
    let result = X.map((xi) => {
      const row = new Float64Array(nf);
      for (let k = 0; k < nf; k++) {
        const v = xi[k] ?? Number.NaN;
        row[k] = Number.isNaN(v) ? (this.statistics_[k] ?? 0) : v;
      }
      return row;
    });
    // Iterative imputation
    for (let iter = 0; iter < this.max_iter; iter++) {
      const prev = result.map((r) => new Float64Array(r));
      for (let k = 0; k < nf; k++) {
        const missingIdx = X.map((xi, i) => Number.isNaN(xi[k] ?? Number.NaN) ? i : -1).filter((i) => i >= 0);
        if (missingIdx.length === 0) continue;
        // Use other features to predict missing values (mean of neighbors)
        for (const i of missingIdx) {
          let weightedSum = 0, weightSum = 0;
          for (let j = 0; j < result.length; j++) {
            if (j === i) continue;
            let dist = 0;
            for (let f = 0; f < nf; f++) {
              if (f === k) continue;
              dist += ((result[i]?.[f] ?? 0) - (result[j]?.[f] ?? 0)) ** 2;
            }
            const w = Math.exp(-dist / (nf || 1));
            weightedSum += w * (result[j]?.[k] ?? 0);
            weightSum += w;
          }
          result[i]![k] = weightSum > 0 ? weightedSum / weightSum : (this.statistics_[k] ?? 0);
        }
      }
      let maxDelta = 0;
      for (let i = 0; i < result.length; i++) {
        for (let k = 0; k < nf; k++) {
          const delta = Math.abs((result[i]?.[k] ?? 0) - (prev[i]?.[k] ?? 0));
          if (delta > maxDelta) maxDelta = delta;
        }
      }
      if (maxDelta < this.tol) break;
    }
    return result;
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface SoftImputeParams {
  max_rank?: number;
  max_iter?: number;
  shrinkage_value?: number;
}

/** SoftImpute: matrix completion via nuclear norm minimization. */
export class SoftImpute extends BaseEstimator {
  max_rank: number;
  max_iter: number;
  shrinkage_value: number;
  n_features_in_ = 0;
  singular_values_: Float64Array = new Float64Array(0);

  constructor(params: SoftImputeParams = {}) {
    super();
    this.max_rank = params.max_rank ?? 10;
    this.max_iter = params.max_iter ?? 100;
    this.shrinkage_value = params.shrinkage_value ?? 0;
  }

  fit(X: Float64Array[]): this {
    this.n_features_in_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = this.n_features_in_;
    // Initialize with column means for missing values
    const colMeans = new Float64Array(p);
    const colCounts = new Int32Array(p);
    for (const xi of X) for (let k = 0; k < p; k++) {
      const v = xi[k] ?? Number.NaN;
      if (!Number.isNaN(v)) { colMeans[k] = (colMeans[k] ?? 0) + v; colCounts[k] = (colCounts[k] ?? 0) + 1; }
    }
    for (let k = 0; k < p; k++) colMeans[k] = (colCounts[k] ?? 0) > 0 ? (colMeans[k] ?? 0) / (colCounts[k] ?? 1) : 0;
    let Z = X.map((xi) => new Float64Array(p).map((_, k) => {
      const v = xi[k] ?? Number.NaN;
      return Number.isNaN(v) ? (colMeans[k] ?? 0) : v;
    }));
    const lambda = this.shrinkage_value;
    for (let iter = 0; iter < this.max_iter; iter++) {
      // SVD truncated (power iteration for top singular vectors)
      const r = Math.min(this.max_rank, Math.min(n, p));
      const U: Float64Array[] = [], S: number[] = [], V: Float64Array[] = [];
      const Zc = Z.map((row) => new Float64Array(row));
      for (let c = 0; c < r; c++) {
        let v = new Float64Array(p).map((_, j) => j === c ? 1 : 0.01);
        let sigma = 0;
        for (let piter = 0; piter < 20; piter++) {
          const Av = new Float64Array(n).map((_, i) => {
            let s = 0; for (let j = 0; j < p; j++) s += (Zc[i]?.[j] ?? 0) * (v[j] ?? 0); return s;
          });
          let norm = 0; for (const vi of Av) norm += vi * vi; norm = Math.sqrt(norm);
          if (norm < 1e-10) break;
          const u = Av.map((vi) => vi / norm);
          const Atu = new Float64Array(p).map((_, j) => {
            let s = 0; for (let i = 0; i < n; i++) s += (Zc[i]?.[j] ?? 0) * (u[i] ?? 0); return s;
          });
          let norm2 = 0; for (const vi of Atu) norm2 += vi * vi; norm2 = Math.sqrt(norm2);
          sigma = norm2;
          if (norm2 < 1e-10) break;
          v = Atu.map((vi) => vi / norm2);
        }
        const sigShrunk = Math.max(sigma - lambda, 0);
        if (sigShrunk > 0) {
          const u = new Float64Array(n).map((_, i) => { let s = 0; for (let j = 0; j < p; j++) s += (Zc[i]?.[j] ?? 0) * (v[j] ?? 0); return s; });
          let un = 0; for (const vi of u) un += vi * vi; un = Math.sqrt(un);
          U.push(new Float64Array(u.map((vi) => vi / Math.max(un, 1e-10))));
          S.push(sigShrunk);
          V.push(v);
          // Deflate
          for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) Zc[i]![j] = (Zc[i]![j] ?? 0) - sigShrunk * (U[c]?.[i] ?? 0) * (v[j] ?? 0);
        }
      }
      const Znew = Array.from({ length: n }, () => new Float64Array(p));
      for (let c = 0; c < U.length; c++) for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) Znew[i]![j] = (Znew[i]![j] ?? 0) + (S[c] ?? 0) * (U[c]?.[i] ?? 0) * (V[c]?.[j] ?? 0);
      // Keep observed entries
      for (let i = 0; i < n; i++) for (let k = 0; k < p; k++) {
        const v = X[i]?.[k] ?? Number.NaN;
        if (!Number.isNaN(v)) Znew[i]![k] = v;
      }
      let maxDelta = 0;
      for (let i = 0; i < n; i++) for (let k = 0; k < p; k++) maxDelta = Math.max(maxDelta, Math.abs((Znew[i]?.[k] ?? 0) - (Z[i]?.[k] ?? 0)));
      Z = Znew;
      if (maxDelta < 1e-4) break;
    }
    return Z;
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
