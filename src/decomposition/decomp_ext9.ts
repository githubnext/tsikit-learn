/**
 * Decomposition extensions: Robust PCA, NNLS, Archetypal Analysis.
 * Mirrors sklearn.decomposition additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Non-negative Least Squares solver for NMF-style problems. */
export function nnls(A: Float64Array[], b: Float64Array, maxIter = 200): Float64Array {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  const x = new Float64Array(n);
  const tol = 1e-6;
  for (let iter = 0; iter < maxIter; iter++) {
    let maxChange = 0;
    for (let j = 0; j < n; j++) {
      // Compute gradient component j
      let grad = 0;
      for (let i = 0; i < m; i++) {
        let ax = 0;
        for (let k = 0; k < n; k++) ax += (A[i]?.[k] ?? 0) * (x[k] ?? 0);
        grad += (A[i]?.[j] ?? 0) * (ax - (b[i] ?? 0));
      }
      const old = x[j] ?? 0;
      x[j] = Math.max(0, (x[j] ?? 0) - 0.01 * grad);
      maxChange = Math.max(maxChange, Math.abs((x[j] ?? 0) - old));
    }
    if (maxChange < tol) break;
  }
  return x;
}

export interface RobustPCAParams {
  n_components?: number;
  lambda_?: number;
  max_iter?: number;
  tol?: number;
}

/** Robust PCA via Principal Component Pursuit (L+S decomposition). */
export class RobustPCA extends BaseEstimator {
  n_components: number;
  lambda_: number;
  max_iter: number;
  tol: number;
  low_rank_: Float64Array[] = [];
  sparse_: Float64Array[] = [];
  components_: Float64Array[] = [];

  constructor(params: RobustPCAParams = {}) {
    super();
    this.n_components = params.n_components ?? 1;
    this.lambda_ = params.lambda_ ?? -1;
    this.max_iter = params.max_iter ?? 100;
    this.tol = params.tol ?? 1e-7;
  }

  fit(X: Float64Array[]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    const lambda = this.lambda_ < 0 ? 1 / Math.sqrt(Math.max(m, n)) : this.lambda_;

    const L: Float64Array[] = X.map(r => new Float64Array(r));
    const S: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n));
    const mu = m * n / (4 * l1norm(X));

    const shrink = (v: number, tau: number): number =>
      v > tau ? v - tau : v < -tau ? v + tau : 0;

    for (let iter = 0; iter < this.max_iter; iter++) {
      // SVT step for L
      const svd = thinSVD(L, this.n_components);
      const sigmas = svd.s.map(s => shrink(s, 1 / mu));
      const newL: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n));
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          for (let r = 0; r < this.n_components; r++) {
            newL[i]![j] = (newL[i]?.[j] ?? 0) + (svd.U[i]?.[r] ?? 0) * (sigmas[r] ?? 0) * (svd.Vt[r]?.[j] ?? 0);
          }
        }
      }
      // Sparse step
      let change = 0;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          const res = (X[i]?.[j] ?? 0) - (newL[i]?.[j] ?? 0);
          const newS = shrink(res, lambda / mu);
          change += Math.abs(newS - (S[i]?.[j] ?? 0));
          S[i]![j] = newS;
          L[i]![j] = (newL[i]?.[j] ?? 0);
        }
      }
      if (change < this.tol) break;
    }
    this.low_rank_ = L;
    this.sparse_ = S;
    return this;
  }
}

function l1norm(X: Float64Array[]): number {
  let s = 0;
  for (const row of X) for (const v of row) s += Math.abs(v);
  return s || 1;
}

function thinSVD(X: Float64Array[], k: number): { U: Float64Array[]; s: number[]; Vt: Float64Array[] } {
  const m = X.length;
  const n = X[0]?.length ?? 0;
  const r = Math.min(k, Math.min(m, n));
  const U: Float64Array[] = Array.from({ length: m }, () => new Float64Array(r));
  const s: number[] = new Array(r).fill(0) as number[];
  const Vt: Float64Array[] = Array.from({ length: r }, () => new Float64Array(n));
  // Power iteration for top-r singular vectors
  for (let ri = 0; ri < r; ri++) {
    let v = new Float64Array(n);
    v[ri % n] = 1;
    for (let iter = 0; iter < 20; iter++) {
      // u = X @ v
      const u = new Float64Array(m);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) u[i] += (X[i]?.[j] ?? 0) * (v[j] ?? 0);
      let norm = 0;
      for (const uv of u) norm += uv * uv;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < m; i++) u[i] /= norm;
      // v = X.T @ u
      const vnew = new Float64Array(n);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) vnew[j] += (X[i]?.[j] ?? 0) * (u[i] ?? 0);
      let vnorm = 0;
      for (const vv of vnew) vnorm += vv * vv;
      vnorm = Math.sqrt(vnorm) || 1;
      for (let j = 0; j < n; j++) vnew[j] /= vnorm;
      v = vnew;
      s[ri] = vnorm;
    }
    for (let i = 0; i < m; i++) {
      let ui = 0;
      for (let j = 0; j < n; j++) ui += (X[i]?.[j] ?? 0) * (v[j] ?? 0);
      U[i]![ri] = ui / (s[ri] ?? 1);
    }
    for (let j = 0; j < n; j++) Vt[ri]![j] = v[j] ?? 0;
    // Deflate
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        X[i]![j] = (X[i]?.[j] ?? 0) - (U[i]?.[ri] ?? 0) * (s[ri] ?? 0) * (Vt[ri]?.[j] ?? 0);
      }
    }
  }
  return { U, s, Vt };
}

export interface ArchetypalAnalysisParams {
  n_archetypes?: number;
  max_iter?: number;
  tol?: number;
}

/** Archetypal Analysis: find extreme points (archetypes) that best represent data. */
export class ArchetypalAnalysis extends BaseEstimator {
  n_archetypes: number;
  max_iter: number;
  tol: number;
  archetypes_: Float64Array[] = [];
  coef_: Float64Array[] = [];

  constructor(params: ArchetypalAnalysisParams = {}) {
    super();
    this.n_archetypes = params.n_archetypes ?? 4;
    this.max_iter = params.max_iter ?? 200;
    this.tol = params.tol ?? 1e-4;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const k = this.n_archetypes;
    // Initialize archetypes as random data points
    const idx = Array.from({ length: k }, (_, i) => Math.floor(i * n / k));
    let archetypes: Float64Array[] = idx.map(i => new Float64Array(X[i]!));

    for (let iter = 0; iter < this.max_iter; iter++) {
      // Update alpha (coefficients for data points)
      const alpha: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        // Simplex projection
        const scores = new Float64Array(k);
        for (let j = 0; j < k; j++) {
          let s = 0;
          for (let f = 0; f < d; f++) s -= ((X[i]?.[f] ?? 0) - (archetypes[j]?.[f] ?? 0)) ** 2;
          scores[j] = s;
        }
        const expScores = scores.map(s => Math.exp(s - Math.max(...scores)));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        for (let j = 0; j < k; j++) alpha[i]![j] = (expScores[j] ?? 0) / sumExp;
      }
      // Update archetypes
      const newArcH: Float64Array[] = Array.from({ length: k }, () => new Float64Array(d));
      const betaDen = new Float64Array(k);
      for (let j = 0; j < k; j++) {
        for (let i = 0; i < n; i++) {
          betaDen[j] = (betaDen[j] ?? 0) + (alpha[i]?.[j] ?? 0);
        }
        for (let i = 0; i < n; i++) {
          const w = (alpha[i]?.[j] ?? 0) / (betaDen[j] ?? 1);
          for (let f = 0; f < d; f++) newArcH[j]![f] = (newArcH[j]?.[f] ?? 0) + w * (X[i]?.[f] ?? 0);
        }
      }

      let change = 0;
      for (let j = 0; j < k; j++) for (let f = 0; f < d; f++) {
        change += Math.abs((newArcH[j]?.[f] ?? 0) - (archetypes[j]?.[f] ?? 0));
      }
      archetypes = newArcH;
      this.coef_ = alpha;
      if (change < this.tol) break;
    }
    this.archetypes_ = archetypes;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const k = this.n_archetypes;
    const d = X[0]?.length ?? 0;
    return Array.from({ length: n }, (_, i) => {
      const scores = new Float64Array(k);
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let f = 0; f < d; f++) s -= ((X[i]?.[f] ?? 0) - (this.archetypes_[j]?.[f] ?? 0)) ** 2;
        scores[j] = s;
      }
      const expScores = scores.map(s => Math.exp(s - Math.max(...scores)));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      return expScores.map(e => e / sumExp);
    });
  }
}
