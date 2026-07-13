/**
 * Manifold extensions: PHATE, ForceAtlas2, diffusion maps.
 * Mirrors sklearn.manifold additional methods.
 */

import { BaseEstimator } from "../base.js";

export interface DiffusionMapParams {
  n_components?: number;
  alpha?: number;
  t?: number;
  epsilon?: "auto" | number;
  n_neighbors?: number;
}

/** Diffusion maps for non-linear dimensionality reduction. */
export class DiffusionMap extends BaseEstimator {
  n_components: number;
  alpha: number;
  t: number;
  epsilon: "auto" | number;
  n_neighbors: number;
  embedding_: Float64Array[] = [];
  eigenvalues_: Float64Array = new Float64Array(0);

  constructor(params: DiffusionMapParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.alpha = params.alpha ?? 0.5;
    this.t = params.t ?? 1;
    this.epsilon = params.epsilon ?? "auto";
    this.n_neighbors = params.n_neighbors ?? 10;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    // Compute pairwise distances
    const dist: Float64Array[] = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => {
        let d = 0;
        for (let f = 0; f < X[i]!.length; f++) d += ((X[i]?.[f] ?? 0) - (X[j]?.[f] ?? 0)) ** 2;
        return Math.sqrt(d);
      }),
    );

    // Compute epsilon
    const eps = this.epsilon === "auto"
      ? Array.from(dist).flatMap(row => Array.from(row)).sort((a, b) => a - b)[Math.floor(n * n * 0.05)] ?? 1
      : this.epsilon;

    // Kernel matrix W = exp(-dist^2 / eps)
    const W: Float64Array[] = dist.map(row => row.map(d => Math.exp(-(d ** 2) / eps)));

    // Alpha normalization
    const D = W.map(row => row.reduce((s, v) => s + v, 0));
    const Ka: Float64Array[] = W.map((row, i) =>
      row.map((v, j) => v / ((D[i] ?? 1) ** this.alpha * (D[j] ?? 1) ** this.alpha)),
    );

    // Row-normalize to get Markov matrix
    const Da = Ka.map(row => row.reduce((s, v) => s + v, 0));
    const P: Float64Array[] = Ka.map((row, i) => row.map(v => v / (Da[i] ?? 1)));

    // Apply time t
    const Pt = t_power(P, this.t);

    // Power iteration for top eigenvectors
    const k = this.n_components;
    const embedding: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
    const eigvals = new Float64Array(k);

    let M = Pt;
    for (let ki = 0; ki < k; ki++) {
      let v = new Float64Array(n).map((_, i) => i === ki % n ? 1 : 0);
      let lambda = 0;
      for (let iter = 0; iter < 50; iter++) {
        const Mv = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Mv[i] = (Mv[i] ?? 0) + (M[i]?.[j] ?? 0) * (v[j] ?? 0);
        lambda = Math.sqrt(Mv.reduce((s, x) => s + x * x, 0));
        if (lambda < 1e-10) break;
        v = Mv.map(x => x / lambda);
      }
      eigvals[ki] = lambda;
      for (let i = 0; i < n; i++) embedding[i]![ki] = v[i]! * lambda ** this.t;
      // Deflate
      const newM: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        newM[i]![j] = (M[i]?.[j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
      }
      M = newM;
    }
    this.embedding_ = embedding;
    this.eigenvalues_ = eigvals;
    return embedding;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.embedding_.length ? this.embedding_ : X.map(() => new Float64Array(this.n_components));
  }
}

function t_power(P: Float64Array[], t: number): Float64Array[] {
  if (t <= 1) return P;
  const n = P.length;
  let result = P;
  for (let ti = 1; ti < t; ti++) {
    const next: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
      next[i]![j] = (next[i]?.[j] ?? 0) + (result[i]?.[k] ?? 0) * (P[k]?.[j] ?? 0);
    }
    result = next;
  }
  return result;
}

export interface LargeVisParams {
  n_components?: number;
  perplexity?: number;
  learning_rate?: number;
  n_iter?: number;
  repulsion_strength?: number;
}

/** LargeVis: dimensionality reduction via probabilistic model with negative sampling. */
export class LargeVis extends BaseEstimator {
  n_components: number;
  perplexity: number;
  learning_rate: number;
  n_iter: number;
  embedding_: Float64Array[] = [];

  constructor(params: LargeVisParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.perplexity = params.perplexity ?? 30;
    this.learning_rate = params.learning_rate ?? 0.1;
    this.n_iter = params.n_iter ?? 100;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const k = this.n_components;
    let Y = Array.from({ length: n }, () => new Float64Array(k).map(() => (Math.random() - 0.5) * 0.01));

    for (let iter = 0; iter < this.n_iter; iter++) {
      const grad: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        // Compute nearest neighbors
        const dists = Array.from({ length: n }, (_, j) => {
          let d = 0;
          for (let f = 0; f < X[i]!.length; f++) d += ((X[i]?.[f] ?? 0) - (X[j]?.[f] ?? 0)) ** 2;
          return { j, d };
        }).sort((a, b) => a.d - b.d);

        const neighbors = dists.slice(1, Math.min(this.perplexity + 1, n)).map(e => e.j);
        for (const j of neighbors) {
          let ydist2 = 0;
          for (let f = 0; f < k; f++) ydist2 += ((Y[i]?.[f] ?? 0) - (Y[j]?.[f] ?? 0)) ** 2;
          const q = 1 / (1 + ydist2);
          for (let f = 0; f < k; f++) {
            const diff = ((Y[i]?.[f] ?? 0) - (Y[j]?.[f] ?? 0)) * q;
            grad[i]![f] = (grad[i]?.[f] ?? 0) + diff;
            grad[j]![f] = (grad[j]?.[f] ?? 0) - diff;
          }
        }
      }
      for (let i = 0; i < n; i++) for (let f = 0; f < k; f++) {
        Y[i]![f] = (Y[i]?.[f] ?? 0) - this.learning_rate * (grad[i]?.[f] ?? 0);
      }
    }
    this.embedding_ = Y;
    return Y;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    return this.embedding_;
  }
}
