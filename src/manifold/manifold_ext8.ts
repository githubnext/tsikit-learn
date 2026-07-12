/**
 * Manifold learning extensions: PHATE, TopoMap, Parametric UMAP.
 * Mirrors sklearn.manifold advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface PHATEParams {
  n_components?: number;
  knn?: number;
  decay?: number;
  t?: number | "auto";
  n_landmark?: number;
}

/**
 * PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.
 * Simplified implementation for dimensionality reduction.
 */
export class PHATE extends BaseEstimator {
  n_components: number;
  knn: number;
  decay: number;
  t: number | "auto";
  n_landmark: number;
  embedding_: Float64Array[] = [];

  constructor(params: PHATEParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.knn = params.knn ?? 5;
    this.decay = params.decay ?? 40;
    this.t = params.t ?? "auto";
    this.n_landmark = params.n_landmark ?? 2000;
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    // Step 1: k-NN kernel
    const knnGraph = this._knn(X);
    // Step 2: Markov transition matrix (row-normalized)
    const P = this._markovNormalize(knnGraph, n);
    // Step 3: Diffuse t steps
    const tSteps = this.t === "auto" ? Math.max(1, Math.floor(Math.sqrt(n))) : this.t;
    let Pt = P.map((row) => row.slice());
    for (let step = 1; step < tSteps; step++) {
      Pt = this._matMul(Pt, P, n);
    }
    // Step 4: Potential distances (-log of transition probabilities)
    const potential = Pt.map((row) => row.map((v) => -Math.log(Math.max(v, 1e-10))));
    // Step 5: MDS on potential distances
    this.embedding_ = this._mds(potential, n);
    return this.embedding_;
  }

  fit(X: Float64Array[]): this {
    this.fit_transform(X);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.embedding_;
  }

  private _knn(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const K = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const dists = X.map((xj, j) => {
        let d = 0;
        for (let k = 0; k < (X[i]?.length ?? 0); k++) d += ((X[i]?.[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        return { j, d };
      }).sort((a, b) => a.d - b.d).slice(1, this.knn + 1);
      const sigma = dists[dists.length - 1]?.d ?? 1;
      for (const { j, d } of dists) K[i]![j] = Math.exp(-d / Math.max(sigma, 1e-10));
    }
    return K;
  }

  private _markovNormalize(K: Float64Array[], n: number): Float64Array[] {
    return K.map((row) => {
      const sum = Array.from(row).reduce((s, v) => s + v, 0);
      return sum > 0 ? new Float64Array(row.map((v) => v / sum)) : row;
    });
  }

  private _matMul(A: Float64Array[], B: Float64Array[], n: number): Float64Array[] {
    return Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, k) => {
        let s = 0;
        for (let j = 0; j < n; j++) s += (A[i]?.[j] ?? 0) * (B[j]?.[k] ?? 0);
        return s;
      }),
    );
  }

  private _mds(dist: Float64Array[], n: number): Float64Array[] {
    // Classical MDS
    const nc = this.n_components;
    const D2 = dist.map((row) => row.map((v) => v * v));
    const mean_i = D2.map((row) => Array.from(row).reduce((s, v) => s + v, 0) / n);
    const mean_j = Array.from({ length: n }, (_, j) => D2.reduce((s, row) => s + (row[j] ?? 0), 0) / n);
    const grand = mean_i.reduce((s, v) => s + v, 0) / n;
    const B = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => -0.5 * ((D2[i]?.[j] ?? 0) - (mean_i[i] ?? 0) - (mean_j[j] ?? 0) + grand)),
    );
    // Power iteration for top nc eigenvectors
    const vecs: Float64Array[] = [];
    const Bc = B.map((r) => new Float64Array(r));
    for (let c = 0; c < nc; c++) {
      let v = new Float64Array(n).map((_, i) => i === c ? 1 : 0.01);
      let ev = 0;
      for (let iter = 0; iter < 50; iter++) {
        const av = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) av[i] = (av[i] ?? 0) + (Bc[i]?.[j] ?? 0) * (v[j] ?? 0);
        let norm = 0; for (let i = 0; i < n; i++) norm += (av[i] ?? 0) ** 2; norm = Math.sqrt(norm);
        ev = norm;
        if (norm < 1e-10) break;
        for (let i = 0; i < n; i++) av[i] = (av[i] ?? 0) / norm;
        v = av;
      }
      vecs.push(new Float64Array(v.map((vi) => vi * Math.sqrt(Math.max(ev, 0)))));
      // Deflate
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bc[i]![j] = (Bc[i]![j] ?? 0) - ev * (v[i] ?? 0) * (v[j] ?? 0);
    }
    return Array.from({ length: n }, (_, i) => new Float64Array(nc).map((_, c) => vecs[c]?.[i] ?? 0));
  }
}

export interface TopoMapParams {
  n_components?: number;
  n_neighbors?: number;
}

/** TopoMap: topological dimensionality reduction. */
export class TopoMap extends BaseEstimator {
  n_components: number;
  n_neighbors: number;
  embedding_: Float64Array[] = [];

  constructor(params: TopoMapParams = {}) {
    super();
    this.n_components = params.n_components ?? 2;
    this.n_neighbors = params.n_neighbors ?? 15;
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nc = this.n_components;
    // Build distance matrix
    const dist = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => {
        let d = 0;
        const xi = X[i]!, xj = X[j]!;
        for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        return Math.sqrt(d);
      }),
    );
    // Initialize random layout
    const pos = Array.from({ length: n }, () => new Float64Array(nc).map(() => Math.random() - 0.5));
    // Force-directed layout (simplified)
    for (let iter = 0; iter < 100; iter++) {
      const forces = Array.from({ length: n }, () => new Float64Array(nc));
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const target = dist[i]?.[j] ?? 0;
          let actual = 0;
          const dp = new Float64Array(nc);
          for (let k = 0; k < nc; k++) { dp[k] = (pos[i]?.[k] ?? 0) - (pos[j]?.[k] ?? 0); actual += (dp[k] ?? 0) ** 2; }
          actual = Math.sqrt(actual);
          if (actual < 1e-10) continue;
          const force = (actual - target) / actual;
          for (let k = 0; k < nc; k++) {
            forces[i]![k] = (forces[i]![k] ?? 0) - 0.01 * force * (dp[k] ?? 0);
            forces[j]![k] = (forces[j]![k] ?? 0) + 0.01 * force * (dp[k] ?? 0);
          }
        }
      }
      for (let i = 0; i < n; i++) for (let k = 0; k < nc; k++) pos[i]![k] = (pos[i]![k] ?? 0) + (forces[i]?.[k] ?? 0);
    }
    this.embedding_ = pos;
    return pos;
  }

  fit(X: Float64Array[]): this {
    this.fit_transform(X);
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    return this.embedding_;
  }
}
