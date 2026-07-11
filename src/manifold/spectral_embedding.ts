/**
 * SpectralEmbedding for manifold learning.
 * Mirrors sklearn.manifold.SpectralEmbedding.
 */

import { NotFittedError } from "../exceptions.js";

export interface SpectralEmbeddingOptions {
  nComponents?: number;
  gamma?: number;
  randomState?: number;
  nNeighbors?: number;
}

function rbfAffinity(X: Float64Array[], gamma: number): Float64Array[] {
  const n = X.length;
  return X.map((xi, i) =>
    Float64Array.from(X, (xj, j) => {
      if (i === j) return 0;
      let d = 0;
      for (let k = 0; k < xi.length; k++)
        d += ((xi[k] ?? 0) - ((xj as Float64Array)[k] ?? 0)) ** 2;
      return Math.exp(-gamma * d);
    }),
  );
}

function symmetricNormLaplacian(W: Float64Array[]): Float64Array[] {
  const n = W.length;
  const D = W.map((row) => row.reduce((s, v) => s + v, 0));
  const Dinvhalf = D.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  return W.map((row, i) =>
    Float64Array.from(
      row,
      (w, j) => (Dinvhalf[i] ?? 0) * w * (Dinvhalf[j] ?? 0),
    ),
  );
}

function powerIterEigenvecs(
  L: Float64Array[],
  k: number,
  maxIter = 500,
): Float64Array[] {
  const n = L.length;
  let seed = 42;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };

  const vecs: Float64Array[] = Array.from({ length: k }, () =>
    Float64Array.from({ length: n }, () => rand() - 0.5),
  );

  for (let iter = 0; iter < maxIter; iter++) {
    for (let col = 0; col < k; col++) {
      const v = vecs[col] as Float64Array;
      const Lv = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const row = L[i] as Float64Array;
        let s = 0;
        for (let j = 0; j < n; j++) s += (row[j] ?? 0) * (v[j] ?? 0);
        Lv[i]! = s;
      }
      for (let prev = 0; prev < col; prev++) {
        const u = vecs[prev] as Float64Array;
        let dot = 0;
        for (let i = 0; i < n; i++) dot += (Lv[i] ?? 0) * (u[i] ?? 0);
        for (let i = 0; i < n; i++) Lv[i]! -= dot * (u[i] ?? 0);
      }
      let norm = 0;
      for (let i = 0; i < n; i++) norm += (Lv[i] ?? 0) ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) Lv[i]! /= norm;
      vecs[col] = Lv;
    }
  }
  return vecs;
}

export class SpectralEmbedding {
  nComponents: number;
  gamma: number;
  randomState: number;

  embedding_: Float64Array[] | null = null;
  affinityMatrix_: Float64Array[] | null = null;

  constructor(opts: SpectralEmbeddingOptions = {}) {
    this.nComponents = opts.nComponents ?? 2;
    this.gamma = opts.gamma ?? 1.0;
    this.randomState = opts.randomState ?? 42;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const W = rbfAffinity(X, this.gamma);
    this.affinityMatrix_ = W;
    const L = symmetricNormLaplacian(W);
    const vecs = powerIterEigenvecs(L, this.nComponents + 1);
    // Skip the first eigenvector (constant), use the next nComponents
    const embedding: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(this.nComponents);
      for (let c = 0; c < this.nComponents; c++) {
        row[c]! = (vecs[c + 1] as Float64Array)[i] ?? 0;
      }
      return row;
    });
    this.embedding_ = embedding;
    return embedding;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }
}
