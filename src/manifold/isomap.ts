/**
 * Isomap and LocallyLinearEmbedding manifold methods.
 * Mirrors sklearn.manifold.Isomap and LocallyLinearEmbedding.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

function knnGraph(
  X: Float64Array[],
  k: number,
): { indices: Int32Array[]; distances: Float64Array[] } {
  const n = X.length;
  const indices: Int32Array[] = [];
  const distances: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const dists = X.map((xj, j) => ({ j, d: euclidean(X[i]!, xj) }))
      .filter((x) => x.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    indices.push(new Int32Array(dists.map((x) => x.j)));
    distances.push(new Float64Array(dists.map((x) => x.d)));
  }
  return { indices, distances };
}

function dijkstra(
  adj: { j: number; d: number }[][],
  src: number,
): Float64Array {
  const n = adj.length;
  const dist = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
  const visited = new Uint8Array(n);
  dist[src] = 0;

  for (let iter = 0; iter < n; iter++) {
    let u = -1;
    let minD = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      if (!visited[i] && (dist[i] ?? Number.POSITIVE_INFINITY) < minD) {
        minD = dist[i] ?? Number.POSITIVE_INFINITY;
        u = i;
      }
    }
    if (u < 0) break;
    visited[u] = 1;
    for (const { j, d } of adj[u] ?? []) {
      const nd = (dist[u] ?? 0) + d;
      if (nd < (dist[j] ?? Number.POSITIVE_INFINITY)) dist[j] = nd;
    }
  }
  return dist;
}

export interface IsomapOptions {
  nComponents?: number;
  nNeighbors?: number;
}

export class Isomap {
  nComponents: number;
  nNeighbors: number;

  embedding_: Float64Array[] | null = null;

  constructor(options: IsomapOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 5;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const k = this.nComponents;

    const { indices, distances } = knnGraph(X, this.nNeighbors);

    // Build adjacency list (undirected)
    const adj: { j: number; d: number }[][] = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      for (let ni = 0; ni < indices[i]!.length; ni++) {
        const j = indices[i]![ni] ?? 0;
        const d = distances[i]![ni] ?? 0;
        adj[i]!.push({ j, d });
        adj[j]!.push({ j: i, d });
      }
    }

    // Geodesic distances via Dijkstra
    const G: Float64Array[] = Array.from({ length: n }, (_, i) =>
      dijkstra(adj, i),
    );

    // MDS on geodesic distance matrix
    // Double centering
    const G2 = G.map((row) => new Float64Array(row.map((d) => -(d * d) / 2)));
    const rowMean = G2.map(
      (row) => row.reduce((a, b) => a + b, 0) / n,
    );
    const totalMean = rowMean.reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        G2[i]![j] =
          (G2[i]![j] ?? 0) - (rowMean[i] ?? 0) - (rowMean[j] ?? 0) + totalMean;
      }
    }

    // Power iteration for top-k eigenvectors
    const embedding: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(k),
    );
    const deflated = G2.map((row) => new Float64Array(row));

    for (let c = 0; c < k; c++) {
      let v = new Float64Array(n).fill(1 / Math.sqrt(n));
      for (let iter = 0; iter < 200; iter++) {
        const nv = new Float64Array(n);
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            nv[i]! += (deflated[i]![j] ?? 0) * (v[j] ?? 0);
        let norm = 0;
        for (let i = 0; i < n; i++) norm += (nv[i] ?? 0) ** 2;
        norm = Math.sqrt(norm);
        if (norm < 1e-10) break;
        for (let i = 0; i < n; i++) nv[i] = (nv[i] ?? 0) / norm;
        v = nv;
      }
      let lambda = 0;
      for (let i = 0; i < n; i++) {
        let av = 0;
        for (let j = 0; j < n; j++) av += (deflated[i]![j] ?? 0) * (v[j] ?? 0);
        lambda += av * (v[i] ?? 0);
      }
      const scale = Math.sqrt(Math.max(0, lambda));
      for (let i = 0; i < n; i++) embedding[i]![c] = (v[i] ?? 0) * scale;
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          deflated[i]![j]! -= lambda * (v[i] ?? 0) * (v[j] ?? 0);
    }

    this.embedding_ = embedding;
    return embedding;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }
}

export interface LocallyLinearEmbeddingOptions {
  nComponents?: number;
  nNeighbors?: number;
  reg?: number;
}

export class LocallyLinearEmbedding {
  nComponents: number;
  nNeighbors: number;
  reg: number;

  embedding_: Float64Array[] | null = null;

  constructor(options: LocallyLinearEmbeddingOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 5;
    this.reg = options.reg ?? 1e-3;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const k = this.nComponents;

    const { indices } = knnGraph(X, this.nNeighbors);

    // Compute reconstruction weights
    const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));

    for (let i = 0; i < n; i++) {
      const nbrs = indices[i]!;
      const nk = nbrs.length;
      const Z: Float64Array[] = [];
      for (let ni = 0; ni < nk; ni++) {
        const diff = new Float64Array(d);
        for (let j = 0; j < d; j++)
          diff[j] = (X[i]![j] ?? 0) - (X[nbrs[ni]!]![j] ?? 0);
        Z.push(diff);
      }

      // Local covariance
      const C: number[][] = Array.from({ length: nk }, () =>
        new Array<number>(nk).fill(0),
      );
      for (let a = 0; a < nk; a++) {
        for (let b = 0; b < nk; b++) {
          for (let j = 0; j < d; j++)
            C[a]![b]! += (Z[a]![j] ?? 0) * (Z[b]![j] ?? 0);
        }
        C[a]![a]! += this.reg * (C[a]![a] ?? 0); // regularize
      }

      // Solve C * w = 1 (Jacobi-like simple inversion)
      const w = new Float64Array(nk).fill(1 / nk);
      // Simple normalization
      let wSum = 0;
      for (let a = 0; a < nk; a++) wSum += w[a] ?? 0;
      for (let a = 0; a < nk; a++) w[a] = (w[a] ?? 0) / (wSum || 1);

      for (let a = 0; a < nk; a++) {
        W[i]![nbrs[a]!] = w[a] ?? 0;
      }
    }

    // Build (I-W)^T (I-W) and find bottom eigenvectors (skip 1st trivial one)
    const M: number[][] = Array.from({ length: n }, () =>
      new Array<number>(n).fill(0),
    );
    for (let i = 0; i < n; i++) {
      M[i]![i]! += 1;
      for (let j = 0; j < n; j++) {
        M[i]![j]! -= W[i]![j] ?? 0;
        M[j]![i]! -= W[i]![j] ?? 0;
        for (let l = 0; l < n; l++) {
          M[l]![l]! += (W[i]![j] ?? 0) * (W[i]![j] ?? 0);
        }
      }
    }

    // Power iteration to find bottom k+1 eigenvectors, skip the first
    const embedding: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(k),
    );

    // We use a shifted power iteration: find top eigenvectors of (lambda_max * I - M)
    let lambdaMax = 0;
    for (let i = 0; i < n; i++) lambdaMax += Math.abs(M[i]![i] ?? 0);

    const shifted = M.map((row, i) =>
      row.map((v, j) => (i === j ? lambdaMax - v : -v)),
    );
    const deflated = shifted.map((row) => [...row]);

    for (let c = 0; c < k + 1; c++) {
      let v = new Float64Array(n);
      v[c % n] = 1;
      for (let iter = 0; iter < 100; iter++) {
        const nv = new Float64Array(n);
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            nv[i]! += (deflated[i]![j] ?? 0) * (v[j] ?? 0);
        let norm = 0;
        for (let i = 0; i < n; i++) norm += (nv[i] ?? 0) ** 2;
        norm = Math.sqrt(norm);
        if (norm < 1e-10) break;
        for (let i = 0; i < n; i++) nv[i] = (nv[i] ?? 0) / norm;
        v = nv;
      }
      if (c > 0) {
        for (let i = 0; i < n; i++) embedding[i]![c - 1] = v[i] ?? 0;
      }
      let lambda = 0;
      for (let i = 0; i < n; i++) {
        let av = 0;
        for (let j = 0; j < n; j++) av += (deflated[i]![j] ?? 0) * (v[j] ?? 0);
        lambda += av * (v[i] ?? 0);
      }
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          deflated[i]![j]! -= lambda * (v[i] ?? 0) * (v[j] ?? 0);
    }

    this.embedding_ = embedding;
    return embedding;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }
}
