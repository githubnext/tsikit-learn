/**
 * Graph utilities for neighbors: kneighbors_graph and radius_neighbors_graph.
 * Mirrors sklearn.neighbors.kneighbors_graph and radius_neighbors_graph.
 */

import { NotFittedError } from "../exceptions.js";

export interface SparseMatrix {
  data: Float64Array;
  indices: Int32Array;
  indptr: Int32Array;
  shape: [number, number];
}

export type GraphMode = "connectivity" | "distance";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    s += d * d;
  }
  return Math.sqrt(s);
}

/**
 * Build a CSR adjacency matrix from k-nearest neighbor relationships.
 */
export function neighborsGraph(
  X: Float64Array[],
  nNeighbors: number,
  mode: GraphMode = "connectivity",
  includesSelf = false,
): SparseMatrix {
  const n = X.length;
  const nnz = n * nNeighbors;
  const data = new Float64Array(nnz);
  const indices = new Int32Array(nnz);
  const indptr = new Int32Array(n + 1);

  for (let i = 0; i < n; i++) {
    const dists: Array<[number, number]> = [];
    for (let j = 0; j < n; j++) {
      if (!includesSelf && i === j) continue;
      dists.push([euclidean(X[i]!, X[j]!), j]);
    }
    dists.sort((a, b) => a[0] - b[0]);
    const neighbors = dists.slice(0, nNeighbors);
    const base = i * nNeighbors;
    for (let k = 0; k < neighbors.length; k++) {
      indices[base + k] = neighbors[k]![1];
      data[base + k] = mode === "connectivity" ? 1 : neighbors[k]![0];
    }
    indptr[i + 1] = (i + 1) * nNeighbors;
  }

  return { data, indices, indptr, shape: [n, n] };
}

/**
 * Build a CSR adjacency matrix from radius neighbors.
 */
export function radiusNeighborsGraph(
  X: Float64Array[],
  radius: number,
  mode: GraphMode = "connectivity",
  includesSelf = false,
): SparseMatrix {
  const n = X.length;
  const allIndices: number[][] = [];
  const allDists: number[][] = [];

  for (let i = 0; i < n; i++) {
    const idxList: number[] = [];
    const distList: number[] = [];
    for (let j = 0; j < n; j++) {
      if (!includesSelf && i === j) continue;
      const d = euclidean(X[i]!, X[j]!);
      if (d <= radius) {
        idxList.push(j);
        distList.push(d);
      }
    }
    allIndices.push(idxList);
    allDists.push(distList);
  }

  const nnz = allIndices.reduce((s, row) => s + row.length, 0);
  const data = new Float64Array(nnz);
  const indices = new Int32Array(nnz);
  const indptr = new Int32Array(n + 1);

  let ptr = 0;
  for (let i = 0; i < n; i++) {
    const idxList = allIndices[i]!;
    const distList = allDists[i]!;
    for (let k = 0; k < idxList.length; k++) {
      indices[ptr] = idxList[k]!;
      data[ptr] = mode === "connectivity" ? 1 : distList[k]!;
      ptr++;
    }
    indptr[i + 1] = ptr;
  }

  return { data, indices, indptr, shape: [n, n] };
}

/** Dense adjacency matrix from sparse CSR */
export function sparseToDense(sparse: SparseMatrix): Float64Array[] {
  const [n] = sparse.shape;
  const dense = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const start = sparse.indptr[i] ?? 0;
    const end = sparse.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sparse.indices[k] ?? 0;
      dense[i]![j] = sparse.data[k] ?? 0;
    }
  }
  return dense;
}
