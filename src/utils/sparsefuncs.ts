/**
 * Sparse matrix utility functions.
 * Mirrors sklearn.utils.sparsefuncs.
 */

export interface SparseMatrix {
  data: Float64Array;
  indices: Int32Array;
  indptr: Int32Array;
  shape: [number, number];
}

/** Create a CSR sparse matrix from dense 2D array. */
export function denseToCsr(X: Float64Array[]): SparseMatrix {
  const nRows = X.length;
  const nCols = X[0]?.length ?? 0;
  const data: number[] = [];
  const indices: number[] = [];
  const indptr: number[] = [0];
  for (let i = 0; i < nRows; i++) {
    const row = X[i]!;
    for (let j = 0; j < nCols; j++) {
      const v = row[j] ?? 0;
      if (v !== 0) { data.push(v); indices.push(j); }
    }
    indptr.push(data.length);
  }
  return {
    data: new Float64Array(data),
    indices: new Int32Array(indices),
    indptr: new Int32Array(indptr),
    shape: [nRows, nCols],
  };
}

/** Convert CSR sparse matrix back to dense. */
export function csrToDense(sp: SparseMatrix): Float64Array[] {
  const [nRows, nCols] = sp.shape;
  const result: Float64Array[] = Array.from({ length: nRows }, () => new Float64Array(nCols));
  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      result[i]![sp.indices[k] ?? 0] = sp.data[k] ?? 0;
    }
  }
  return result;
}

/** Compute mean of each column in a CSR sparse matrix. */
export function meanVarianceAxis0(sp: SparseMatrix): { mean: Float64Array; variance: Float64Array } {
  const [nRows, nCols] = sp.shape;
  const mean = new Float64Array(nCols);
  const variance = new Float64Array(nCols);
  const count = new Float64Array(nCols);

  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sp.indices[k] ?? 0;
      mean[j]! += sp.data[k] ?? 0;
      count[j]! += 1;
    }
  }
  for (let j = 0; j < nCols; j++) mean[j]! /= nRows;

  // Variance: sum of (x - mean)^2 / n; sparse entries with value 0 contribute mean^2
  for (let j = 0; j < nCols; j++) {
    const m = mean[j] ?? 0;
    variance[j]! = m * m * (nRows - (count[j] ?? 0));
  }
  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sp.indices[k] ?? 0;
      const diff = (sp.data[k] ?? 0) - (mean[j] ?? 0);
      variance[j]! += diff * diff;
    }
  }
  for (let j = 0; j < nCols; j++) variance[j]! /= nRows;

  return { mean, variance };
}

/** Inplace row-wise scaling: X[i] *= scales[i] */
export function inplaceRowScale(sp: SparseMatrix, scales: Float64Array): void {
  for (let i = 0; i < sp.shape[0]; i++) {
    const s = scales[i] ?? 1;
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) sp.data[k]! *= s;
  }
}

/** Inplace column-wise scaling: X[:, j] *= scales[j] */
export function inplaceColumnScale(sp: SparseMatrix, scales: Float64Array): void {
  for (let i = 0; i < sp.shape[0]; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sp.indices[k] ?? 0;
      sp.data[k]! *= scales[j] ?? 1;
    }
  }
}

/** Compute min and max of each column in a CSR sparse matrix. */
export function minMaxAxis(sp: SparseMatrix, axis: 0 | 1 = 0): { min: Float64Array; max: Float64Array } {
  const [nRows, nCols] = sp.shape;
  const size = axis === 0 ? nCols : nRows;
  const min = new Float64Array(size).fill(Number.POSITIVE_INFINITY);
  const max = new Float64Array(size).fill(Number.NEGATIVE_INFINITY);
  const hasExplicit = new Uint8Array(size);

  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sp.indices[k] ?? 0;
      const idx = axis === 0 ? j : i;
      const v = sp.data[k] ?? 0;
      if (v < (min[idx] ?? Number.POSITIVE_INFINITY)) min[idx]! = v;
      if (v > (max[idx] ?? Number.NEGATIVE_INFINITY)) max[idx]! = v;
      hasExplicit[idx] = 1;
    }
  }
  // Implicit zeros must be considered
  for (let idx = 0; idx < size; idx++) {
    if (!(hasExplicit[idx] ?? 0)) { min[idx] = 0; max[idx] = 0; }
    else {
      if ((min[idx] ?? 0) > 0) min[idx] = 0;
      if ((max[idx] ?? 0) < 0) max[idx] = 0;
    }
  }
  return { min, max };
}

/** Compute L1/L2 norms of each row or column. */
export function normAxis(sp: SparseMatrix, axis: 0 | 1 = 1, norm: 1 | 2 = 2): Float64Array {
  const [nRows, nCols] = sp.shape;
  const size = axis === 1 ? nRows : nCols;
  const out = new Float64Array(size);
  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = sp.indices[k] ?? 0;
      const idx = axis === 1 ? i : j;
      const v = sp.data[k] ?? 0;
      out[idx]! += norm === 1 ? Math.abs(v) : v * v;
    }
  }
  if (norm === 2) for (let i = 0; i < size; i++) out[i]! = Math.sqrt(out[i] ?? 0);
  return out;
}

/** CSR matrix-vector product: result = sp @ v */
export function csrMatVec(sp: SparseMatrix, v: Float64Array): Float64Array {
  const [nRows] = sp.shape;
  const result = new Float64Array(nRows);
  for (let i = 0; i < nRows; i++) {
    const start = sp.indptr[i] ?? 0;
    const end = sp.indptr[i + 1] ?? 0;
    let s = 0;
    for (let k = start; k < end; k++) s += (sp.data[k] ?? 0) * (v[sp.indices[k] ?? 0] ?? 0);
    result[i] = s;
  }
  return result;
}
