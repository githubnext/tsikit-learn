/**
 * Fast sparse matrix operations.
 * Mirrors scikit-learn's utils.sparsefuncs_fast (CSR/CSC operations).
 */

/** CSR (Compressed Sparse Row) matrix representation */
export interface CSRMatrix {
  data: Float64Array;
  indices: Int32Array; // column indices
  indptr: Int32Array; // row pointers
  shape: [number, number];
}

/** CSC (Compressed Sparse Column) matrix representation */
export interface CSCMatrix {
  data: Float64Array;
  indices: Int32Array; // row indices
  indptr: Int32Array; // column pointers
  shape: [number, number];
}

/** Create a CSR matrix from a dense array */
export function denseToCSR(X: Float64Array[], tol = 0): CSRMatrix {
  const nRows = X.length;
  const nCols = X[0]?.length ?? 0;
  const data: number[] = [];
  const indices: number[] = [];
  const indptr: number[] = [0];

  for (const row of X) {
    for (let j = 0; j < nCols; j++) {
      const v = row[j] ?? 0;
      if (Math.abs(v) > tol) {
        data.push(v);
        indices.push(j);
      }
    }
    indptr.push(data.length);
  }

  return {
    data: new Float64Array(data),
    indices: Int32Array.from(indices),
    indptr: Int32Array.from(indptr),
    shape: [nRows, nCols],
  };
}

/** Convert a CSR matrix back to dense */
export function csrToDense(csr: CSRMatrix): Float64Array[] {
  const [nRows, nCols] = csr.shape;
  return Array.from({ length: nRows }, (_, i) => {
    const row = new Float64Array(nCols);
    for (let k = csr.indptr[i]!; k < csr.indptr[i + 1]!; k++) {
      row[csr.indices[k]!] = csr.data[k]!;
    }
    return row;
  });
}

/** Compute column means of a CSR matrix */
export function csrColumnMeans(csr: CSRMatrix): Float64Array {
  const [nRows, nCols] = csr.shape;
  const means = new Float64Array(nCols);
  for (let i = 0; i < csr.data.length; i++) {
    means[csr.indices[i]!] = (means[csr.indices[i]!] ?? 0) + (csr.data[i] ?? 0);
  }
  for (let j = 0; j < nCols; j++) means[j] = (means[j] ?? 0) / nRows;
  return means;
}

/** Compute column variances of a CSR matrix */
export function csrColumnVariances(csr: CSRMatrix): Float64Array {
  const [nRows, nCols] = csr.shape;
  const means = csrColumnMeans(csr);
  const vars = new Float64Array(nCols);

  // Add contribution from non-zero elements
  for (let i = 0; i < csr.data.length; i++) {
    const j = csr.indices[i]!;
    const v = (csr.data[i] ?? 0) - (means[j] ?? 0);
    vars[j] = (vars[j] ?? 0) + v * v / nRows;
  }

  // Add contribution from zero elements (x - mean)^2
  const nnzPerCol = new Int32Array(nCols);
  for (let i = 0; i < csr.indices.length; i++) {
    nnzPerCol[csr.indices[i]!] = (nnzPerCol[csr.indices[i]!] ?? 0) + 1;
  }
  for (let j = 0; j < nCols; j++) {
    const nZero = nRows - (nnzPerCol[j] ?? 0);
    vars[j] = (vars[j] ?? 0) + nZero * (means[j] ?? 0) ** 2 / nRows;
  }

  return vars;
}

/** CSR matrix-vector product */
export function csrMatVec(csr: CSRMatrix, v: Float64Array): Float64Array {
  const [nRows] = csr.shape;
  const result = new Float64Array(nRows);
  for (let i = 0; i < nRows; i++) {
    let s = 0;
    for (let k = csr.indptr[i]!; k < csr.indptr[i + 1]!; k++) {
      s += (csr.data[k] ?? 0) * (v[csr.indices[k]!] ?? 0);
    }
    result[i] = s;
  }
  return result;
}

/** Inplace row scaling of a CSR matrix */
export function csrInplaceRowScale(csr: CSRMatrix, scale: Float64Array): void {
  const [nRows] = csr.shape;
  for (let i = 0; i < nRows; i++) {
    const s = scale[i] ?? 1;
    for (let k = csr.indptr[i]!; k < csr.indptr[i + 1]!; k++) {
      csr.data[k] = (csr.data[k] ?? 0) * s;
    }
  }
}
