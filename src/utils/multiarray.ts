/**
 * Multi-dimensional array utilities.
 * Provides ndarray-like 2D operations over Float64Array.
 * Mirrors numpy/sklearn internal array utilities.
 */

/** A 2D array backed by a flat Float64Array. */
export class NDArray2D {
  readonly data: Float64Array;
  readonly rows: number;
  readonly cols: number;

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j] ?? 0;
  }

  set(i: number, j: number, val: number): void {
    this.data[i * this.cols + j] = val;
  }

  row(i: number): Float64Array {
    return this.data.subarray(i * this.cols, (i + 1) * this.cols);
  }

  col(j: number): Float64Array {
    const out = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) out[i] = this.get(i, j);
    return out;
  }

  /** Transpose */
  T(): NDArray2D {
    const out = new NDArray2D(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        out.set(j, i, this.get(i, j));
      }
    }
    return out;
  }

  /** Matrix multiplication */
  matmul(other: NDArray2D): NDArray2D {
    if (this.cols !== other.rows)
      throw new Error(`Shape mismatch: ${this.cols} vs ${other.rows}`);
    const out = new NDArray2D(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let k = 0; k < this.cols; k++) {
        const aik = this.get(i, k);
        if (aik === 0) continue;
        for (let j = 0; j < other.cols; j++) {
          out.data[i * out.cols + j]! += aik * other.get(k, j);
        }
      }
    }
    return out;
  }

  /** Element-wise add */
  add(other: NDArray2D): NDArray2D {
    const out = new NDArray2D(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      out.data[i] = (this.data[i] ?? 0) + (other.data[i] ?? 0);
    }
    return out;
  }

  /** Element-wise subtract */
  sub(other: NDArray2D): NDArray2D {
    const out = new NDArray2D(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      out.data[i] = (this.data[i] ?? 0) - (other.data[i] ?? 0);
    }
    return out;
  }

  /** Scalar multiply */
  scale(s: number): NDArray2D {
    const out = new NDArray2D(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) out.data[i] = (this.data[i] ?? 0) * s;
    return out;
  }

  /** Row-wise sum */
  sumRows(): Float64Array {
    const out = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let s = 0;
      for (let j = 0; j < this.cols; j++) s += this.get(i, j);
      out[i] = s;
    }
    return out;
  }

  /** Column-wise sum */
  sumCols(): Float64Array {
    const out = new Float64Array(this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) out[j]! += this.get(i, j);
    }
    return out;
  }

  /** Column means */
  mean(): Float64Array {
    const s = this.sumCols();
    for (let j = 0; j < this.cols; j++) s[j]! /= this.rows;
    return s;
  }

  /** Frobenius norm */
  norm(): number {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += (this.data[i] ?? 0) ** 2;
    return Math.sqrt(s);
  }

  /** Create from array of rows */
  static fromRows(rows: Float64Array[]): NDArray2D {
    if (rows.length === 0) return new NDArray2D(0, 0);
    const cols = rows[0]?.length ?? 0;
    const out = new NDArray2D(rows.length, cols);
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i] ?? new Float64Array(cols);
      for (let j = 0; j < cols; j++) out.set(i, j, row[j] ?? 0);
    }
    return out;
  }

  /** Convert to array of rows */
  toRows(): Float64Array[] {
    const result: Float64Array[] = [];
    for (let i = 0; i < this.rows; i++) {
      result.push(new Float64Array(this.row(i)));
    }
    return result;
  }

  /** Create identity matrix */
  static eye(n: number): NDArray2D {
    const out = new NDArray2D(n, n);
    for (let i = 0; i < n; i++) out.set(i, i, 1);
    return out;
  }

  /** Create zero matrix */
  static zeros(rows: number, cols: number): NDArray2D {
    return new NDArray2D(rows, cols);
  }

  /** Reshape (only allowed if total elements match) */
  reshape(rows: number, cols: number): NDArray2D {
    if (rows * cols !== this.rows * this.cols)
      throw new Error("Cannot reshape: element count mismatch");
    return new NDArray2D(rows, cols, new Float64Array(this.data));
  }

  /** Compute covariance matrix of columns */
  cov(): NDArray2D {
    const p = this.cols;
    const n = this.rows;
    const mu = this.mean();
    const out = new NDArray2D(p, p);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        const dj = this.get(i, j) - (mu[j] ?? 0);
        for (let k = j; k < p; k++) {
          const dk = this.get(i, k) - (mu[k] ?? 0);
          out.data[j * p + k]! += dj * dk;
          if (k !== j) out.data[k * p + j]! += dj * dk;
        }
      }
    }
    const denom = n > 1 ? n - 1 : 1;
    for (let i = 0; i < out.data.length; i++) out.data[i]! /= denom;
    return out;
  }
}

/**
 * Broadcast add: each row of X gets vector v added.
 */
export function broadcastAdd(X: NDArray2D, v: Float64Array): NDArray2D {
  const out = new NDArray2D(X.rows, X.cols);
  for (let i = 0; i < X.rows; i++) {
    for (let j = 0; j < X.cols; j++) {
      out.set(i, j, X.get(i, j) + (v[j] ?? 0));
    }
  }
  return out;
}

/**
 * Pairwise squared Euclidean distances between rows of A and rows of B.
 * Returns an NDArray2D of shape [n_a, n_b].
 */
export function pairwiseSqDist(A: NDArray2D, B: NDArray2D): NDArray2D {
  const na = A.rows;
  const nb = B.rows;
  const out = new NDArray2D(na, nb);
  for (let i = 0; i < na; i++) {
    for (let j = 0; j < nb; j++) {
      let s = 0;
      for (let k = 0; k < A.cols; k++) {
        s += (A.get(i, k) - B.get(j, k)) ** 2;
      }
      out.set(i, j, s);
    }
  }
  return out;
}

/**
 * Compute column-wise standard deviation.
 */
export function colStd(X: NDArray2D, ddof = 0): Float64Array {
  const n = X.rows;
  const p = X.cols;
  const mu = X.mean();
  const out = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      out[j]! += (X.get(i, j) - (mu[j] ?? 0)) ** 2;
    }
  }
  const denom = n - ddof > 0 ? n - ddof : 1;
  for (let j = 0; j < p; j++) out[j] = Math.sqrt((out[j] ?? 0) / denom);
  return out;
}
