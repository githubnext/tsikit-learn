/**
 * Mathematical utilities for tsikit-learn.
 * Mirrors sklearn.utils.extmath.
 */

/** Compute the log of the logistic function element-wise. */
export function logLogistic(x: Float64Array): Float64Array {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i] ?? 0;
    result[i] =
      xi >= 0 ? -Math.log1p(Math.exp(-xi)) : xi - Math.log1p(Math.exp(xi));
  }
  return result;
}

/** Compute softmax values for each row of X. */
export function softmax(X: Float64Array[], copy = true): Float64Array[] {
  const result = copy ? X.map((row) => new Float64Array(row)) : X;
  for (const row of result) {
    const maxVal = Math.max(...row);
    let sum = 0;
    for (let j = 0; j < row.length; j++) {
      row[j] = Math.exp((row[j] ?? 0) - maxVal);
      sum += row[j] ?? 0;
    }
    for (let j = 0; j < row.length; j++) {
      row[j] = (row[j] ?? 0) / sum;
    }
  }
  return result;
}

/** Compute row norms of a matrix. */
export function rowNorms(X: Float64Array[], squared = false): Float64Array {
  const norms = new Float64Array(X.length);
  for (let i = 0; i < X.length; i++) {
    const row = X[i] ?? new Float64Array(0);
    let norm2 = 0;
    for (const v of row) norm2 += v * v;
    norms[i] = squared ? norm2 : Math.sqrt(norm2);
  }
  return norms;
}

/** Safe sparse dot (dense version). Computes X @ y. */
export function safeDot(X: Float64Array[], y: Float64Array): Float64Array {
  const n = X.length;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = X[i] ?? new Float64Array(0);
    let dot = 0;
    for (let j = 0; j < row.length; j++) {
      dot += (row[j] ?? 0) * (y[j] ?? 0);
    }
    result[i] = dot;
  }
  return result;
}

/** Matrix transpose. */
export function transpose(X: Float64Array[]): Float64Array[] {
  if (X.length === 0) return [];
  const nRows = X.length;
  const nCols = (X[0] ?? new Float64Array(0)).length;
  const result: Float64Array[] = Array.from(
    { length: nCols },
    () => new Float64Array(nRows),
  );
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      (result[j] ?? new Float64Array(0))[i] =
        (X[i] ?? new Float64Array(0))[j] ?? 0;
    }
  }
  return result;
}

/** Matrix-matrix multiply: A @ B. */
export function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  if (A.length === 0 || B.length === 0) return [];
  const nRows = A.length;
  const nCols = (B[0] ?? new Float64Array(0)).length;
  const nInner = B.length;
  const result: Float64Array[] = Array.from(
    { length: nRows },
    () => new Float64Array(nCols),
  );
  for (let i = 0; i < nRows; i++) {
    for (let k = 0; k < nInner; k++) {
      const aik = (A[i] ?? new Float64Array(0))[k] ?? 0;
      if (aik === 0) continue;
      for (let j = 0; j < nCols; j++) {
        const resultRow = result[i] ?? new Float64Array(0);
        resultRow[j] =
          (resultRow[j] ?? 0) + aik * ((B[k] ?? new Float64Array(0))[j] ?? 0);
      }
    }
  }
  return result;
}

/**
 * Solve a lower triangular system Lx = b using forward substitution.
 */
export function forwardSubstitution(
  L: Float64Array[],
  b: Float64Array,
): Float64Array {
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i] ?? 0;
    for (let j = 0; j < i; j++) {
      sum -= ((L[i] ?? new Float64Array(0))[j] ?? 0) * (x[j] ?? 0);
    }
    x[i] = sum / ((L[i] ?? new Float64Array(0))[i] ?? 1);
  }
  return x;
}

/**
 * Solve an upper triangular system Ux = b using back substitution.
 */
export function backSubstitution(
  U: Float64Array[],
  b: Float64Array,
): Float64Array {
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i] ?? 0;
    for (let j = i + 1; j < n; j++) {
      sum -= ((U[i] ?? new Float64Array(0))[j] ?? 0) * (x[j] ?? 0);
    }
    x[i] = sum / ((U[i] ?? new Float64Array(0))[i] ?? 1);
  }
  return x;
}

/**
 * Cholesky decomposition of a symmetric positive definite matrix.
 * Returns L such that A = L @ L.T
 */
export function cholesky(A: Float64Array[]): Float64Array[] {
  const n = A.length;
  const L: Float64Array[] = Array.from(
    { length: n },
    () => new Float64Array(n),
  );
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = (A[i] ?? new Float64Array(0))[j] ?? 0;
      for (let k = 0; k < j; k++) {
        sum -=
          ((L[i] ?? new Float64Array(0))[k] ?? 0) *
          ((L[j] ?? new Float64Array(0))[k] ?? 0);
      }
      if (i === j) {
        (L[i] ?? new Float64Array(0))[j] = Math.sqrt(Math.max(sum, 0));
      } else {
        const ljj = (L[j] ?? new Float64Array(0))[j] ?? 1;
        (L[i] ?? new Float64Array(0))[j] = ljj !== 0 ? sum / ljj : 0;
      }
    }
  }
  return L;
}

/**
 * Solve the linear system Ax = b using Cholesky decomposition.
 * A must be symmetric positive definite.
 */
export function choleskyLinsolve(
  A: Float64Array[],
  b: Float64Array,
): Float64Array {
  const L = cholesky(A);
  const y = forwardSubstitution(L, b);
  const Lt = transpose(L);
  return backSubstitution(Lt, y);
}

/** Compute the Euclidean distance between two vectors. */
export function euclideanDistance(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/** Add identity * alpha to a matrix (in-place). */
export function addDiagonal(A: Float64Array[], alpha: number): Float64Array[] {
  for (let i = 0; i < A.length; i++) {
    (A[i] ?? new Float64Array(0))[i] =
      ((A[i] ?? new Float64Array(0))[i] ?? 0) + alpha;
  }
  return A;
}

/** Compute X.T @ X (Gram matrix). */
export function gramMatrix(X: Float64Array[]): Float64Array[] {
  const Xt = transpose(X);
  return matMul(Xt, X);
}

/** Compute X.T @ y. */
export function xtDotY(X: Float64Array[], y: Float64Array): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const result = new Float64Array(p);
  for (let i = 0; i < X.length; i++) {
    const yi = y[i] ?? 0;
    const row = X[i] ?? new Float64Array(0);
    for (let j = 0; j < p; j++) {
      result[j] = (result[j] ?? 0) + (row[j] ?? 0) * yi;
    }
  }
  return result;
}
