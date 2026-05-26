/**
 * Linear algebra utilities: QR decomposition, eigendecomposition, matrix exponential, linear system solver
 */

export function qrDecomposition(A: Float64Array[]): {
  Q: Float64Array[];
  R: Float64Array[];
} {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  const Q: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n));
  const R: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));

  // Gram-Schmidt
  const cols: Float64Array[] = Array.from({ length: n }, (_, j) => {
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = (A[i] ?? new Float64Array(n))[j] ?? 0;
    return col;
  });

  const qCols: Float64Array[] = [];
  for (let j = 0; j < n; j++) {
    let v = Float64Array.from(cols[j] ?? new Float64Array(m));
    for (let k = 0; k < j; k++) {
      const qk = qCols[k] ?? new Float64Array(m);
      const r = dot(qk, cols[j] ?? new Float64Array(m));
      R[k]![j] = r;
      for (let i = 0; i < m; i++) v[i] -= r * (qk[i] ?? 0);
    }
    const norm = Math.sqrt(dot(v, v));
    R[j]![j] = norm;
    if (norm > 1e-10) {
      for (let i = 0; i < m; i++) v[i] /= norm;
    }
    qCols.push(v);
    for (let i = 0; i < m; i++) Q[i]![j] = v[i] ?? 0;
  }
  return { Q, R };
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

export function solveLinearSystem(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  // Gaussian elimination with partial pivoting
  const aug: Float64Array[] = A.map((row, i) => {
    const r = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
    r[n] = b[i] ?? 0;
    return r;
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col]![col] ?? 0);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row]![col] ?? 0);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    const tmp = aug[col]!;
    aug[col] = aug[maxRow]!;
    aug[maxRow] = tmp;

    const pivot = aug[col]![col] ?? 0;
    if (Math.abs(pivot) < 1e-12) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = (aug[row]![col] ?? 0) / pivot;
      for (let j = col; j <= n; j++) {
        aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
      }
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i]![n] ?? 0;
    for (let j = i + 1; j < n; j++) sum -= (aug[i]![j] ?? 0) * (x[j] ?? 0);
    const diag = aug[i]![i] ?? 0;
    x[i] = Math.abs(diag) < 1e-12 ? 0 : sum / diag;
  }
  return x;
}

export function choleskyDecompositionExt(A: Float64Array[]): Float64Array[] {
  const n = A.length;
  const L: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i]![j] ?? 0;
      for (let k = 0; k < j; k++) sum -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      if (i === j) {
        L[i]![j] = Math.sqrt(Math.max(0, sum));
      } else {
        const diag = L[j]![j] ?? 1;
        L[i]![j] = diag < 1e-12 ? 0 : sum / diag;
      }
    }
  }
  return L;
}

export function matrixExp(A: Float64Array[], terms = 20): Float64Array[] {
  const n = A.length;
  const identity = (): Float64Array[] =>
    Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      row[i] = 1;
      return row;
    });

  const matMul = (X: Float64Array[], Y: Float64Array[]): Float64Array[] =>
    Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => {
        let s = 0;
        for (let k = 0; k < n; k++) s += (X[i]![k] ?? 0) * (Y[k]![j] ?? 0);
        return s;
      })
    );

  const matScale = (X: Float64Array[], s: number): Float64Array[] =>
    X.map((row) => row.map((v) => v * s) as unknown as Float64Array);

  const matAdd = (X: Float64Array[], Y: Float64Array[]): Float64Array[] =>
    X.map((row, i) => row.map((v, j) => v + (Y[i]![j] ?? 0)) as unknown as Float64Array);

  let result = identity();
  let term = identity();
  for (let k = 1; k < terms; k++) {
    term = matScale(matMul(term, A), 1 / k);
    result = matAdd(result, term);
  }
  return result;
}

export function eigenDecomposition2x2(A: Float64Array[]): {
  eigenvalues: Float64Array;
  eigenvectors: Float64Array[];
} {
  if (A.length !== 2) throw new Error("Only 2x2 supported");
  const a = A[0]![0] ?? 0, b = A[0]![1] ?? 0;
  const c = A[1]![0] ?? 0, d = A[1]![1] ?? 0;
  const trace = a + d;
  const det = a * d - b * c;
  const disc = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
  const l1 = trace / 2 + disc;
  const l2 = trace / 2 - disc;
  const eigenvalues = new Float64Array([l1, l2]);
  const v1 = Math.abs(b) > 1e-10
    ? new Float64Array([b, l1 - a])
    : new Float64Array([1, 0]);
  const v2 = Math.abs(b) > 1e-10
    ? new Float64Array([b, l2 - a])
    : new Float64Array([0, 1]);
  const n1 = Math.sqrt((v1[0] ?? 0) ** 2 + (v1[1] ?? 0) ** 2) || 1;
  const n2 = Math.sqrt((v2[0] ?? 0) ** 2 + (v2[1] ?? 0) ** 2) || 1;
  return {
    eigenvalues,
    eigenvectors: [
      new Float64Array([(v1[0] ?? 0) / n1, (v1[1] ?? 0) / n1]),
      new Float64Array([(v2[0] ?? 0) / n2, (v2[1] ?? 0) / n2]),
    ],
  };
}
