/**
 * Matrix operations utilities: Cholesky, LU decomposition, matrix inverse.
 */

export function choleskyDecomposition(A: Float64Array[]): Float64Array[] | null {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i]![j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      if (i === j) {
        if (s <= 0) return null; // Not positive definite
        L[i]![j] = Math.sqrt(s);
      } else {
        L[i]![j] = s / (L[j]![j] ?? 1);
      }
    }
  }
  return L;
}

export function luDecomposition(A: Float64Array[]): { L: Float64Array[]; U: Float64Array[]; P: Int32Array } {
  const n = A.length;
  const L = Array.from({ length: n }, (_, i) => new Float64Array(n).map((_, j) => (i === j ? 1 : 0)));
  const U = A.map(row => new Float64Array(row));
  const P = new Int32Array(n).map((_, i) => i);

  for (let k = 0; k < n; k++) {
    // Partial pivoting
    let maxVal = Math.abs(U[k]![k] ?? 0), maxRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(U[i]![k] ?? 0) > maxVal) { maxVal = Math.abs(U[i]![k] ?? 0); maxRow = i; }
    }
    if (maxRow !== k) {
      [U[k], U[maxRow]] = [U[maxRow]!, U[k]!];
      [P[k], P[maxRow]] = [P[maxRow]!, P[k]!];
      for (let j = 0; j < k; j++) { const tmp = L[k]![j]!; L[k]![j] = L[maxRow]![j]!; L[maxRow]![j] = tmp; }
    }
    for (let i = k + 1; i < n; i++) {
      const factor = (U[i]![k] ?? 0) / (U[k]![k] ?? 1);
      L[i]![k] = factor;
      for (let j = k; j < n; j++) U[i]![j] = (U[i]![j] ?? 0) - factor * (U[k]![j] ?? 0);
    }
  }
  return { L, U, P };
}

export function matrixInverse(A: Float64Array[]): Float64Array[] | null {
  const n = A.length;
  const { L, U } = luDecomposition(A);
  const inv = Array.from({ length: n }, () => new Float64Array(n));
  for (let col = 0; col < n; col++) {
    const e = new Float64Array(n);
    e[col] = 1;
    // Forward substitution: L y = e
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = (e[i] ?? 0) - L[i]!.slice(0, i).reduce((s, v, k) => s + v * (y[k] ?? 0), 0);
    }
    // Back substitution: U x = y
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = ((y[i] ?? 0) - U[i]!.slice(i + 1).reduce((s, v, k) => s + v * (x[i + 1 + k] ?? 0), 0)) / (U[i]![i] ?? 1);
    }
    for (let row = 0; row < n; row++) inv[row]![col] = x[row] ?? 0;
  }
  return inv;
}

export function matrixDeterminant(A: Float64Array[]): number {
  const { U, P } = luDecomposition(A);
  const n = A.length;
  const diagProd = U.reduce((s, row, i) => s * (row[i] ?? 1), 1);
  // Count permutation sign
  const perm = Array.from(P);
  let sign = 1;
  for (let i = 0; i < n; i++) {
    while (perm[i] !== i) {
      const j = perm[i]!;
      [perm[i], perm[j]] = [perm[j]!, perm[i]!];
      sign *= -1;
    }
  }
  return sign * diagProd;
}

export function eigenvaluesPowerMethod(A: Float64Array[], k = 3, maxIter = 100): { values: Float64Array; vectors: Float64Array[] } {
  const n = A.length;
  const values = new Float64Array(k);
  const vectors: Float64Array[] = [];
  let deflated = A.map(row => new Float64Array(row));

  for (let e = 0; e < k; e++) {
    let v = new Float64Array(n).map(() => Math.random());
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) / (norm + 1e-10);

    for (let iter = 0; iter < maxIter; iter++) {
      const Av = new Float64Array(n).map((_, i) => deflated[i]!.reduce((s, aij, j) => s + aij * (v[j] ?? 0), 0));
      norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0));
      const diff = Math.sqrt(Av.reduce((s, vj, j) => s + (vj / (norm + 1e-10) - (v[j] ?? 0)) ** 2, 0));
      v = new Float64Array(Av.map(x => x / (norm + 1e-10)));
      if (diff < 1e-10) break;
    }
    const lambda = v.reduce((s, vj, i) => s + vj * deflated[i]!.reduce((ss, aij, j) => ss + aij * (v[j] ?? 0), 0), 0);
    values[e] = lambda;
    vectors.push(new Float64Array(v));
    deflated = deflated.map((row, i) => new Float64Array(row.map((aij, j) => aij - lambda * (v[i] ?? 0) * (v[j] ?? 0))));
  }
  return { values, vectors };
}
