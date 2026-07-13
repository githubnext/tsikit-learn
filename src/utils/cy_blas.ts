/**
 * BLAS-like routines: DGEMM, DGEMV, DSYRK, DTRSM optimized implementations.
 */

export function dgemm(
  A: Float64Array[],
  B: Float64Array[],
  transA = false,
  transB = false,
  alpha = 1.0,
  beta = 0.0,
  C?: Float64Array[],
): Float64Array[] {
  const m = transA ? (A[0]?.length ?? 0) : A.length;
  const n = transB ? B.length : (B[0]?.length ?? 0);
  const k = transA ? A.length : (A[0]?.length ?? 0);
  const result: Float64Array[] =
    C ?? Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        const aVal = transA ? (A[p]?.[i] ?? 0) : (A[i]?.[p] ?? 0);
        const bVal = transB ? (B[j]?.[p] ?? 0) : (B[p]?.[j] ?? 0);
        sum += aVal * bVal;
      }
      result[i]![j] = alpha * sum + beta * (result[i]![j] ?? 0);
    }
  }
  return result;
}

export function dgemv(
  A: Float64Array[],
  x: Float64Array,
  transA = false,
  alpha = 1.0,
  beta = 0.0,
  y?: Float64Array,
): Float64Array {
  const m = transA ? (A[0]?.length ?? 0) : A.length;
  const n = transA ? A.length : (A[0]?.length ?? 0);
  const result = y ?? new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const aVal = transA ? (A[j]?.[i] ?? 0) : (A[i]?.[j] ?? 0);
      sum += aVal * (x[j] ?? 0);
    }
    result[i] = alpha * sum + beta * (result[i] ?? 0);
  }
  return result;
}

export function dsyrk(
  A: Float64Array[],
  lower = true,
  trans = false,
  alpha = 1.0,
  beta = 0.0,
): Float64Array[] {
  const n = trans ? (A[0]?.length ?? 0) : A.length;
  const k = trans ? A.length : (A[0]?.length ?? 0);
  const C: Float64Array[] = Array.from(
    { length: n },
    () => new Float64Array(n),
  );
  for (let i = 0; i < n; i++) {
    const jStart = lower ? 0 : i;
    const jEnd = lower ? i + 1 : n;
    for (let j = jStart; j < jEnd; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        const ai = trans ? (A[p]?.[i] ?? 0) : (A[i]?.[p] ?? 0);
        const aj = trans ? (A[p]?.[j] ?? 0) : (A[j]?.[p] ?? 0);
        sum += ai * aj;
      }
      C[i]![j] = alpha * sum + beta * (C[i]![j] ?? 0);
      if (i !== j) C[j]![i] = C[i]![j]!;
    }
  }
  return C;
}

export function dtrsm(
  L: Float64Array[],
  B: Float64Array[],
  lower = true,
  transL = false,
): Float64Array[] {
  const n = L.length;
  const nRhs = B[0]?.length ?? 1;
  const X: Float64Array[] = B.map((row) => new Float64Array(row));
  if (lower && !transL) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nRhs; j++) {
        let s = X[i]?.[j] ?? 0;
        for (let k = 0; k < i; k++) s -= (L[i]?.[k] ?? 0) * (X[k]?.[j] ?? 0);
        X[i]![j] = s / Math.max(L[i]?.[i] ?? 1, 1e-10);
      }
    }
  } else {
    for (let i = n - 1; i >= 0; i--) {
      for (let j = 0; j < nRhs; j++) {
        let s = X[i]?.[j] ?? 0;
        for (let k = i + 1; k < n; k++)
          s -= (L[k]?.[i] ?? 0) * (X[k]?.[j] ?? 0);
        X[i]![j] = s / Math.max(L[i]?.[i] ?? 1, 1e-10);
      }
    }
  }
  return X;
}

export function ddot(x: Float64Array, y: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (y[i] ?? 0);
  return s;
}

export function dnrm2(x: Float64Array): number {
  let s = 0;
  for (const v of x) s += v * v;
  return Math.sqrt(s);
}

export function dscal(alpha: number, x: Float64Array): void {
  for (let i = 0; i < x.length; i++) x[i] = alpha * (x[i] ?? 0);
}

export function daxpy(alpha: number, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < x.length; i++) y[i] = (y[i] ?? 0) + alpha * (x[i] ?? 0);
}

export function idamax(x: Float64Array): number {
  let maxVal = -1;
  let maxIdx = 0;
  for (let i = 0; i < x.length; i++) {
    const v = Math.abs(x[i] ?? 0);
    if (v > maxVal) {
      maxVal = v;
      maxIdx = i;
    }
  }
  return maxIdx;
}

export function dcopy(x: Float64Array): Float64Array {
  return new Float64Array(x);
}
