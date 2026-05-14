/**
 * Pairwise distance and kernel metrics.
 * Mirrors sklearn.metrics.pairwise.
 */

export type MetricName = "euclidean" | "cosine" | "manhattan" | "chebyshev" | "minkowski";

export function euclideanDistances(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const A = Y ?? X;
  const n = X.length;
  const m = A.length;
  const D: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(0);
    for (let j = 0; j < m; j++) {
      const aj = A[j] ?? new Float64Array(0);
      let d = 0;
      for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (aj[k] ?? 0)) ** 2;
      (D[i] as Float64Array)[j] = Math.sqrt(d);
    }
  }
  return D;
}

export function manhattanDistances(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const A = Y ?? X;
  const n = X.length;
  const m = A.length;
  const D: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(0);
    for (let j = 0; j < m; j++) {
      const aj = A[j] ?? new Float64Array(0);
      let d = 0;
      for (let k = 0; k < xi.length; k++) d += Math.abs((xi[k] ?? 0) - (aj[k] ?? 0));
      (D[i] as Float64Array)[j] = d;
    }
  }
  return D;
}

export function cosineSimilarity(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const A = Y ?? X;
  const n = X.length;
  const m = A.length;
  const S: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(0);
    let normX = 0;
    for (let k = 0; k < xi.length; k++) normX += (xi[k] ?? 0) ** 2;
    normX = Math.sqrt(normX) || 1;
    for (let j = 0; j < m; j++) {
      const aj = A[j] ?? new Float64Array(0);
      let dot = 0; let normA = 0;
      for (let k = 0; k < xi.length; k++) {
        dot += (xi[k] ?? 0) * (aj[k] ?? 0);
        normA += (aj[k] ?? 0) ** 2;
      }
      normA = Math.sqrt(normA) || 1;
      (S[i] as Float64Array)[j] = dot / (normX * normA);
    }
  }
  return S;
}

export function cosineDistances(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const sim = cosineSimilarity(X, Y);
  return sim.map(row => Float64Array.from(row.map(v => 1 - v)));
}

export function pairwiseDistances(
  X: Float64Array[],
  Y?: Float64Array[],
  metric: MetricName = "euclidean",
): Float64Array[] {
  switch (metric) {
    case "euclidean": return euclideanDistances(X, Y);
    case "manhattan": return manhattanDistances(X, Y);
    case "cosine": return cosineDistances(X, Y);
    case "chebyshev": {
      const A = Y ?? X;
      const n = X.length;
      const m = A.length;
      return Array.from({ length: n }, (_, i) => {
        const xi = X[i] ?? new Float64Array(0);
        const row = new Float64Array(m);
        for (let j = 0; j < m; j++) {
          const aj = A[j] ?? new Float64Array(0);
          let d = 0;
          for (let k = 0; k < xi.length; k++) d = Math.max(d, Math.abs((xi[k] ?? 0) - (aj[k] ?? 0)));
          row[j] = d;
        }
        return row;
      });
    }
    default: return euclideanDistances(X, Y);
  }
}

export function rbfKernelMatrix(X: Float64Array[], Y?: Float64Array[], gamma?: number): Float64Array[] {
  const A = Y ?? X;
  const p = (X[0] ?? new Float64Array(0)).length;
  const g = gamma ?? 1 / p;
  const D = euclideanDistances(X, A);
  return D.map(row => Float64Array.from(row.map(d => Math.exp(-g * d ** 2))));
}

export function linearKernel(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const A = Y ?? X;
  const n = X.length;
  const m = A.length;
  return Array.from({ length: n }, (_, i) => {
    const xi = X[i] ?? new Float64Array(0);
    const row = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      const aj = A[j] ?? new Float64Array(0);
      let dot = 0;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (aj[k] ?? 0);
      row[j] = dot;
    }
    return row;
  });
}

export function polynomialKernel(
  X: Float64Array[],
  Y?: Float64Array[],
  degree = 3,
  gamma?: number,
  coef0 = 1,
): Float64Array[] {
  const A = Y ?? X;
  const p = (X[0] ?? new Float64Array(0)).length;
  const g = gamma ?? 1 / p;
  const lin = linearKernel(X, A);
  return lin.map(row => Float64Array.from(row.map(v => (g * v + coef0) ** degree)));
}
