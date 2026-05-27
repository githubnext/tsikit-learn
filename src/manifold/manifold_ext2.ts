/**
 * Extended manifold learning: UMAP utilities, Isomap geodesic distances,
 * LLE reconstruction error, and manifold quality metrics.
 */

/** Compute geodesic distances via Floyd-Warshall on k-NN graph. */
export function geodesicDistances(
  X: Float64Array[],
  kNeighbors: number,
): Float64Array[] {
  const n = X.length;
  const INF = Number.POSITIVE_INFINITY;
  // Initialize distance matrix
  const D: Float64Array[] = Array.from({ length: n }, () =>
    new Float64Array(n).fill(INF)
  );
  for (let i = 0; i < n; i++) D[i]![i] = 0;

  // Build k-NN graph with Euclidean distances
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    if (xi === undefined) continue;
    const dists = X.map((xj, j) => {
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
      return { j, d: Math.sqrt(d2) };
    }).sort((a, b) => a.d - b.d).slice(1, kNeighbors + 1);

    for (const { j, d } of dists) {
      D[i]![j] = Math.min(D[i]![j] ?? INF, d);
      D[j]![i] = Math.min(D[j]![i] ?? INF, d);
    }
  }

  // Floyd-Warshall
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const through = (D[i]?.[k] ?? INF) + (D[k]?.[j] ?? INF);
        if (through < (D[i]?.[j] ?? INF)) D[i]![j] = through;
      }
    }
  }
  return D;
}

/** Trustworthiness metric for manifold quality (sklearn-compatible). */
export function trustworthiness(
  X: Float64Array[],
  XEmbedded: Float64Array[],
  nNeighbors: number,
): number {
  const n = X.length;

  const rankHigh = computeRanks(X, n);
  const rankLow = computeRanks(XEmbedded, n);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    // Neighbors in low-dim space
    const lowNeighbors = getSortedNeighbors(XEmbedded, i, nNeighbors);
    for (const j of lowNeighbors) {
      const r = rankHigh[i]?.[j] ?? 0;
      if (r > nNeighbors) sum += r - nNeighbors;
    }
  }
  const denom = nNeighbors * n * (2 * n - 3 * nNeighbors - 1) / 2;
  return 1 - 2 * sum / (denom + 1e-10);
}

function computeRanks(X: Float64Array[], n: number): Int32Array[] {
  return Array.from({ length: n }, (_, i) => {
    const xi = X[i];
    if (xi === undefined) return new Int32Array(n);
    const dists = X.map((xj, j) => {
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
      return { j, d: d2 };
    }).sort((a, b) => a.d - b.d);
    const ranks = new Int32Array(n);
    for (let r = 0; r < dists.length; r++) ranks[dists[r]!.j] = r;
    return ranks;
  });
}

function getSortedNeighbors(X: Float64Array[], i: number, k: number): number[] {
  const xi = X[i];
  if (xi === undefined) return [];
  return X.map((xj, j) => {
    let d2 = 0;
    for (let l = 0; l < xi.length; l++) d2 += ((xi[l] ?? 0) - (xj[l] ?? 0)) ** 2;
    return { j, d: d2 };
  })
    .filter(({ j }) => j !== i)
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
    .map(({ j }) => j);
}

/** Continuity metric: complement of trustworthiness. */
export function continuity(
  X: Float64Array[],
  XEmbedded: Float64Array[],
  nNeighbors: number,
): number {
  // Swap roles of high-dim and low-dim
  return trustworthiness(XEmbedded, X, nNeighbors);
}

/** Stress (Kruskal stress) for MDS quality. */
export function kruskalStress(
  dHigh: Float64Array[],
  dLow: Float64Array[],
): number {
  let numerator = 0, denominator = 0;
  const n = dHigh.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dH = dHigh[i]?.[j] ?? 0;
      const dL = dLow[i]?.[j] ?? 0;
      numerator += (dH - dL) ** 2;
      denominator += dH ** 2;
    }
  }
  return Math.sqrt(numerator / (denominator + 1e-10));
}

/** LLE reconstruction error. */
export function lleReconstructionError(
  X: Float64Array[],
  W: Float64Array[],
): number {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  let error = 0;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    if (xi === undefined) continue;
    const wi = W[i];
    if (wi === undefined) continue;
    for (let k = 0; k < d; k++) {
      let rec = 0;
      for (let j = 0; j < n; j++) rec += (wi[j] ?? 0) * (X[j]?.[k] ?? 0);
      error += ((xi[k] ?? 0) - rec) ** 2;
    }
  }
  return error;
}
