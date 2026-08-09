/**
 * Pairwise distance argmin utilities.
 * Mirrors sklearn.metrics.pairwise.pairwise_distances_argmin,
 * pairwise_distances_argmin_min, pairwise_distances_chunked.
 * Note: haversine_distances is in metrics/distance.ts.
 */

/** Euclidean distance between two vectors. */
function euclideanDist(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

/** Manhattan distance. */
function manhattanDist(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  return s;
}

function dist(
  a: Float64Array,
  b: Float64Array,
  metric: "euclidean" | "manhattan" | "cosine" | "l2" | "l1",
): number {
  if (metric === "manhattan" || metric === "l1") return manhattanDist(a, b);
  if (metric === "cosine") {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += (a[i] ?? 0) * (b[i] ?? 0);
      na += (a[i] ?? 0) ** 2;
      nb += (b[i] ?? 0) ** 2;
    }
    return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
  }
  return euclideanDist(a, b);
}

/**
 * Compute the index of the nearest point in Y for each point in X.
 * Mirrors sklearn.metrics.pairwise.pairwise_distances_argmin.
 */
export function pairwiseDistancesArgmin(
  X: Float64Array[],
  Y: Float64Array[],
  metric: "euclidean" | "manhattan" | "cosine" | "l2" | "l1" = "euclidean",
): Int32Array {
  return new Int32Array(
    X.map(xi => {
      let minD = Number.POSITIVE_INFINITY;
      let minJ = 0;
      for (let j = 0; j < Y.length; j++) {
        const d = dist(xi, Y[j]!, metric);
        if (d < minD) {
          minD = d;
          minJ = j;
        }
      }
      return minJ;
    }),
  );
}

/**
 * Compute the index and minimum distance to the nearest point in Y for each X.
 * Mirrors sklearn.metrics.pairwise.pairwise_distances_argmin_min.
 */
export function pairwiseDistancesArgminMin(
  X: Float64Array[],
  Y: Float64Array[],
  metric: "euclidean" | "manhattan" | "cosine" | "l2" | "l1" = "euclidean",
): { indices: Int32Array; distances: Float64Array } {
  const indices = new Int32Array(X.length);
  const distances = new Float64Array(X.length);
  for (let i = 0; i < X.length; i++) {
    let minD = Number.POSITIVE_INFINITY;
    let minJ = 0;
    for (let j = 0; j < Y.length; j++) {
      const d = dist(X[i]!, Y[j]!, metric);
      if (d < minD) {
        minD = d;
        minJ = j;
      }
    }
    indices[i] = minJ;
    distances[i] = minD;
  }
  return { indices, distances };
}

export function pairwiseDistancesChunked(
  X: Float64Array[],
  Y: Float64Array[],
  metric: "euclidean" | "manhattan" | "cosine" | "l2" | "l1" = "euclidean",
  workingMemory: number = 1024,
): Float64Array[] {
  const n = X.length;
  const m = Y.length;
  const rowsPerChunk = Math.max(1, Math.floor((workingMemory * 1024) / (m * 8)));
  const result: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));

  for (let start = 0; start < n; start += rowsPerChunk) {
    const end = Math.min(start + rowsPerChunk, n);
    for (let i = start; i < end; i++) {
      for (let j = 0; j < m; j++) {
        result[i]![j] = dist(X[i]!, Y[j]!, metric);
      }
    }
  }
  return result;
}
