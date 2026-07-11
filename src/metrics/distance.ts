/**
 * Distance metrics and similarity functions.
 * Mirrors sklearn.metrics.pairwise and scipy.spatial.distance functions.
 */

export type DistanceMetric =
  | "euclidean"
  | "manhattan"
  | "chebyshev"
  | "minkowski"
  | "cosine"
  | "correlation"
  | "hamming"
  | "jaccard";

/**
 * Compute pairwise distances between rows of X (and optionally Y).
 */
export function pairwiseDistances(
  X: Float64Array[],
  Y?: Float64Array[],
  metric: DistanceMetric = "euclidean",
  p = 2,
): Float64Array[] {
  const Ydata = Y ?? X;
  const n = X.length;
  const m = Ydata.length;
  return Array.from({ length: n }, (_, i) =>
    new Float64Array(m).map((_, j) =>
      _computeDist(X[i]!, Ydata[j]!, metric, p),
    ),
  );
}

function _computeDist(
  a: Float64Array,
  b: Float64Array,
  metric: DistanceMetric,
  p: number,
): number {
  const n = a.length;
  switch (metric) {
    case "euclidean": {
      let s = 0;
      for (let k = 0; k < n; k++) s += ((a[k] ?? 0) - (b[k] ?? 0)) ** 2;
      return Math.sqrt(s);
    }
    case "manhattan": {
      let s = 0;
      for (let k = 0; k < n; k++) s += Math.abs((a[k] ?? 0) - (b[k] ?? 0));
      return s;
    }
    case "chebyshev": {
      let s = 0;
      for (let k = 0; k < n; k++)
        s = Math.max(s, Math.abs((a[k] ?? 0) - (b[k] ?? 0)));
      return s;
    }
    case "minkowski": {
      let s = 0;
      for (let k = 0; k < n; k++) s += Math.abs((a[k] ?? 0) - (b[k] ?? 0)) ** p;
      return s ** (1 / p);
    }
    case "cosine": {
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let k = 0; k < n; k++) {
        dot += (a[k] ?? 0) * (b[k] ?? 0);
        na += (a[k] ?? 0) ** 2;
        nb += (b[k] ?? 0) ** 2;
      }
      const denom = Math.sqrt(na * nb);
      return denom < 1e-12 ? 1 : 1 - dot / denom;
    }
    case "correlation": {
      let aMean = 0;
      let bMean = 0;
      for (let k = 0; k < n; k++) {
        aMean += a[k] ?? 0;
        bMean += b[k] ?? 0;
      }
      aMean /= n;
      bMean /= n;
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let k = 0; k < n; k++) {
        const da = (a[k] ?? 0) - aMean;
        const db = (b[k] ?? 0) - bMean;
        dot += da * db;
        na += da * da;
        nb += db * db;
      }
      const denom = Math.sqrt(na * nb);
      return denom < 1e-12 ? 1 : 1 - dot / denom;
    }
    case "hamming": {
      let diff = 0;
      for (let k = 0; k < n; k++) if ((a[k] ?? 0) !== (b[k] ?? 0)) diff++;
      return diff / n;
    }
    case "jaccard": {
      let inter = 0;
      let union = 0;
      for (let k = 0; k < n; k++) {
        const av = (a[k] ?? 0) !== 0;
        const bv = (b[k] ?? 0) !== 0;
        if (av || bv) {
          union++;
          if (av && bv) inter++;
        }
      }
      return union === 0 ? 0 : 1 - inter / union;
    }
  }
}

/**
 * Compute pairwise cosine similarity matrix.
 */
export function cosineSimilarity(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ydata = Y ?? X;
  const n = X.length;
  const m = Ydata.length;

  // Normalize rows
  const normX = X.map((xi) => {
    let norm = 0;
    for (let j = 0; j < xi.length; j++) norm += (xi[j] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return xi.slice();
    return xi.map((v) => v / norm);
  });
  const normY = Ydata.map((yi) => {
    let norm = 0;
    for (let j = 0; j < yi.length; j++) norm += (yi[j] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return yi.slice();
    return yi.map((v) => v / norm);
  });

  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let k = 0; k < normX[i]!.length; k++)
        dot += (normX[i]![k] ?? 0) * (normY[j]![k] ?? 0);
      row[j]! = dot;
    }
    return row;
  });
}

/**
 * Compute pairwise Euclidean distances (squared) matrix — fast version.
 */
export function euclideanDistances(
  X: Float64Array[],
  Y?: Float64Array[],
  squared = false,
): Float64Array[] {
  const Ydata = Y ?? X;
  const n = X.length;
  const m = Ydata.length;
  const p = X[0]?.length ?? 0;

  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < p; k++)
        s += ((X[i]![k] ?? 0) - (Ydata[j]![k] ?? 0)) ** 2;
      row[j]! = squared ? s : Math.sqrt(s);
    }
    return row;
  });
}

/**
 * haversine_distances — great-circle distance between lat/long pairs (in radians).
 */
export function haversineDistances(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ydata = Y ?? X;
  const n = X.length;
  const m = Ydata.length;

  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    const lat1 = X[i]![0] ?? 0;
    const lon1 = X[i]![1] ?? 0;
    for (let j = 0; j < m; j++) {
      const lat2 = Ydata[j]![0] ?? 0;
      const lon2 = Ydata[j]![1] ?? 0;
      const dlat = lat2 - lat1;
      const dlon = lon2 - lon1;
      const a =
        Math.sin(dlat / 2) ** 2 +
        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
      row[j]! = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }
    return row;
  });
}

/**
 * Compute distance matrix (alias for pairwiseDistances with euclidean default).
 */
export function distanceMatrix(
  X: Float64Array[],
  Y?: Float64Array[],
  metric: DistanceMetric = "euclidean",
): Float64Array[] {
  return pairwiseDistances(X, Y, metric);
}
