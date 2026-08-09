/**
 * Cluster diagnostic utilities.
 * Mirrors scikit-learn's metrics.silhouette_score, calinski_harabasz_score, davies_bouldin_score.
 */

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

/**
 * Compute the Silhouette Coefficient for each sample.
 */
export function silhouetteSamples(
  X: Float64Array[],
  labels: Int32Array,
): Float64Array {
  const n = X.length;
  const clusterIds = Array.from(new Set(Array.from(labels))).sort(
    (a, b) => a - b,
  );
  const scores = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const li = labels[i]!;
    // Intra-cluster mean distance (a)
    const sameCluster = clusterIds
      .filter((c) => c === li)
      .map(() => {
        let sum = 0;
        let count = 0;
        for (let j = 0; j < n; j++) {
          if (j !== i && labels[j] === li) {
            sum += euclidean(X[i]!, X[j]!);
            count++;
          }
        }
        return count === 0 ? 0 : sum / count;
      });
    const a = sameCluster[0] ?? 0;

    // Nearest-cluster mean distance (b)
    let b = Number.POSITIVE_INFINITY;
    for (const c of clusterIds) {
      if (c === li) continue;
      let sum = 0;
      let count = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === c) {
          sum += euclidean(X[i]!, X[j]!);
          count++;
        }
      }
      if (count > 0) b = Math.min(b, sum / count);
    }

    const maxAB = Math.max(a, Number.isFinite(b) ? b : 0);
    scores[i] = maxAB < 1e-10 ? 0 : ((Number.isFinite(b) ? b : 0) - a) / maxAB;
  }
  return scores;
}

/**
 * Mean silhouette coefficient.
 */
export function silhouetteScore(X: Float64Array[], labels: Int32Array): number {
  const samples = silhouetteSamples(X, labels);
  return samples.reduce((s, v) => s + v, 0) / samples.length;
}

/**
 * Calinski-Harabasz Index (Variance Ratio Criterion).
 * Higher is better.
 */
export function calinskiHarabaszScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const clusterIds = Array.from(new Set(Array.from(labels)));
  const k = clusterIds.length;
  if (k <= 1 || k >= n) return 0;

  const globalMean = new Float64Array(nFeatures);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++)
      globalMean[j] = (globalMean[j] ?? 0) + (row[j] ?? 0) / n;
  }

  let trBw = 0; // Between-cluster scatter
  let trWw = 0; // Within-cluster scatter

  for (const c of clusterIds) {
    const clusterPoints = X.filter((_, i) => labels[i] === c);
    const nc = clusterPoints.length;
    if (nc === 0) continue;
    const centroid = new Float64Array(nFeatures);
    for (const p of clusterPoints) {
      for (let j = 0; j < nFeatures; j++)
        centroid[j] = (centroid[j] ?? 0) + (p[j] ?? 0) / nc;
    }
    for (let j = 0; j < nFeatures; j++) {
      trBw += nc * ((centroid[j] ?? 0) - (globalMean[j] ?? 0)) ** 2;
    }
    for (const p of clusterPoints) {
      for (let j = 0; j < nFeatures; j++) {
        trWw += ((p[j] ?? 0) - (centroid[j] ?? 0)) ** 2;
      }
    }
  }

  if (trWw < 1e-10) return 1;
  return trBw / (k - 1) / (trWw / (n - k));
}

/**
 * Davies-Bouldin Index. Lower is better.
 */
export function daviesBouldinScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const nFeatures = X[0]?.length ?? 0;
  const clusterIds = Array.from(new Set(Array.from(labels)));
  const k = clusterIds.length;
  if (k <= 1) return 0;

  const centroids: Float64Array[] = [];
  const dispersions: number[] = [];

  for (const c of clusterIds) {
    const pts = X.filter((_, i) => labels[i] === c);
    const nc = pts.length;
    const centroid = new Float64Array(nFeatures);
    for (const p of pts) {
      for (let j = 0; j < nFeatures; j++)
        centroid[j] = (centroid[j] ?? 0) + (p[j] ?? 0) / nc;
    }
    centroids.push(centroid);
    dispersions.push(pts.reduce((s, p) => s + euclidean(p, centroid), 0) / nc);
  }

  let db = 0;
  for (let i = 0; i < k; i++) {
    let maxR = 0;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const dij = euclidean(centroids[i]!, centroids[j]!);
      if (dij > 1e-10) {
        maxR = Math.max(
          maxR,
          ((dispersions[i] ?? 0) + (dispersions[j] ?? 0)) / dij,
        );
      }
    }
    db += maxR;
  }
  return db / k;
}
