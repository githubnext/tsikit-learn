/**
 * Extended clustering utilities: cluster quality scoring helpers,
 * cluster merge/split operations, and consensus clustering.
 */

/** Compute inertia (within-cluster sum of squares) given labels and centroids. */
export function computeInertia(
  X: Float64Array[],
  labels: Int32Array,
  centroids: Float64Array[],
): number {
  let inertia = 0.0;
  for (let i = 0; i < X.length; i++) {
    const label = labels[i] ?? 0;
    const centroid = centroids[label];
    if (centroid === undefined) continue;
    const xi = X[i];
    if (xi === undefined) continue;
    let dist2 = 0.0;
    for (let j = 0; j < xi.length; j++) {
      const diff = (xi[j] ?? 0) - (centroid[j] ?? 0);
      dist2 += diff * diff;
    }
    inertia += dist2;
  }
  return inertia;
}

/** Compute cluster sizes given labels and n_clusters. */
export function clusterSizes(labels: Int32Array, nClusters: number): Int32Array {
  const sizes = new Int32Array(nClusters);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i] ?? 0;
    if (l >= 0 && l < nClusters) {
      sizes[l] = (sizes[l] ?? 0) + 1;
    }
  }
  return sizes;
}

/** Compute centroids from data and labels. */
export function computeCentroids(
  X: Float64Array[],
  labels: Int32Array,
  nClusters: number,
  nFeatures: number,
): Float64Array[] {
  const sums: Float64Array[] = Array.from({ length: nClusters }, () => new Float64Array(nFeatures));
  const counts = new Int32Array(nClusters);
  for (let i = 0; i < X.length; i++) {
    const l = labels[i] ?? 0;
    if (l < 0 || l >= nClusters) continue;
    const xi = X[i];
    if (xi === undefined) continue;
    const s = sums[l];
    if (s === undefined) continue;
    for (let j = 0; j < nFeatures; j++) {
      s[j] = (s[j] ?? 0) + (xi[j] ?? 0);
    }
    counts[l] = (counts[l] ?? 0) + 1;
  }
  return sums.map((s, k) => {
    const c = counts[k] ?? 1;
    return s.map((v) => v / Math.max(1, c));
  });
}

/** Davies-Bouldin index (lower is better). */
export function daviesBouldinScore(X: Float64Array[], labels: Int32Array): number {
  const uniqueLabels = [...new Set(Array.from(labels))].filter((l) => l >= 0);
  const nClusters = uniqueLabels.length;
  if (nClusters < 2) return 0;
  const nFeatures = X[0]?.length ?? 0;
  const centroids = computeCentroids(X, labels, nClusters, nFeatures);

  const s: number[] = centroids.map((c, k) => {
    const members = X.filter((_, i) => (labels[i] ?? -1) === k);
    if (members.length === 0) return 0;
    const avg = members.reduce((acc, xi) => {
      let dist = 0;
      for (let j = 0; j < c.length; j++) dist += ((xi[j] ?? 0) - (c[j] ?? 0)) ** 2;
      return acc + Math.sqrt(dist);
    }, 0) / members.length;
    return avg;
  });

  let db = 0;
  for (let i = 0; i < nClusters; i++) {
    let maxR = 0;
    for (let j = 0; j < nClusters; j++) {
      if (i === j) continue;
      const ci = centroids[i];
      const cj = centroids[j];
      if (ci === undefined || cj === undefined) continue;
      let dist = 0;
      for (let d = 0; d < nFeatures; d++) dist += ((ci[d] ?? 0) - (cj[d] ?? 0)) ** 2;
      dist = Math.sqrt(dist);
      const r = ((s[i] ?? 0) + (s[j] ?? 0)) / (dist + 1e-10);
      if (r > maxR) maxR = r;
    }
    db += maxR;
  }
  return db / nClusters;
}

/** Calinski-Harabasz index (higher is better). */
export function calinskiHarabaszScore(X: Float64Array[], labels: Int32Array): number {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const uniqueLabels = [...new Set(Array.from(labels))].filter((l) => l >= 0);
  const k = uniqueLabels.length;
  if (k < 2 || n <= k) return 0;

  const grandMean = new Float64Array(nFeatures);
  for (const xi of X) {
    for (let j = 0; j < nFeatures; j++) grandMean[j] = (grandMean[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < nFeatures; j++) grandMean[j] = (grandMean[j] ?? 0) / n;

  const centroids = computeCentroids(X, labels, k, nFeatures);
  const sizes = clusterSizes(labels, k);

  let bcd = 0;
  for (let c = 0; c < k; c++) {
    const centroid = centroids[c];
    if (centroid === undefined) continue;
    let dist = 0;
    for (let j = 0; j < nFeatures; j++) dist += ((centroid[j] ?? 0) - (grandMean[j] ?? 0)) ** 2;
    bcd += (sizes[c] ?? 0) * dist;
  }

  let wcd = 0;
  for (let i = 0; i < n; i++) {
    const l = labels[i] ?? 0;
    const centroid = centroids[l];
    const xi = X[i];
    if (centroid === undefined || xi === undefined) continue;
    for (let j = 0; j < nFeatures; j++) wcd += ((xi[j] ?? 0) - (centroid[j] ?? 0)) ** 2;
  }

  return (bcd / (k - 1)) / (wcd / (n - k) + 1e-10);
}
