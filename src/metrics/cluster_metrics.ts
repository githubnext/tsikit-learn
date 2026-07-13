/**
 * Additional cluster metrics.
 * Port of sklearn.metrics._cluster_ext
 */

/** Compute Calinski-Harabasz score (Variance Ratio Criterion) */
export function calinskiHarabaszScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const classSet = Array.from(new Set(Array.from(labels))).sort(
    (a, b) => a - b,
  );
  const k = classSet.length;
  if (k <= 1 || k >= n) return 0;

  // Global centroid
  const globalMean = new Float64Array(d);
  for (const x of X)
    for (let j = 0; j < d; j++) globalMean[j]! += (x[j] ?? 0) / n;

  // Between-cluster dispersion
  let bcd = 0;
  for (const c of classSet) {
    const mask = Array.from({ length: n }, (_, i) => labels[i] === c);
    const nc = mask.filter(Boolean).length;
    const cm = new Float64Array(d);
    for (let i = 0; i < n; i++)
      if (mask[i]) for (let j = 0; j < d; j++) cm[j]! += (X[i]?.[j] ?? 0) / nc;
    for (let j = 0; j < d; j++) bcd += nc * (cm[j]! - globalMean[j]!) ** 2;
  }

  // Within-cluster dispersion
  let wcd = 0;
  for (const c of classSet) {
    const mask = Array.from({ length: n }, (_, i) => labels[i] === c);
    const nc = mask.filter(Boolean).length;
    if (nc === 0) continue;
    const cm = new Float64Array(d);
    for (let i = 0; i < n; i++)
      if (mask[i]) for (let j = 0; j < d; j++) cm[j]! += (X[i]?.[j] ?? 0) / nc;
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue;
      for (let j = 0; j < d; j++) wcd += ((X[i]?.[j] ?? 0) - cm[j]!) ** 2;
    }
  }

  if (wcd === 0) return 1;
  return bcd / (k - 1) / (wcd / (n - k));
}

/** Compute Davies-Bouldin index */
export function daviesBouldinScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const classSet = Array.from(new Set(Array.from(labels))).sort(
    (a, b) => a - b,
  );
  const k = classSet.length;
  if (k <= 1) return 0;

  const centroids: Float64Array[] = [];
  const scatters: number[] = [];

  for (const c of classSet) {
    const mask = Array.from({ length: n }, (_, i) => labels[i] === c);
    const nc = mask.filter(Boolean).length;
    const cm = new Float64Array(d);
    for (let i = 0; i < n; i++)
      if (mask[i]) for (let j = 0; j < d; j++) cm[j]! += (X[i]?.[j] ?? 0) / nc;
    centroids.push(cm);
    let scatter = 0;
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue;
      for (let j = 0; j < d; j++) scatter += ((X[i]?.[j] ?? 0) - cm[j]!) ** 2;
    }
    scatters.push(Math.sqrt(scatter / nc));
  }

  let db = 0;
  for (let i = 0; i < k; i++) {
    let maxRatio = 0;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      let dist = 0;
      for (let l = 0; l < d; l++)
        dist += ((centroids[i]?.[l] ?? 0) - (centroids[j]?.[l] ?? 0)) ** 2;
      dist = Math.sqrt(dist);
      const ratio = ((scatters[i] ?? 0) + (scatters[j] ?? 0)) / (dist || 1e-10);
      if (ratio > maxRatio) maxRatio = ratio;
    }
    db += maxRatio;
  }
  return db / k;
}

/** Dunn index — ratio of min inter-cluster distance to max intra-cluster diameter */
export function dunnIndex(X: Float64Array[], labels: Int32Array): number {
  const n = X.length;
  const classSet = Array.from(new Set(Array.from(labels))).sort(
    (a, b) => a - b,
  );
  const k = classSet.length;
  if (k <= 1) return 0;

  const dist = (a: Float64Array, b: Float64Array): number => {
    let d = 0;
    for (let j = 0; j < a.length; j++) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(d);
  };

  // Min inter-cluster distance
  let minInter = Number.POSITIVE_INFINITY;
  for (let ci = 0; ci < k; ci++) {
    for (let cj = ci + 1; cj < k; cj++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (labels[i] === classSet[ci] && labels[j] === classSet[cj]) {
            const d = dist(X[i]!, X[j]!);
            if (d < minInter) minInter = d;
          }
        }
      }
    }
  }

  // Max intra-cluster diameter
  let maxIntra = 0;
  for (const c of classSet) {
    const clusterX = X.filter((_, i) => labels[i] === c);
    for (let i = 0; i < clusterX.length; i++) {
      for (let j = i + 1; j < clusterX.length; j++) {
        const d = dist(clusterX[i]!, clusterX[j]!);
        if (d > maxIntra) maxIntra = d;
      }
    }
  }

  return maxIntra > 0 ? minInter / maxIntra : 0;
}

/** Compute Xie-Beni index for fuzzy clustering */
export function xieBeniIndex(
  X: Float64Array[],
  membershipMatrix: Float64Array[],
  m = 2,
): number {
  const n = X.length;
  const k = membershipMatrix[0]?.length ?? 0;
  const d = X[0]?.length ?? 0;

  // Compute fuzzy centroids
  const centroids: Float64Array[] = Array.from(
    { length: k },
    () => new Float64Array(d),
  );
  const membershipSums = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      const mu = (membershipMatrix[i]?.[c] ?? 0) ** m;
      membershipSums[c]! += mu;
      for (let j = 0; j < d; j++) centroids[c]![j]! += mu * (X[i]?.[j] ?? 0);
    }
  }
  for (let c = 0; c < k; c++) {
    for (let j = 0; j < d; j++) centroids[c]![j]! /= membershipSums[c]! || 1;
  }

  // Compactness
  let compactness = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      const mu = (membershipMatrix[i]?.[c] ?? 0) ** m;
      let dist = 0;
      for (let j = 0; j < d; j++)
        dist += ((X[i]?.[j] ?? 0) - (centroids[c]?.[j] ?? 0)) ** 2;
      compactness += mu * dist;
    }
  }

  // Separation
  let minDist = Number.POSITIVE_INFINITY;
  for (let ci = 0; ci < k; ci++) {
    for (let cj = ci + 1; cj < k; cj++) {
      let dist = 0;
      for (let j = 0; j < d; j++)
        dist += ((centroids[ci]?.[j] ?? 0) - (centroids[cj]?.[j] ?? 0)) ** 2;
      if (dist < minDist) minDist = dist;
    }
  }

  return compactness / (n * minDist + 1e-10);
}
