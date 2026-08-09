/**
 * OPTICS clustering utility functions — ported from sklearn.cluster._optics
 * clusterOpticsDbscan, clusterOpticsXi, extractDbscanClustering
 */

export interface OpticsClusterResult {
  /** Cluster labels for each sample (-1 = noise) */
  labels: Int32Array;
  /** Number of clusters found (excluding noise) */
  nClusters: number;
}

/**
 * Perform DBSCAN extraction from OPTICS reachability distances.
 *
 * @param reachabilityDistances Reachability distances from OPTICS
 * @param coreDistances Core distances from OPTICS
 * @param ordering Sample ordering from OPTICS
 * @param eps The maximum reachability distance for cluster membership
 * @returns Cluster labels for each sample
 */
export function clusterOpticsDbscan(
  reachabilityDistances: Float64Array,
  coreDistances: Float64Array,
  ordering: Int32Array,
  eps: number,
): OpticsClusterResult {
  const nSamples = reachabilityDistances.length;
  const labels = new Int32Array(nSamples).fill(-1);
  let clusterLabel = 0;

  let i = 0;
  while (i < nSamples) {
    const sampleIdx = ordering[i] ?? i;
    const reach = reachabilityDistances[sampleIdx] ?? Number.POSITIVE_INFINITY;
    const core = coreDistances[sampleIdx] ?? Number.POSITIVE_INFINITY;

    if (reach > eps) {
      // This point starts a potential new cluster or is noise
      if (core <= eps) {
        // It is a core point — start a new cluster
        clusterLabel++;
        labels[sampleIdx] = clusterLabel;
        i++;
        // Expand cluster
        while (i < nSamples) {
          const nextIdx = ordering[i] ?? i;
          const nextReach = reachabilityDistances[nextIdx] ?? Number.POSITIVE_INFINITY;
          if (nextReach <= eps) {
            labels[nextIdx] = clusterLabel;
            i++;
          } else {
            break;
          }
        }
      } else {
        // Noise point
        i++;
      }
    } else {
      // Continue current cluster
      if (clusterLabel > 0) {
        labels[sampleIdx] = clusterLabel;
      }
      i++;
    }
  }

  return { labels, nClusters: clusterLabel };
}

/**
 * Perform xi-based cluster extraction from OPTICS results.
 *
 * @param reachabilityDistances Reachability distances from OPTICS
 * @param ordering Sample ordering from OPTICS
 * @param minSamples Minimum number of samples in a cluster
 * @param xi Determines the minimum steepness (0 < xi < 1)
 * @param minClusterSize Minimum size of a cluster (as fraction or count)
 * @returns Cluster labels
 */
export function clusterOpticsXi(
  reachabilityDistances: Float64Array,
  ordering: Int32Array,
  minSamples: number,
  xi = 0.05,
  minClusterSize?: number,
): OpticsClusterResult {
  const nSamples = ordering.length;
  const minSize = minClusterSize ?? minSamples;
  const labels = new Int32Array(nSamples).fill(-1);

  // Build ordered reachabilities
  const orderedReach = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    orderedReach[i] = reachabilityDistances[ordering[i] ?? i] ?? Number.POSITIVE_INFINITY;
  }

  // Find steep upward and downward areas
  interface SteepArea {
    start: number;
    end: number;
    kind: "up" | "down";
  }

  const steepAreas: SteepArea[] = [];

  for (let i = 0; i < nSamples - 1; i++) {
    const r1 = orderedReach[i] ?? 0;
    const r2 = orderedReach[i + 1] ?? 0;
    if (r1 === 0) continue;

    const ratio = r2 / r1;
    if (ratio >= 1 + xi) {
      steepAreas.push({ start: i, end: i + 1, kind: "up" });
    } else if (r2 > 0 && r1 / r2 >= 1 + xi) {
      steepAreas.push({ start: i, end: i + 1, kind: "down" });
    }
  }

  // Simple cluster extraction: pair each down area with a matching up area
  let clusterLabel = 0;

  for (let di = 0; di < steepAreas.length; di++) {
    const down = steepAreas[di]!;
    if (down.kind !== "down") continue;

    for (let ui = di + 1; ui < steepAreas.length; ui++) {
      const up = steepAreas[ui]!;
      if (up.kind !== "up") continue;

      const clusterStart = down.end;
      const clusterEnd = up.start;
      const size = clusterEnd - clusterStart;

      if (size < minSize) continue;

      clusterLabel++;
      for (let i = clusterStart; i <= clusterEnd && i < nSamples; i++) {
        const sampleIdx = ordering[i] ?? i;
        if (labels[sampleIdx] === -1) {
          labels[sampleIdx] = clusterLabel;
        }
      }
      break;
    }
  }

  return { labels, nClusters: clusterLabel };
}

/**
 * Extract DBSCAN-style clusters from OPTICS at multiple eps values.
 */
export interface EpsClusterResult {
  eps: number;
  labels: Int32Array;
  nClusters: number;
}

export function extractDbscanClustering(
  reachabilityDistances: Float64Array,
  coreDistances: Float64Array,
  ordering: Int32Array,
  epsValues: Float64Array,
): EpsClusterResult[] {
  return Array.from(epsValues).map((eps) => {
    const result = clusterOpticsDbscan(reachabilityDistances, coreDistances, ordering, eps);
    return { eps, ...result };
  });
}

/**
 * Compute the reachability plot for visualization.
 * Returns pairs of (order_index, reachability_distance) for plotting.
 */
export function reachabilityPlotData(
  reachabilityDistances: Float64Array,
  ordering: Int32Array,
): { orderIndex: Int32Array; reachDistance: Float64Array } {
  const n = ordering.length;
  const orderIndex = new Int32Array(n);
  const reachDistance = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    orderIndex[i] = i;
    reachDistance[i] = reachabilityDistances[ordering[i] ?? i] ?? Number.POSITIVE_INFINITY;
  }

  return { orderIndex, reachDistance };
}
