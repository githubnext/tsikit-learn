/**
 * Hierarchical clustering utilities — analogous to scipy.cluster.hierarchy and
 * sklearn's internal _agglomerative_clustering helpers.
 */

/** Linkage methods supported by the `linkage` function. */
export type LinkageMethod =
  | "single"
  | "complete"
  | "average"
  | "ward"
  | "centroid"
  | "median"
  | "weighted";

/** A single row of a linkage matrix: [idx1, idx2, distance, count]. */
export interface HierarchicalLinkageRow {
  idx1: number;
  idx2: number;
  distance: number;
  count: number;
}

/**
 * Computes a hierarchical clustering linkage matrix from a condensed distance matrix.
 *
 * @param distMatrix Condensed distance matrix (length = n*(n-1)/2 for n observations).
 * @param n          Number of observations.
 * @param method     Linkage method (default "single").
 * @returns Array of (n-1) HierarchicalLinkageRow entries in merge order.
 */
export function linkage(
  distMatrix: Float64Array,
  n: number,
  method: LinkageMethod = "single",
): HierarchicalLinkageRow[] {
  // Build full distance matrix for simplicity (nn-chain would be faster)
  const D = new Float64Array(n * n).fill(Number.POSITIVE_INFINITY);
  for (let i = 0; i < n; i++) D[i * n + i] = 0;
  let k = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distMatrix[k++]!;
      D[i * n + j] = d;
      D[j * n + i] = d;
    }
  }

  // Active cluster set
  const active = new Set<number>(Array.from({ length: n }, (_, i) => i));
  // Cluster sizes
  const sizes = new Float64Array(2 * n).fill(1);
  // Cluster centroids (for ward / centroid / median)
  const identity = new Float64Array(n * n); // n×n identity as initial centroids placeholder
  for (let i = 0; i < n; i++) identity[i * n + i] = 1;

  const result: HierarchicalLinkageRow[] = [];
  let nextId = n;

  // Expanded distance matrix that grows with new cluster nodes
  const maxN = 2 * n;
  const bigD = new Float64Array(maxN * maxN).fill(Number.POSITIVE_INFINITY);
  for (let i = 0; i < n; i++) {
    bigD[i * maxN + i] = 0;
    for (let j = 0; j < n; j++) bigD[i * maxN + j] = D[i * n + j]!;
  }

  while (active.size > 1) {
    // Find nearest pair
    let minDist = Number.POSITIVE_INFINITY;
    let a = -1;
    let b = -1;
    for (const i of active) {
      for (const j of active) {
        if (j <= i) continue;
        const d = bigD[i * maxN + j]!;
        if (d < minDist) {
          minDist = d;
          a = i;
          b = j;
        }
      }
    }
    if (a < 0) break;

    const sA = sizes[a]!;
    const sB = sizes[b]!;
    const sNew = sA + sB;
    sizes[nextId] = sNew;

    // Compute distances from new cluster to all remaining clusters
    for (const c of active) {
      if (c === a || c === b) continue;
      const dac = bigD[a * maxN + c]!;
      const dbc = bigD[b * maxN + c]!;
      const sC = sizes[c]!;
      let dNew: number;
      switch (method) {
        case "single":
          dNew = Math.min(dac, dbc);
          break;
        case "complete":
          dNew = Math.max(dac, dbc);
          break;
        case "average":
          dNew = (sA * dac + sB * dbc) / sNew;
          break;
        case "ward": {
          const dab = bigD[a * maxN + b]!;
          dNew = Math.sqrt(
            ((sA + sC) * dac * dac + (sB + sC) * dbc * dbc - sC * dab * dab) /
              (sNew + sC),
          );
          break;
        }
        case "centroid":
          dNew = Math.sqrt(
            (sA * dac * dac + sB * dbc * dbc) / sNew -
              (sA * sB * bigD[a * maxN + b]! * bigD[a * maxN + b]!) /
                (sNew * sNew),
          );
          break;
        case "median":
          dNew = Math.sqrt(
            0.5 * dac * dac +
              0.5 * dbc * dbc -
              0.25 * bigD[a * maxN + b]! * bigD[a * maxN + b]!,
          );
          break;
        case "weighted":
          dNew = 0.5 * dac + 0.5 * dbc;
          break;
        default:
          dNew = Math.min(dac, dbc);
      }
      bigD[nextId * maxN + c] = dNew;
      bigD[c * maxN + nextId] = dNew;
    }
    bigD[nextId * maxN + nextId] = 0;

    result.push({ idx1: a, idx2: b, distance: minDist, count: sNew });
    active.delete(a);
    active.delete(b);
    active.add(nextId);
    nextId++;
  }

  return result;
}

/**
 * Cuts a dendrogram at a given number of clusters.
 * Returns an Int32Array of cluster labels (length = n).
 */
export function cutTree(
  rows: HierarchicalLinkageRow[],
  n: number,
  nClusters: number,
): Int32Array {
  // Each leaf starts in its own cluster; merge bottom-up, stop early
  const parent = new Int32Array(2 * n).fill(-1);
  const mergeOrder = rows.slice(0, n - nClusters);

  let nextId = n;
  for (const row of mergeOrder) {
    parent[row.idx1] = nextId;
    parent[row.idx2] = nextId;
    nextId++;
  }

  const labels = new Int32Array(n);
  const rootLabels = new Map<number, number>();
  let labelCounter = 0;

  for (let i = 0; i < n; i++) {
    let cur = i;
    while (parent[cur] !== -1) cur = parent[cur]!;
    let label = rootLabels.get(cur);
    if (label === undefined) {
      label = labelCounter++;
      rootLabels.set(cur, label);
    }
    labels[i] = label;
  }
  return labels;
}

/**
 * Converts a condensed distance matrix to a full (n×n) symmetric matrix.
 */
export function squareform(condensed: Float64Array, n: number): Float64Array {
  const full = new Float64Array(n * n);
  let k = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = condensed[k++]!;
      full[i * n + j] = d;
      full[j * n + i] = d;
    }
  }
  return full;
}

/**
 * Computes the cophenetic correlation coefficient for a linkage matrix.
 * Measures how faithfully the dendrogram preserves pairwise distances.
 */
export function copheneticCorr(
  rows: HierarchicalLinkageRow[],
  condensed: Float64Array,
  n: number,
): number {
  // Build cophenetic distance matrix from linkage
  const cophenetic = new Float64Array((n * (n - 1)) / 2);
  const clusterHeight = new Map<number, number>();
  const clusterMembers = new Map<number, number[]>();

  for (let i = 0; i < n; i++) clusterMembers.set(i, [i]);

  let nextId = n;
  for (const row of rows) {
    const mA = clusterMembers.get(row.idx1) ?? [];
    const mB = clusterMembers.get(row.idx2) ?? [];
    for (const a of mA) {
      for (const b of mB) {
        const [lo, hi] = a < b ? [a, b] : [b, a];
        // Condensed index
        const idx = lo * n - (lo * (lo + 1)) / 2 + hi - lo - 1;
        cophenetic[idx] = row.distance;
      }
    }
    clusterMembers.set(nextId, [...mA, ...mB]);
    clusterHeight.set(nextId, row.distance);
    nextId++;
  }

  // Pearson correlation between condensed and cophenetic distances
  const m = condensed.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < m; i++) {
    mx += condensed[i]!;
    my += cophenetic[i]!;
  }
  mx /= m;
  my /= m;
  let cov = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < m; i++) {
    const dx = condensed[i]! - mx;
    const dy = cophenetic[i]! - my;
    cov += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  const denom = Math.sqrt(sx * sy);
  return denom === 0 ? 0 : cov / denom;
}
