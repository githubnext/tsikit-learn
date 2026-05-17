/**
 * Ward linkage and hierarchical clustering utilities.
 * Mirrors scipy.cluster.hierarchy (linkage, fcluster, dendrogram helpers)
 * as used within sklearn.cluster.AgglomerativeClustering.
 */

export interface LinkageRow {
  clusterA: number;
  clusterB: number;
  distance: number;
  size: number;
}

/** Compute the Ward linkage matrix for a dataset (O(n^3) naive implementation). */
export function wardLinkage(X: Float64Array[]): LinkageRow[] {
  const n = X.length;
  if (n < 2) return [];

  // Each point starts as its own cluster
  const clusterPoints: Map<number, number[]> = new Map();
  for (let i = 0; i < n; i++) clusterPoints.set(i, [i]);

  // Current cluster centroids
  const centroids: Map<number, Float64Array> = new Map();
  for (let i = 0; i < n; i++) centroids.set(i, new Float64Array(X[i]!));

  let nextCluster = n;
  const result: LinkageRow[] = [];
  const activeClusters = new Set<number>(Array.from({ length: n }, (_, i) => i));

  function centroid(indices: number[]): Float64Array {
    const d = X[0]!.length;
    const c = new Float64Array(d);
    for (const idx of indices) {
      const pt = X[idx]!;
      for (let j = 0; j < d; j++) c[j]! += pt[j] ?? 0;
    }
    for (let j = 0; j < d; j++) c[j]! /= indices.length;
    return c;
  }

  function wardDist(a: number, b: number): number {
    const pa = clusterPoints.get(a)!;
    const pb = clusterPoints.get(b)!;
    const na = pa.length;
    const nb = pb.length;
    const ca = centroids.get(a)!;
    const cb = centroids.get(b)!;
    let sq = 0;
    for (let j = 0; j < ca.length; j++) {
      const diff = (ca[j] ?? 0) - (cb[j] ?? 0);
      sq += diff * diff;
    }
    return Math.sqrt((na * nb) / (na + nb) * sq);
  }

  while (activeClusters.size > 1) {
    // Find closest pair
    const active = [...activeClusters];
    let minDist = Number.POSITIVE_INFINITY;
    let bestA = -1;
    let bestB = -1;
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const d = wardDist(active[i]!, active[j]!);
        if (d < minDist) { minDist = d; bestA = active[i]!; bestB = active[j]!; }
      }
    }

    const pA = clusterPoints.get(bestA)!;
    const pB = clusterPoints.get(bestB)!;
    const merged = [...pA, ...pB];
    clusterPoints.set(nextCluster, merged);
    centroids.set(nextCluster, centroid(merged));

    result.push({ clusterA: bestA, clusterB: bestB, distance: minDist, size: merged.length });
    activeClusters.delete(bestA);
    activeClusters.delete(bestB);
    activeClusters.add(nextCluster);
    nextCluster++;
  }

  return result;
}

/** Flatten the linkage matrix to cluster labels (fcluster with criterion='maxclust'). */
export function fcluster(linkage: LinkageRow[], nClusters: number, nPoints: number): Int32Array {
  const labels = new Int32Array(nPoints);
  if (nClusters >= nPoints) { for (let i = 0; i < nPoints; i++) labels[i] = i; return labels; }

  // Track which top-level cluster each point belongs to
  const children: Map<number, [number, number]> = new Map();
  for (const row of linkage) {
    children.set(nPoints + children.size, [row.clusterA, row.clusterB]);
  }

  // The root is the last merged cluster
  const root = nPoints + linkage.length - 1;
  // BFS to assign labels — cut the tree to produce nClusters clusters
  const cutAt = linkage.length - nClusters; // cut after this many merges from the root
  const mergeCount = linkage.length;
  const cutThreshold = mergeCount >= nClusters ? linkage[mergeCount - nClusters]?.distance ?? 0 : 0;

  // Assign label by DFS
  let nextLabel = 0;
  function assign(node: number, label: number): void {
    if (node < nPoints) { labels[node] = label; return; }
    const ch = children.get(node);
    if (!ch) return;
    assign(ch[0], label);
    assign(ch[1], label);
  }

  // Walk from root, splitting where distance > cutThreshold
  function split(node: number, rowIdx: number): void {
    if (node < nPoints) { labels[node] = nextLabel++; return; }
    const ch = children.get(node);
    if (!ch) { assign(node, nextLabel++); return; }
    const row = linkage[rowIdx];
    if (!row) { assign(node, nextLabel++); return; }
    if (row.distance > cutThreshold && nextLabel < nClusters) {
      split(ch[0], rowIdx - 1 - (linkage.length - 1 - rowIdx));
      split(ch[1], rowIdx - 1);
    } else {
      assign(node, nextLabel++);
    }
  }

  // Simple BFS approach: top nClusters nodes in the linkage
  const queue: number[] = [root];
  const clusters: number[] = [];
  let label = 0;
  while (clusters.length < nClusters && queue.length > 0) {
    const node = queue.shift()!;
    const ch = children.get(node);
    if (!ch || clusters.length + queue.length >= nClusters) {
      clusters.push(node);
    } else {
      queue.push(ch[0], ch[1]);
    }
  }
  for (const cl of clusters) assign(cl, label++);

  return labels;
}

/** Compute cophenetic distances from linkage matrix. */
export function copheneticDistances(linkage: LinkageRow[], nPoints: number): Float64Array {
  const n = nPoints;
  const dist = new Float64Array(n * n);
  // For each pair of points, find when they first merge
  function findMerge(a: number, b: number): number {
    // Walk through linkage in order
    const clusterOf = new Int32Array(nPoints + linkage.length);
    for (let i = 0; i < nPoints; i++) clusterOf[i] = i;
    for (let step = 0; step < linkage.length; step++) {
      const row = linkage[step]!;
      const newId = nPoints + step;
      // Check if a and b are in clusterA and clusterB
      const inA = isIn(a, row.clusterA, nPoints, linkage, step);
      const inB = isIn(b, row.clusterB, nPoints, linkage, step);
      const inBA = isIn(b, row.clusterA, nPoints, linkage, step);
      const inAB = isIn(a, row.clusterB, nPoints, linkage, step);
      if ((inA && inB) || (inBA && inAB)) return row.distance;
    }
    return 0;
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = findMerge(i, j);
      dist[i * n + j] = d; dist[j * n + i] = d;
    }
  }
  return dist;
}

function isIn(point: number, cluster: number, nPoints: number, linkage: LinkageRow[], upTo: number): boolean {
  if (cluster === point) return true;
  if (cluster < nPoints) return false;
  const idx = cluster - nPoints;
  if (idx >= upTo) return false;
  const row = linkage[idx]!;
  return isIn(point, row.clusterA, nPoints, linkage, idx) || isIn(point, row.clusterB, nPoints, linkage, idx);
}

export type { LinkageRow as WardLinkageRow };
