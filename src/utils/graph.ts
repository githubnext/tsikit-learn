/**
 * Graph utilities: connected components, minimum spanning tree, shortest paths.
 * Used internally by manifold learning and clustering algorithms.
 * Mirrors sklearn.utils.graph and scipy.sparse.csgraph utilities.
 */

/** Adjacency list representation of a weighted graph. */
export interface Graph {
  n: number;
  edges: Array<{ u: number; v: number; w: number }>;
}

/** Union-Find (Disjoint Set Union) data structure. */
export class UnionFind {
  parent: Int32Array;
  rank: Int32Array;

  constructor(n: number) {
    this.parent = new Int32Array(n);
    this.rank = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }

  find(x: number): number {
    let cur = x;
    while (this.parent[cur] !== cur) {
      this.parent[cur] = this.parent[this.parent[cur] ?? cur] ?? cur;
      cur = this.parent[cur] ?? cur;
    }
    return cur;
  }

  union(x: number, y: number): boolean {
    const px = this.find(x);
    const py = this.find(y);
    if (px === py) return false;
    if ((this.rank[px] ?? 0) < (this.rank[py] ?? 0)) {
      this.parent[px] = py;
    } else if ((this.rank[px] ?? 0) > (this.rank[py] ?? 0)) {
      this.parent[py] = px;
    } else {
      this.parent[py] = px;
      this.rank[px] = (this.rank[px] ?? 0) + 1;
    }
    return true;
  }
}

/**
 * Find connected components in an undirected graph.
 * Returns component label for each node (0-indexed component IDs).
 */
export function connectedComponents(adjacency: Float64Array[]): {
  nComponents: number;
  labels: Int32Array;
} {
  const n = adjacency.length;
  const uf = new UnionFind(n);
  for (let i = 0; i < n; i++) {
    const row = adjacency[i] ?? new Float64Array(n);
    for (let j = i + 1; j < n; j++) {
      if ((row[j] ?? 0) > 0) uf.union(i, j);
    }
  }
  const labels = new Int32Array(n);
  const compMap = new Map<number, number>();
  let nComp = 0;
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    if (!compMap.has(root)) compMap.set(root, nComp++);
    labels[i] = compMap.get(root)!;
  }
  return { nComponents: nComp, labels };
}

/**
 * Minimum spanning tree via Kruskal's algorithm.
 * Returns list of edges in the MST.
 */
export function minimumSpanningTree(
  adjacency: Float64Array[],
): Array<{ u: number; v: number; w: number }> {
  const n = adjacency.length;
  const edges: Array<{ u: number; v: number; w: number }> = [];
  for (let i = 0; i < n; i++) {
    const row = adjacency[i] ?? new Float64Array(n);
    for (let j = i + 1; j < n; j++) {
      const w = row[j] ?? 0;
      if (w > 0) edges.push({ u: i, v: j, w });
    }
  }
  edges.sort((a, b) => a.w - b.w);

  const uf = new UnionFind(n);
  const mst: Array<{ u: number; v: number; w: number }> = [];
  for (const { u, v, w } of edges) {
    if (uf.union(u, v)) mst.push({ u, v, w });
    if (mst.length === n - 1) break;
  }
  return mst;
}

/**
 * Single-source shortest paths via Dijkstra's algorithm.
 * Returns distances from source to all other nodes.
 */
export function dijkstra(
  adjacency: Float64Array[],
  source: number,
): Float64Array {
  const n = adjacency.length;
  const dist = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
  dist[source] = 0;
  const visited = new Uint8Array(n);

  for (let iter = 0; iter < n; iter++) {
    // Find min-distance unvisited node
    let u = -1;
    let minDist = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      if (!visited[i] && (dist[i] ?? Number.POSITIVE_INFINITY) < minDist) {
        minDist = dist[i] ?? Number.POSITIVE_INFINITY;
        u = i;
      }
    }
    if (u === -1) break;
    visited[u] = 1;

    const row = adjacency[u] ?? new Float64Array(n);
    for (let v = 0; v < n; v++) {
      const w = row[v] ?? 0;
      if (w > 0 && !visited[v]) {
        const newDist = (dist[u] ?? 0) + w;
        if (newDist < (dist[v] ?? Number.POSITIVE_INFINITY)) dist[v] = newDist;
      }
    }
  }
  return dist;
}

/**
 * All-pairs shortest paths via Floyd-Warshall.
 * Returns distance matrix.
 */
export function shortestPaths(adjacency: Float64Array[]): Float64Array[] {
  const n = adjacency.length;
  // Initialize with adjacency (0 on diagonal, Infinity where no edge)
  const dist = adjacency.map(
    (row, i) =>
      new Float64Array(
        row.map((v, j) => {
          if (i === j) return 0;
          return v > 0 ? v : Number.POSITIVE_INFINITY;
        }),
      ),
  );

  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const via =
          (dist[i]![k] ?? Number.POSITIVE_INFINITY) +
          (dist[k]![j] ?? Number.POSITIVE_INFINITY);
        if (via < (dist[i]![j] ?? Number.POSITIVE_INFINITY)) dist[i]![j] = via;
      }
    }
  }
  return dist;
}

/**
 * Compute graph Laplacian (normalized or unnormalized).
 * Used by spectral methods.
 */
export function graphLaplacian(
  adjacency: Float64Array[],
  options: { normalized?: boolean } = {},
): Float64Array[] {
  const n = adjacency.length;
  const { normalized = false } = options;

  // Degree matrix
  const degree = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = adjacency[i] ?? new Float64Array(n);
    for (let j = 0; j < n; j++) degree[i] = (degree[i] ?? 0) + (row[j] ?? 0);
  }

  const L = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    const adjRow = adjacency[i] ?? new Float64Array(n);
    row[i] = degree[i] ?? 0;
    for (let j = 0; j < n; j++) if (i !== j) row[j] = -(adjRow[j] ?? 0);
    return row;
  });

  if (!normalized) return L;

  // Normalized Laplacian: D^{-1/2} L D^{-1/2}
  const dInvSqrt = new Float64Array(n).map((_, i) => {
    const d = degree[i] ?? 0;
    return d > 0 ? 1 / Math.sqrt(d) : 0;
  });
  return L.map(
    (row, i) =>
      new Float64Array(
        row.map((v, j) => v * (dInvSqrt[i] ?? 0) * (dInvSqrt[j] ?? 0)),
      ),
  );
}

/**
 * Build a k-nearest-neighbors graph from a distance matrix.
 * Returns an adjacency matrix (symmetric).
 */
export function kneighborsGraph(
  distances: Float64Array[],
  k: number,
  mode: "connectivity" | "distance" = "connectivity",
): Float64Array[] {
  const n = distances.length;
  const adj = Array.from({ length: n }, () => new Float64Array(n));

  for (let i = 0; i < n; i++) {
    const row = distances[i] ?? new Float64Array(n);
    const sorted = Array.from({ length: n }, (_, j) => ({ j, d: row[j] ?? 0 }))
      .filter((x) => x.j !== i)
      .sort((a, b) => a.d - b.d);

    for (let ki = 0; ki < Math.min(k, sorted.length); ki++) {
      const { j, d } = sorted[ki] ?? { j: 0, d: 0 };
      const val = mode === "connectivity" ? 1 : d;
      adj[i]![j] = val;
      adj[j]![i] = val;
    }
  }
  return adj;
}
