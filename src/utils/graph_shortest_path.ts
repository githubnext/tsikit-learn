/**
 * Graph shortest path utilities.
 * Mirrors scikit-learn's utils.graph_shortest_path.graph_shortest_path.
 */

/** Compute shortest paths between all pairs in a dense distance matrix using Floyd-Warshall. */
export function graphShortestPath(
  dist: Float64Array[],
  directed = true,
): Float64Array[] {
  const n = dist.length;
  const d: Float64Array[] = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
    for (let j = 0; j < n; j++) {
      row[j] = dist[i]?.[j] ?? Number.POSITIVE_INFINITY;
    }
    row[i] = 0;
    return row;
  });

  if (!directed) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const dij = d[i]?.[j] ?? Number.POSITIVE_INFINITY;
        const dji = d[j]?.[i] ?? Number.POSITIVE_INFINITY;
        const minVal = Math.min(dij, dji);
        if (d[i] !== undefined) d[i]![j] = minVal;
        if (d[j] !== undefined) d[j]![i] = minVal;
      }
    }
  }

  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const ik = d[i]?.[k] ?? Number.POSITIVE_INFINITY;
        const kj = d[k]?.[j] ?? Number.POSITIVE_INFINITY;
        const ij = d[i]?.[j] ?? Number.POSITIVE_INFINITY;
        if (ik + kj < ij && d[i] !== undefined) {
          d[i]![j] = ik + kj;
        }
      }
    }
  }

  return d;
}

/** Dijkstra's single-source shortest path on a sparse adjacency list. */
export function dijkstra(
  adjacency: Map<number, Array<[number, number]>>,
  source: number,
  nNodes: number,
): Float64Array {
  const dist = new Float64Array(nNodes).fill(Number.POSITIVE_INFINITY);
  dist[source] = 0;
  // Min-heap: [dist, node]
  const heap: Array<[number, number]> = [[0, source]];

  while (heap.length > 0) {
    heap.sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0));
    const entry = heap.shift()!;
    const d = entry[0]!;
    const u = entry[1]!;
    if (d > (dist[u] ?? Number.POSITIVE_INFINITY)) continue;
    const neighbors = adjacency.get(u) ?? [];
    for (const [v, w] of neighbors) {
      const newDist = d + w;
      if (newDist < (dist[v] ?? Number.POSITIVE_INFINITY)) {
        dist[v] = newDist;
        heap.push([newDist, v]);
      }
    }
  }

  return dist;
}
