/**
 * Graph feature extraction utilities: GraphFeatureExtractor, adjacency operations
 */

export interface Graph {
  nodes: number[];
  edges: [number, number][];
  weights?: Map<string, number>;
}

export function graphToAdjacencyMatrix(graph: Graph): Float64Array[] {
  const n = graph.nodes.length;
  const nodeIndex = new Map(graph.nodes.map((node, i) => [node, i]));
  const adj = Array.from({ length: n }, () => new Float64Array(n));
  for (const [u, v] of graph.edges) {
    const i = nodeIndex.get(u) ?? 0;
    const j = nodeIndex.get(v) ?? 0;
    const w = graph.weights?.get(`${u},${v}`) ?? 1;
    adj[i]![j] = w;
    adj[j]![i] = w;
  }
  return adj;
}

export function computeDegrees(graph: Graph): Map<number, number> {
  const degrees = new Map<number, number>(graph.nodes.map((n) => [n, 0]));
  for (const [u, v] of graph.edges) {
    degrees.set(u, (degrees.get(u) ?? 0) + 1);
    degrees.set(v, (degrees.get(v) ?? 0) + 1);
  }
  return degrees;
}

export function computeClusteringCoefficients(
  graph: Graph,
): Map<number, number> {
  const adj = new Map<number, Set<number>>();
  for (const node of graph.nodes) adj.set(node, new Set());
  for (const [u, v] of graph.edges) {
    adj.get(u)?.add(v);
    adj.get(v)?.add(u);
  }
  const coeffs = new Map<number, number>();
  for (const node of graph.nodes) {
    const neighbors = [...(adj.get(node) ?? new Set())];
    const k = neighbors.length;
    if (k < 2) {
      coeffs.set(node, 0);
      continue;
    }
    let triangles = 0;
    for (let i = 0; i < neighbors.length; i++) {
      for (let j = i + 1; j < neighbors.length; j++) {
        if (adj.get(neighbors[i]!)?.has(neighbors[j]!)) triangles++;
      }
    }
    coeffs.set(node, (2 * triangles) / (k * (k - 1)));
  }
  return coeffs;
}

export class GraphFeatureExtractor {
  private features: string[];

  constructor(features: string[] = ["degree", "clustering"]) {
    this.features = features;
  }

  transform(graphs: Graph[]): Float64Array[] {
    return graphs.map((g) => {
      const featureVec: number[] = [];
      if (this.features.includes("degree")) {
        const degs = [...computeDegrees(g).values()];
        const mean = degs.reduce((a, b) => a + b, 0) / (degs.length || 1);
        featureVec.push(mean, Math.max(...degs, 0), Math.min(...degs, 0));
      }
      if (this.features.includes("clustering")) {
        const ccs = [...computeClusteringCoefficients(g).values()];
        const mean = ccs.reduce((a, b) => a + b, 0) / (ccs.length || 1);
        featureVec.push(mean);
      }
      if (this.features.includes("density")) {
        const n = g.nodes.length;
        const maxEdges = (n * (n - 1)) / 2;
        featureVec.push(maxEdges > 0 ? g.edges.length / maxEdges : 0);
      }
      return new Float64Array(featureVec);
    });
  }
}

export function shortestPathBFS(
  graph: Graph,
  source: number,
): Map<number, number> {
  const adj = new Map<number, number[]>();
  for (const node of graph.nodes) adj.set(node, []);
  for (const [u, v] of graph.edges) {
    adj.get(u)?.push(v);
    adj.get(v)?.push(u);
  }
  const dist = new Map<number, number>();
  dist.set(source, 0);
  const queue = [source];
  let qi = 0;
  while (qi < queue.length) {
    const u = queue[qi++]!;
    for (const v of adj.get(u) ?? []) {
      if (!dist.has(v)) {
        dist.set(v, (dist.get(u) ?? 0) + 1);
        queue.push(v);
      }
    }
  }
  return dist;
}
