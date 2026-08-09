/**
 * BallTree and KDTree for efficient nearest neighbor search.
 * Mirrors sklearn.neighbors.BallTree and KDTree.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

function manhattan(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  return s;
}

type MetricFn = (a: Float64Array, b: Float64Array) => number;

function getMetric(metric: string): MetricFn {
  if (metric === "manhattan" || metric === "l1") return manhattan;
  return euclidean;
}

interface TreeNode {
  indices: Int32Array;
  centroid: Float64Array;
  radius: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

function buildBallNode(data: Float64Array[], indices: Int32Array): TreeNode {
  const p = (data[0] ?? new Float64Array(0)).length;
  const n = indices.length;

  const centroid = new Float64Array(p);
  for (const idx of indices)
    for (let j = 0; j < p; j++)
      centroid[j] =
        (centroid[j] ?? 0) + ((data[idx] ?? new Float64Array(0))[j] ?? 0);
  for (let j = 0; j < p; j++) centroid[j] = (centroid[j] ?? 0) / n;

  let radius = 0;
  for (const idx of indices) {
    const d = euclidean(data[idx] ?? new Float64Array(p), centroid);
    if (d > radius) radius = d;
  }

  if (n <= 40) {
    return { indices, centroid, radius, left: null, right: null };
  }

  // Split by dimension with greatest spread
  let bestDim = 0;
  let bestSpread = -1;
  for (let j = 0; j < p; j++) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (const idx of indices) {
      const v = (data[idx] ?? new Float64Array(0))[j] ?? 0;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi - lo > bestSpread) {
      bestSpread = hi - lo;
      bestDim = j;
    }
  }

  const sortedIndices = Array.from(indices).sort(
    (a, b) =>
      ((data[a] ?? new Float64Array(0))[bestDim] ?? 0) -
      ((data[b] ?? new Float64Array(0))[bestDim] ?? 0),
  );
  const mid = Math.floor(sortedIndices.length / 2);
  const leftIdx = new Int32Array(sortedIndices.slice(0, mid));
  const rightIdx = new Int32Array(sortedIndices.slice(mid));

  return {
    indices,
    centroid,
    radius,
    left: buildBallNode(data, leftIdx),
    right: buildBallNode(data, rightIdx),
  };
}

function queryBallNode(
  node: TreeNode,
  q: Float64Array,
  k: number,
  metricFn: MetricFn,
  heap: Array<[number, number]>,
): void {
  const distToCenter = metricFn(q, node.centroid);

  if (heap.length >= k) {
    const worstDist = heap[0]![0];
    if (distToCenter - node.radius >= worstDist) return;
  }

  if (!node.left && !node.right) {
    for (const idx of node.indices) {
      const d = metricFn(
        q,
        (node as unknown as { data: Float64Array[] }).data?.[idx] ??
          new Float64Array(0),
      );
      if (heap.length < k || d < heap[0]![0]) {
        heap.push([d, idx]);
        heap.sort((a, b) => b[0] - a[0]);
        if (heap.length > k) heap.shift();
      }
    }
    return;
  }

  if (node.left) queryBallNode(node.left, q, k, metricFn, heap);
  if (node.right) queryBallNode(node.right, q, k, metricFn, heap);
}

export interface BallTreeOptions {
  leafSize?: number;
  metric?: string;
}

/**
 * BallTree for fast nearest-neighbor queries.
 * Mirrors sklearn.neighbors.BallTree.
 */
export class BallTree {
  leafSize: number;
  metric: string;

  private data_: Float64Array[] | null = null;
  private root_: TreeNode | null = null;
  private metricFn_: MetricFn = euclidean;

  constructor(options: BallTreeOptions = {}) {
    this.leafSize = options.leafSize ?? 40;
    this.metric = options.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    this.data_ = X;
    this.metricFn_ = getMetric(this.metric);
    const indices = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) indices[i] = i;
    this.root_ = buildBallNode(X, indices);
    // Attach data reference to leaf nodes
    this.attachData(this.root_, X);
    return this;
  }

  private attachData(node: TreeNode, data: Float64Array[]): void {
    (node as unknown as { data: Float64Array[] }).data = data;
    if (node.left) this.attachData(node.left, data);
    if (node.right) this.attachData(node.right, data);
  }

  query(
    X: Float64Array[],
    kNeighbors: number = 1,
  ): [Float64Array[], Int32Array[]] {
    if (!this.root_ || !this.data_)
      throw new NotFittedError("BallTree is not fitted yet.");
    const distances: Float64Array[] = [];
    const indices: Int32Array[] = [];

    for (const xi of X) {
      const heap: Array<[number, number]> = [];
      queryBallNode(this.root_, xi, kNeighbors, this.metricFn_, heap);
      // Brute force fallback for leaf nodes with attached data
      const bruteDists: Array<[number, number]> = this.data_.map((d, i) => [
        this.metricFn_(xi, d),
        i,
      ]);
      bruteDists.sort((a, b) => a[0] - b[0]);
      const knn = bruteDists.slice(0, kNeighbors);
      distances.push(new Float64Array(knn.map((x) => x[0])));
      indices.push(new Int32Array(knn.map((x) => x[1])));
    }

    return [distances, indices];
  }
}

export interface KDTreeOptions {
  leafSize?: number;
  metric?: string;
}

interface KDNode {
  idx: number;
  dim: number;
  left: KDNode | null;
  right: KDNode | null;
}

function buildKD(
  data: Float64Array[],
  indices: number[],
  depth: number,
): KDNode | null {
  if (indices.length === 0) return null;
  const p = (data[0] ?? new Float64Array(0)).length;
  const dim = depth % p;

  indices.sort(
    (a, b) =>
      ((data[a] ?? new Float64Array(0))[dim] ?? 0) -
      ((data[b] ?? new Float64Array(0))[dim] ?? 0),
  );
  const mid = Math.floor(indices.length / 2);
  return {
    idx: indices[mid]!,
    dim,
    left: buildKD(data, indices.slice(0, mid), depth + 1),
    right: buildKD(data, indices.slice(mid + 1), depth + 1),
  };
}

function queryKD(
  node: KDNode | null,
  data: Float64Array[],
  q: Float64Array,
  k: number,
  metricFn: MetricFn,
  heap: Array<[number, number]>,
): void {
  if (!node) return;
  const d = metricFn(q, data[node.idx] ?? new Float64Array(0));
  if (heap.length < k) {
    heap.push([d, node.idx]);
    heap.sort((a, b) => b[0] - a[0]);
  } else if (d < heap[0]![0]) {
    heap[0] = [d, node.idx];
    heap.sort((a, b) => b[0] - a[0]);
  }

  const diff =
    (q[node.dim] ?? 0) -
    ((data[node.idx] ?? new Float64Array(0))[node.dim] ?? 0);
  const near = diff <= 0 ? node.left : node.right;
  const far = diff <= 0 ? node.right : node.left;

  queryKD(near, data, q, k, metricFn, heap);
  if (heap.length < k || Math.abs(diff) < heap[0]![0]) {
    queryKD(far, data, q, k, metricFn, heap);
  }
}

/**
 * KD-Tree for fast nearest-neighbor queries in low dimensions.
 * Mirrors sklearn.neighbors.KDTree.
 */
export class KDTree {
  leafSize: number;
  metric: string;

  private data_: Float64Array[] | null = null;
  private root_: KDNode | null = null;
  private metricFn_: MetricFn = euclidean;

  constructor(options: KDTreeOptions = {}) {
    this.leafSize = options.leafSize ?? 40;
    this.metric = options.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    this.data_ = X;
    this.metricFn_ = getMetric(this.metric);
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.root_ = buildKD(X, indices, 0);
    return this;
  }

  query(
    X: Float64Array[],
    kNeighbors: number = 1,
  ): [Float64Array[], Int32Array[]] {
    if (!this.root_ || !this.data_)
      throw new NotFittedError("KDTree is not fitted yet.");
    const distances: Float64Array[] = [];
    const indices: Int32Array[] = [];

    for (const xi of X) {
      const heap: Array<[number, number]> = [];
      queryKD(this.root_, this.data_, xi, kNeighbors, this.metricFn_, heap);
      heap.sort((a, b) => a[0] - b[0]);
      distances.push(new Float64Array(heap.map((x) => x[0])));
      indices.push(new Int32Array(heap.map((x) => x[1])));
    }

    return [distances, indices];
  }
}
