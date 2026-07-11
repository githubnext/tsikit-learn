/**
 * Approximate Nearest Neighbors and BallTree extension.
 */

export class BallTree {
  leafSize: number;
  private _root: BallTreeNode | null = null;
  private _X: Float64Array[] = [];

  constructor(leafSize = 40) {
    this.leafSize = leafSize;
  }

  fit(X: Float64Array[]): this {
    this._X = X;
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this._root = this._build(indices);
    return this;
  }

  private _centroid(indices: number[]): Float64Array {
    const p = this._X[0]?.length ?? 0;
    const c = new Float64Array(p);
    for (const i of indices) {
      const row = this._X[i] as Float64Array;
      for (let d = 0; d < p; d++) c[d] = (c[d] ?? 0) + (row[d] ?? 0) / indices.length;
    }
    return c;
  }

  private _radius(centroid: Float64Array, indices: number[]): number {
    let r = 0;
    for (const i of indices) {
      let d2 = 0;
      const row = this._X[i] as Float64Array;
      for (let d = 0; d < centroid.length; d++) d2 += ((row[d] ?? 0) - (centroid[d] ?? 0)) ** 2;
      r = Math.max(r, Math.sqrt(d2));
    }
    return r;
  }

  private _build(indices: number[]): BallTreeNode {
    const centroid = this._centroid(indices);
    const radius = this._radius(centroid, indices);
    if (indices.length <= this.leafSize) {
      return { centroid, radius, indices, left: null, right: null };
    }
    // Split on the dimension with maximum spread
    const p = centroid.length;
    let bestDim = 0, bestSpread = -1;
    for (let d = 0; d < p; d++) {
      const vals = indices.map((i) => this._X[i]?.[d] ?? 0);
      const spread = Math.max(...vals) - Math.min(...vals);
      if (spread > bestSpread) { bestSpread = spread; bestDim = d; }
    }
    const median = indices.map((i) => this._X[i]?.[bestDim] ?? 0).sort((a, b) => a - b)[Math.floor(indices.length / 2)] ?? 0;
    const leftIdx = indices.filter((i) => (this._X[i]?.[bestDim] ?? 0) <= median);
    const rightIdx = indices.filter((i) => (this._X[i]?.[bestDim] ?? 0) > median);
    if (leftIdx.length === 0 || rightIdx.length === 0) {
      return { centroid, radius, indices, left: null, right: null };
    }
    return {
      centroid,
      radius,
      indices: null,
      left: this._build(leftIdx),
      right: this._build(rightIdx),
    };
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let d2 = 0;
    for (let i = 0; i < a.length; i++) d2 += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.sqrt(d2);
  }

  query(X: Float64Array[], kNeighbors = 1): { indices: Int32Array[]; distances: Float64Array[] } {
    const indices: Int32Array[] = [];
    const distances: Float64Array[] = [];
    for (const x of X) {
      const heap: Array<{ dist: number; idx: number }> = [];
      this._queryNode(this._root, x, kNeighbors, heap);
      heap.sort((a, b) => a.dist - b.dist);
      indices.push(Int32Array.from(heap.slice(0, kNeighbors), (e) => e.idx));
      distances.push(Float64Array.from(heap.slice(0, kNeighbors), (e) => e.dist));
    }
    return { indices, distances };
  }

  private _queryNode(node: BallTreeNode | null, x: Float64Array, k: number, heap: Array<{ dist: number; idx: number }>): void {
    if (!node) return;
    const ballDist = this._dist(x, node.centroid) - node.radius;
    const worstInHeap = heap.length >= k ? (heap[heap.length - 1]?.dist ?? Number.POSITIVE_INFINITY) : Number.POSITIVE_INFINITY;
    if (ballDist >= worstInHeap) return;

    if (node.indices !== null) {
      for (const i of node.indices) {
        const d = this._dist(x, this._X[i] as Float64Array);
        if (heap.length < k || d < (heap[heap.length - 1]?.dist ?? Number.POSITIVE_INFINITY)) {
          heap.push({ dist: d, idx: i });
          heap.sort((a, b) => a.dist - b.dist);
          if (heap.length > k) heap.pop();
        }
      }
    } else {
      this._queryNode(node.left, x, k, heap);
      this._queryNode(node.right, x, k, heap);
    }
  }
}

interface BallTreeNode {
  centroid: Float64Array;
  radius: number;
  indices: number[] | null;
  left: BallTreeNode | null;
  right: BallTreeNode | null;
}

export class KDTree {
  leafSize: number;
  private _root: KDNode | null = null;
  private _X: Float64Array[] = [];

  constructor(leafSize = 40) {
    this.leafSize = leafSize;
  }

  fit(X: Float64Array[]): this {
    this._X = X;
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this._root = this._build(indices, 0);
    return this;
  }

  private _build(indices: number[], depth: number): KDNode {
    if (indices.length <= this.leafSize) return { splitDim: -1, splitVal: 0, indices, left: null, right: null };
    const dim = depth % (this._X[0]?.length ?? 1);
    const sorted = [...indices].sort((a, b) => (this._X[a]?.[dim] ?? 0) - (this._X[b]?.[dim] ?? 0));
    const mid = Math.floor(sorted.length / 2);
    const splitVal = this._X[sorted[mid] ?? 0]?.[dim] ?? 0;
    return {
      splitDim: dim,
      splitVal,
      indices: null,
      left: this._build(sorted.slice(0, mid), depth + 1),
      right: this._build(sorted.slice(mid), depth + 1),
    };
  }

  query(X: Float64Array[], kNeighbors = 1): { indices: Int32Array[]; distances: Float64Array[] } {
    return {
      indices: X.map((x) => {
        const heap: Array<{ dist: number; idx: number }> = [];
        this._queryNode(this._root, x, kNeighbors, heap);
        heap.sort((a, b) => a.dist - b.dist);
        return Int32Array.from(heap.slice(0, kNeighbors), (e) => e.idx);
      }),
      distances: X.map((x) => {
        const heap: Array<{ dist: number; idx: number }> = [];
        this._queryNode(this._root, x, kNeighbors, heap);
        heap.sort((a, b) => a.dist - b.dist);
        return Float64Array.from(heap.slice(0, kNeighbors), (e) => e.dist);
      }),
    };
  }

  private _queryNode(node: KDNode | null, x: Float64Array, k: number, heap: Array<{ dist: number; idx: number }>): void {
    if (!node) return;
    if (node.indices !== null) {
      for (const i of node.indices) {
        let d2 = 0;
        for (let d = 0; d < x.length; d++) d2 += ((x[d] ?? 0) - ((this._X[i] as Float64Array)[d] ?? 0)) ** 2;
        const dist = Math.sqrt(d2);
        if (heap.length < k || dist < (heap[heap.length - 1]?.dist ?? Number.POSITIVE_INFINITY)) {
          heap.push({ dist, idx: i });
          heap.sort((a, b) => a.dist - b.dist);
          if (heap.length > k) heap.pop();
        }
      }
      return;
    }
    const diff = (x[node.splitDim] ?? 0) - node.splitVal;
    const primary = diff <= 0 ? node.left : node.right;
    const secondary = diff <= 0 ? node.right : node.left;
    this._queryNode(primary, x, k, heap);
    const worstDist = heap.length >= k ? (heap[heap.length - 1]?.dist ?? Number.POSITIVE_INFINITY) : Number.POSITIVE_INFINITY;
    if (Math.abs(diff) < worstDist) this._queryNode(secondary, x, k, heap);
  }
}

interface KDNode {
  splitDim: number;
  splitVal: number;
  indices: number[] | null;
  left: KDNode | null;
  right: KDNode | null;
}

export class RadiusNeighborsTransformer {
  radius: number;
  private _X: Float64Array[] | null = null;

  constructor(radius = 1.0) {
    this.radius = radius;
  }

  fit(X: Float64Array[]): this {
    this._X = X;
    return this;
  }

  transform(X: Float64Array[]): Array<Array<{ index: number; distance: number }>> {
    if (!this._X) throw new Error("Not fitted");
    return X.map((x) => {
      const neighbors: Array<{ index: number; distance: number }> = [];
      for (let i = 0; i < (this._X as Float64Array[]).length; i++) {
        let d2 = 0;
        for (let d = 0; d < x.length; d++) d2 += ((x[d] ?? 0) - ((this._X![i] as Float64Array)[d] ?? 0)) ** 2;
        const dist = Math.sqrt(d2);
        if (dist <= this.radius) neighbors.push({ index: i, distance: dist });
      }
      return neighbors.sort((a, b) => a.distance - b.distance);
    });
  }

  fitTransform(X: Float64Array[]): Array<Array<{ index: number; distance: number }>> {
    return this.fit(X).transform(X);
  }
}
