/**
 * Neighbors extensions: Ball Tree with multiple metrics, Approximate Nearest Neighbors
 */

export type DistanceMetric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski' | 'cosine' | 'hamming';

export function computeDistance(a: Float64Array, b: Float64Array, metric: DistanceMetric, p: number = 2): number {
  switch (metric) {
    case 'euclidean': return Math.sqrt(a.reduce((s, v, j) => s + (v - (b[j] ?? 0)) ** 2, 0));
    case 'manhattan': return a.reduce((s, v, j) => s + Math.abs(v - (b[j] ?? 0)), 0);
    case 'chebyshev': return Math.max(...Array.from(a).map((v, j) => Math.abs(v - (b[j] ?? 0))));
    case 'minkowski': return Math.pow(a.reduce((s, v, j) => s + Math.pow(Math.abs(v - (b[j] ?? 0)), p), 0), 1 / p);
    case 'cosine': {
      const dot = a.reduce((s, v, j) => s + v * (b[j] ?? 0), 0);
      const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
      const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
      return 1 - dot / (na * nb + 1e-10);
    }
    case 'hamming': return a.reduce((s, v, j) => s + (v !== (b[j] ?? 0) ? 1 : 0), 0) / a.length;
    default: return Math.sqrt(a.reduce((s, v, j) => s + (v - (b[j] ?? 0)) ** 2, 0));
  }
}

interface BallTreeNode {
  center: Float64Array;
  radius: number;
  left?: BallTreeNode;
  right?: BallTreeNode;
  indices?: number[];
}

export class BallTreeExt {
  private root_: BallTreeNode | null = null;
  private XFit_: Float64Array[] = [];
  private fitted_ = false;

  constructor(private metric: DistanceMetric = 'euclidean', private leafSize: number = 40, private minkowskiP: number = 2) {}

  fit(X: Float64Array[]): this {
    this.XFit_ = X;
    this.root_ = this._buildTree(Array.from({ length: X.length }, (_, i) => i));
    this.fitted_ = true;
    return this;
  }

  private _buildTree(indices: number[]): BallTreeNode {
    const n = indices.length;
    const p = this.XFit_[0]?.length ?? 0;

    // Compute center
    const center = new Float64Array(p);
    for (const i of indices) for (let j = 0; j < p; j++) center[j] = (center[j] ?? 0) + (this.XFit_[i]?.[j] ?? 0) / n;

    // Compute radius
    const radius = Math.max(...indices.map(i => computeDistance(center, this.XFit_[i]!, this.metric, this.minkowskiP)));

    if (n <= this.leafSize) return { center, radius, indices };

    // Find dimension with max spread
    let maxSpread = 0, splitDim = 0;
    for (let j = 0; j < p; j++) {
      const vals = indices.map(i => this.XFit_[i]?.[j] ?? 0);
      const spread = Math.max(...vals) - Math.min(...vals);
      if (spread > maxSpread) { maxSpread = spread; splitDim = j; }
    }

    // Split at median
    const sorted = [...indices].sort((a, b) => (this.XFit_[a]?.[splitDim] ?? 0) - (this.XFit_[b]?.[splitDim] ?? 0));
    const mid = Math.floor(n / 2);
    return {
      center, radius,
      left: this._buildTree(sorted.slice(0, mid)),
      right: this._buildTree(sorted.slice(mid))
    };
  }

  queryRadius(point: Float64Array, radius: number): { indices: number[]; distances: Float64Array } {
    if (!this.fitted_ || !this.root_) throw new Error('Not fitted');
    const result: Array<{ idx: number; dist: number }> = [];
    this._searchRadius(this.root_, point, radius, result);
    result.sort((a, b) => a.dist - b.dist);
    return { indices: result.map(r => r.idx), distances: new Float64Array(result.map(r => r.dist)) };
  }

  private _searchRadius(node: BallTreeNode, point: Float64Array, radius: number, result: Array<{ idx: number; dist: number }>): void {
    const distToCenter = computeDistance(point, node.center, this.metric, this.minkowskiP);
    if (distToCenter - node.radius > radius) return;

    if (node.indices) {
      for (const i of node.indices) {
        const d = computeDistance(point, this.XFit_[i]!, this.metric, this.minkowskiP);
        if (d <= radius) result.push({ idx: i, dist: d });
      }
      return;
    }
    if (node.left) this._searchRadius(node.left, point, radius, result);
    if (node.right) this._searchRadius(node.right, point, radius, result);
  }

  kNeighbors(X: Float64Array[], k: number): { indices: Int32Array[]; distances: Float64Array[] } {
    if (!this.fitted_ || !this.root_) throw new Error('Not fitted');
    return {
      indices: X.map(point => {
        const heap = this._knnSearch(this.root_!, point, k);
        return new Int32Array(heap.map(h => h.idx));
      }),
      distances: X.map(point => {
        const heap = this._knnSearch(this.root_!, point, k);
        return new Float64Array(heap.map(h => h.dist));
      })
    };
  }

  private _knnSearch(node: BallTreeNode, point: Float64Array, k: number): Array<{ idx: number; dist: number }> {
    const heap: Array<{ idx: number; dist: number }> = [];
    const addToHeap = (idx: number, dist: number) => {
      heap.push({ idx, dist });
      heap.sort((a, b) => b.dist - a.dist);
      if (heap.length > k) heap.pop();
    };
    const search = (n: BallTreeNode) => {
      const distToCenter = computeDistance(point, n.center, this.metric, this.minkowskiP);
      const worstDist = heap.length >= k ? (heap[0]?.dist ?? Number.POSITIVE_INFINITY) : Number.POSITIVE_INFINITY;
      if (distToCenter - n.radius > worstDist) return;

      if (n.indices) {
        for (const i of n.indices) addToHeap(i, computeDistance(point, this.XFit_[i]!, this.metric, this.minkowskiP));
        return;
      }
      if (n.left) search(n.left);
      if (n.right) search(n.right);
    };
    search(node);
    return heap.reverse();
  }
}

export class LSHApproximateNN {
  private tables_: Array<{ planes: Float64Array[]; buckets: Map<string, number[]> }> = [];
  private XFit_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private nTables: number = 10,
    private nBits: number = 8,
    private randomState: number = 42
  ) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    let rng = this.randomState;
    const randNorm = () => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      const u1 = rng / 0xffffffff;
      rng = (rng * 1664525 + 1013904223) >>> 0;
      const u2 = rng / 0xffffffff;
      return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
    };

    this.tables_ = [];
    for (let t = 0; t < this.nTables; t++) {
      const planes = Array.from({ length: this.nBits }, () => new Float64Array(p).map(() => randNorm()));
      const buckets = new Map<string, number[]>();
      for (let i = 0; i < X.length; i++) {
        const code = planes.map(plane => plane.reduce((s, v, j) => s + v * (X[i]?.[j] ?? 0), 0) > 0 ? '1' : '0').join('');
        const bucket = buckets.get(code) ?? [];
        bucket.push(i);
        buckets.set(code, bucket);
      }
      this.tables_.push({ planes, buckets });
    }
    this.XFit_ = X;
    this.fitted_ = true;
    return this;
  }

  queryApproximate(point: Float64Array, k: number = 5): { indices: Int32Array; distances: Float64Array } {
    if (!this.fitted_) throw new Error('Not fitted');
    const candidates = new Set<number>();
    for (const { planes, buckets } of this.tables_) {
      const code = planes.map(plane => plane.reduce((s, v, j) => s + v * (point[j] ?? 0), 0) > 0 ? '1' : '0').join('');
      for (const idx of (buckets.get(code) ?? [])) candidates.add(idx);
    }

    const scored = [...candidates].map(i => ({
      idx: i,
      dist: Math.sqrt((this.XFit_[i]!).reduce((s, v, j) => s + (v - (point[j] ?? 0)) ** 2, 0))
    })).sort((a, b) => a.dist - b.dist).slice(0, k);

    return { indices: new Int32Array(scored.map(s => s.idx)), distances: new Float64Array(scored.map(s => s.dist)) };
  }
}
