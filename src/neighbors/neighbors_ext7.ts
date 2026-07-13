/**
 * Neighbors extensions: NearestCentroidExt, AnnoyLikeIndex, ProductQuantizerExt
 * Port of sklearn.neighbors extensions
 */

import { NotFittedError } from "../exceptions.js";

export class ApproximateNearestNeighbors {
  nNeighbors: number;
  nTrees: number;
  randomState: number;
  metric: "euclidean" | "cosine" | "manhattan";

  private trees_: Array<{
    root: TreeNode;
    data: Float64Array[];
  }> | null = null;

  constructor(opts: {
    nNeighbors?: number;
    nTrees?: number;
    randomState?: number;
    metric?: "euclidean" | "cosine" | "manhattan";
  } = {}) {
    this.nNeighbors = opts.nNeighbors ?? 10;
    this.nTrees = opts.nTrees ?? 10;
    this.randomState = opts.randomState ?? 42;
    this.metric = opts.metric ?? "euclidean";
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    if (this.metric === "cosine") {
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let j = 0; j < a.length; j++) { dot += (a[j] ?? 0) * (b[j] ?? 0); na += (a[j] ?? 0) ** 2; nb += (b[j] ?? 0) ** 2; }
      return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-15);
    }
    if (this.metric === "manhattan") {
      let d = 0;
      for (let j = 0; j < a.length; j++) d += Math.abs((a[j] ?? 0) - (b[j] ?? 0));
      return d;
    }
    let d = 0;
    for (let j = 0; j < a.length; j++) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(d);
  }

  fit(X: Float64Array[]): this {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.trees_ = Array.from({ length: this.nTrees }, () => ({
      root: buildAnnoyTree(X, Array.from({ length: X.length }, (_, i) => i), rng, 0),
      data: X,
    }));
    return this;
  }

  kneighbors(X: Float64Array[]): { distances: Float64Array[]; indices: Int32Array[] } {
    if (!this.trees_) throw new NotFittedError("ApproximateNearestNeighbors not fitted.");
    const distances: Float64Array[] = [];
    const indices: Int32Array[] = [];
    for (const xi of X) {
      const candidates = new Set<number>();
      for (const tree of this.trees_) {
        searchAnnoyTree(tree.root, xi, this.nNeighbors * 2, candidates);
      }
      const scored = [...candidates].map(idx => ({ idx, dist: this._dist(xi, this.trees_![0]!.data[idx]!) }));
      scored.sort((a, b) => a.dist - b.dist);
      const k = Math.min(this.nNeighbors, scored.length);
      distances.push(Float64Array.from(scored.slice(0, k).map(s => s.dist)));
      indices.push(Int32Array.from(scored.slice(0, k).map(s => s.idx)));
    }
    return { distances, indices };
  }
}

interface TreeNode {
  splitFeat?: number;
  splitVal?: number;
  left?: TreeNode;
  right?: TreeNode;
  indices?: number[];
}

function buildAnnoyTree(X: Float64Array[], indices: number[], rng: () => number, depth: number): TreeNode {
  if (indices.length <= 10 || depth > 20) return { indices };
  const p = X[0]?.length ?? 0;
  const f = Math.floor(rng() * p);
  const vals = indices.map(i => X[i]![f] ?? 0);
  vals.sort((a, b) => a - b);
  const median = vals[Math.floor(vals.length / 2)] ?? 0;
  const left = indices.filter(i => (X[i]![f] ?? 0) <= median);
  const right = indices.filter(i => (X[i]![f] ?? 0) > median);
  if (left.length === 0 || right.length === 0) return { indices };
  return { splitFeat: f, splitVal: median, left: buildAnnoyTree(X, left, rng, depth + 1), right: buildAnnoyTree(X, right, rng, depth + 1) };
}

function searchAnnoyTree(node: TreeNode, query: Float64Array, k: number, result: Set<number>): void {
  if (node.indices) { for (const i of node.indices) result.add(i); return; }
  if (result.size >= k) return;
  const f = node.splitFeat ?? 0;
  const v = node.splitVal ?? 0;
  if ((query[f] ?? 0) <= v) {
    if (node.left) searchAnnoyTree(node.left, query, k, result);
    if (result.size < k && node.right) searchAnnoyTree(node.right, query, k, result);
  } else {
    if (node.right) searchAnnoyTree(node.right, query, k, result);
    if (result.size < k && node.left) searchAnnoyTree(node.left, query, k, result);
  }
}

export class ProductQuantizerExt {
  M: number;
  K: number;
  maxIter: number;
  randomState: number;

  private codebooks_: Float64Array[][] | null = null;
  private subDim_ = 0;
  private nFeatures_ = 0;

  constructor(opts: { M?: number; K?: number; maxIter?: number; randomState?: number } = {}) {
    this.M = opts.M ?? 8;
    this.K = opts.K ?? 256;
    this.maxIter = opts.maxIter ?? 20;
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;
    this.subDim_ = Math.ceil(this.nFeatures_ / this.M);
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.codebooks_ = Array.from({ length: this.M }, (_, m) => {
      const start = m * this.subDim_;
      const subX = X.map(xi => Float64Array.from({ length: this.subDim_ }, (__, j) => xi[start + j] ?? 0));
      const k = Math.min(this.K, n);
      let centroids = Array.from({ length: k }, () => subX[Math.floor(rng() * n)]!.slice());
      for (let iter = 0; iter < this.maxIter; iter++) {
        const counts = new Int32Array(k);
        const sums = Array.from({ length: k }, () => new Float64Array(this.subDim_));
        for (const xi of subX) {
          let best = 0;
          let bestDist = Number.POSITIVE_INFINITY;
          for (let c = 0; c < k; c++) {
            let d = 0;
            for (let j = 0; j < this.subDim_; j++) d += ((xi[j] ?? 0) - (centroids[c]![j] ?? 0)) ** 2;
            if (d < bestDist) { bestDist = d; best = c; }
          }
          counts[best]!++;
          for (let j = 0; j < this.subDim_; j++) sums[best]![j] = (sums[best]![j] ?? 0) + (xi[j] ?? 0);
        }
        centroids = centroids.map((_, c) => Float64Array.from({ length: this.subDim_ }, (__, j) => (sums[c]![j] ?? 0) / ((counts[c] ?? 1) + 1e-15)));
        void iter;
      }
      return centroids;
    });
    return this;
  }

  encode(X: Float64Array[]): Int32Array[] {
    if (!this.codebooks_) throw new NotFittedError("ProductQuantizerExt not fitted.");
    return X.map(xi => Int32Array.from({ length: this.M }, (_, m) => {
      const start = m * this.subDim_;
      const sub = Float64Array.from({ length: this.subDim_ }, (__, j) => xi[start + j] ?? 0);
      const cb = this.codebooks_![m]!;
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < cb.length; c++) {
        let d = 0;
        for (let j = 0; j < this.subDim_; j++) d += ((sub[j] ?? 0) - (cb[c]![j] ?? 0)) ** 2;
        if (d < bestDist) { bestDist = d; best = c; }
      }
      return best;
    }));
  }

  decode(codes: Int32Array[]): Float64Array[] {
    if (!this.codebooks_) throw new NotFittedError("ProductQuantizerExt not fitted.");
    return codes.map(code => {
      const xi = new Float64Array(this.nFeatures_);
      for (let m = 0; m < this.M; m++) {
        const start = m * this.subDim_;
        const cb = this.codebooks_![m]!;
        const c = code[m] ?? 0;
        const centroid = cb[c]!;
        for (let j = 0; j < this.subDim_; j++) xi[start + j] = centroid[j] ?? 0;
      }
      return xi;
    });
  }
}

export class NearestCentroidExt {
  metric: "euclidean" | "cosine";
  shrinkThreshold: number | null;

  centroids_: Float64Array[] | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: { metric?: "euclidean" | "cosine"; shrinkThreshold?: number } = {}) {
    this.metric = opts.metric ?? "euclidean";
    this.shrinkThreshold = opts.shrinkThreshold ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = new Set<number>();
    for (let i = 0; i < y.length; i++) classes.add(y[i] ?? 0);
    this.classes_ = Int32Array.from([...classes].sort((a, b) => a - b));
    const p = X[0]?.length ?? 0;
    this.centroids_ = this.classes_.map(c => {
      const sum = new Float64Array(p);
      let count = 0;
      for (let i = 0; i < y.length; i++) {
        if ((y[i] ?? 0) === c) {
          for (let j = 0; j < p; j++) sum[j] = (sum[j] ?? 0) + (X[i]![j] ?? 0);
          count++;
        }
      }
      for (let j = 0; j < p; j++) sum[j] = (sum[j] ?? 0) / (count + 1e-15);
      if (this.shrinkThreshold !== null) {
        const globalMean = new Float64Array(p);
        for (const xi of X) for (let j = 0; j < p; j++) globalMean[j] = (globalMean[j] ?? 0) + (xi[j] ?? 0) / X.length;
        for (let j = 0; j < p; j++) {
          const diff = (sum[j] ?? 0) - (globalMean[j] ?? 0);
          sum[j] = (globalMean[j] ?? 0) + Math.sign(diff) * Math.max(0, Math.abs(diff) - (this.shrinkThreshold ?? 0));
        }
      }
      return sum;
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.centroids_ || !this.classes_) throw new NotFittedError("NearestCentroidExt not fitted.");
    return Int32Array.from(X.map(xi => {
      let bestClass = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.centroids_!.length; k++) {
        let d = 0;
        const ck = this.centroids_![k]!;
        if (this.metric === "cosine") {
          let dot = 0;
          let na = 0;
          let nb = 0;
          for (let j = 0; j < xi.length; j++) { dot += (xi[j] ?? 0) * (ck[j] ?? 0); na += (xi[j] ?? 0) ** 2; nb += (ck[j] ?? 0) ** 2; }
          d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-15);
        } else {
          for (let j = 0; j < xi.length; j++) d += ((xi[j] ?? 0) - (ck[j] ?? 0)) ** 2;
        }
        if (d < bestDist) { bestDist = d; bestClass = this.classes_![k] ?? 0; }
      }
      return bestClass;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }
}
