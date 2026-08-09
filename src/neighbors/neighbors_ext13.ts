/**
 * Approximate Nearest Neighbors with LSH and ANNOY-style trees.
 */

export class LSHIndex {
  private hashTables_!: Map<string, number[]>[];
  private hyperplanes_!: Float64Array[];
  private fitted_ = false;

  constructor(private nHashBits = 8, private nTables = 4) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 1;
    const nHyperplanes = this.nHashBits * this.nTables;
    this.hyperplanes_ = Array.from({ length: nHyperplanes }, () =>
      new Float64Array(p).map(() => (Math.random() - 0.5))
    );
    this.hashTables_ = Array.from({ length: this.nTables }, (_, t) => {
      const table = new Map<string, number[]>();
      for (let i = 0; i < X.length; i++) {
        const key = this._hash(X[i]!, t);
        const bucket = table.get(key) ?? [];
        bucket.push(i);
        table.set(key, bucket);
      }
      return table;
    });
    this.fitted_ = true;
    return this;
  }

  private _hash(x: Float64Array, tableIdx: number): string {
    const bits: number[] = [];
    for (let b = 0; b < this.nHashBits; b++) {
      const hp = this.hyperplanes_[tableIdx * this.nHashBits + b]!;
      bits.push(x.reduce((s, v, j) => s + v * (hp[j] ?? 0), 0) >= 0 ? 1 : 0);
    }
    return bits.join('');
  }

  queryCandidates(x: Float64Array, maxCandidates = 100): number[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const candidates = new Set<number>();
    for (let t = 0; t < this.nTables; t++) {
      const key = this._hash(x, t);
      for (const idx of this.hashTables_[t]?.get(key) ?? []) {
        candidates.add(idx);
        if (candidates.size >= maxCandidates) break;
      }
    }
    return [...candidates];
  }

  queryKNN(X: Float64Array[], XQuery: Float64Array[], k = 5): { indices: Int32Array[]; distances: Float64Array[] } {
    if (!this.fitted_) throw new Error('Not fitted');
    const indices: Int32Array[] = [], distances: Float64Array[] = [];
    for (const q of XQuery) {
      const cands = this.queryCandidates(q);
      const scored = cands.map(i => ({ i, d: X[i]!.reduce((s, v, j) => s + (v - (q[j] ?? 0)) ** 2, 0) }));
      scored.sort((a, b) => a.d - b.d);
      const topK = scored.slice(0, k);
      indices.push(new Int32Array(topK.map(x => x.i)));
      distances.push(new Float64Array(topK.map(x => Math.sqrt(x.d))));
    }
    return { indices, distances };
  }
}

export class RandomProjectionTree {
  private left_!: RandomProjectionTree | null;
  private right_!: RandomProjectionTree | null;
  private hyperplane_!: Float64Array;
  private bias_!: number;
  private indices_!: number[];
  private leaf_ = false;

  constructor(private leafSize = 10) {
    this.left_ = null;
    this.right_ = null;
    this.hyperplane_ = new Float64Array(0);
    this.bias_ = 0;
    this.indices_ = [];
    this.leaf_ = false;
  }

  build(X: Float64Array[], indices: number[]): void {
    if (indices.length <= this.leafSize) {
      this.leaf_ = true;
      this.indices_ = indices;
      return;
    }
    const p = X[0]?.length ?? 1;
    this.hyperplane_ = new Float64Array(p).map(() => (Math.random() - 0.5));
    const norm = Math.sqrt(this.hyperplane_.reduce((s, v) => s + v * v, 0)) + 1e-10;
    for (let j = 0; j < p; j++) this.hyperplane_[j] = (this.hyperplane_[j] ?? 0) / norm;
    const projs = indices.map(i => X[i]!.reduce((s, v, j) => s + v * (this.hyperplane_[j] ?? 0), 0));
    projs.sort((a, b) => a - b);
    this.bias_ = projs[Math.floor(projs.length / 2)] ?? 0;
    const leftIdx = indices.filter(i => X[i]!.reduce((s, v, j) => s + v * (this.hyperplane_[j] ?? 0), 0) < this.bias_);
    const rightIdx = indices.filter(i => !leftIdx.includes(i));
    this.left_ = new RandomProjectionTree(this.leafSize);
    this.right_ = new RandomProjectionTree(this.leafSize);
    this.left_.build(X, leftIdx.length > 0 ? leftIdx : indices.slice(0, Math.floor(indices.length / 2)));
    this.right_.build(X, rightIdx.length > 0 ? rightIdx : indices.slice(Math.floor(indices.length / 2)));
  }

  queryCandidates(x: Float64Array): number[] {
    if (this.leaf_) return this.indices_;
    const proj = x.reduce((s, v, j) => s + v * (this.hyperplane_[j] ?? 0), 0);
    const child = proj < this.bias_ ? this.left_ : this.right_;
    return child?.queryCandidates(x) ?? this.indices_;
  }
}

export class ANNClassifier {
  private tree_!: RandomProjectionTree;
  private XTrain_!: Float64Array[];
  private yTrain_!: Int32Array;
  private fitted_ = false;

  constructor(private k = 5, private nTrees = 5) {}

  fit(X: Float64Array[], y: Int32Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    this.tree_ = new RandomProjectionTree(Math.max(this.k, 10));
    this.tree_.build(X, Array.from({ length: X.length }, (_, i) => i));
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      const cands = this.tree_.queryCandidates(x);
      const scored = cands.map(i => ({ i, d: this.XTrain_[i]!.reduce((s, v, j) => s + (v - (x[j] ?? 0)) ** 2, 0) }))
        .sort((a, b) => a.d - b.d).slice(0, this.k);
      const votes = new Map<number, number>();
      for (const { i } of scored) {
        const c = this.yTrain_[i] ?? 0;
        votes.set(c, (votes.get(c) ?? 0) + 1);
      }
      return [...votes.entries()].reduce((best, [c, v]) => v > best.v ? { c, v } : best, { c: 0, v: -1 }).c;
    }));
  }
}
