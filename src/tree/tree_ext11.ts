/**
 * Oblique Decision Trees and Cost-Sensitive Decision Trees.
 */

export class ObliqueDecisionTree {
  private splitWeights_!: Float64Array[];
  private splitThresholds_!: Float64Array;
  private leafValues_!: Float64Array;
  private nodeFeatures_!: Int32Array;
  private leftChild_!: Int32Array;
  private rightChild_!: Int32Array;
  private nNodes_ = 0;
  private fitted_ = false;

  constructor(private maxDepth = 5, private nFeatures = 2) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const maxNodes = (1 << (this.maxDepth + 1));
    this.splitWeights_ = Array.from({ length: maxNodes }, () => new Float64Array(p));
    this.splitThresholds_ = new Float64Array(maxNodes);
    this.leafValues_ = new Float64Array(maxNodes);
    this.nodeFeatures_ = new Int32Array(maxNodes).fill(-1);
    this.leftChild_ = new Int32Array(maxNodes).fill(-1);
    this.rightChild_ = new Int32Array(maxNodes).fill(-1);

    const queue: Array<{ nodeId: number; indices: number[]; depth: number }> = [
      { nodeId: 0, indices: Array.from({ length: n }, (_, i) => i), depth: 0 }
    ];
    this.nNodes_ = 1;

    while (queue.length > 0) {
      const { nodeId, indices, depth } = queue.shift()!;
      const yNode = new Float64Array(indices.map(i => y[i] ?? 0));
      this.leafValues_[nodeId] = yNode.reduce((s, v) => s + v, 0) / (yNode.length || 1);

      if (depth >= this.maxDepth || indices.length < 4) continue;

      // Random oblique split
      const feats = Array.from({ length: p }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, this.nFeatures);
      const w = new Float64Array(p);
      for (const f of feats) w[f] = Math.random() - 0.5;
      const norm = Math.sqrt(w.reduce((s, v) => s + v * v, 0)) + 1e-10;
      for (let j = 0; j < p; j++) w[j] = (w[j] ?? 0) / norm;

      const projs = indices.map(i => X[i]!.reduce((s, v, j) => s + v * (w[j] ?? 0), 0));
      const sorted = [...projs].sort((a, b) => a - b);
      const thresh = sorted[Math.floor(sorted.length / 2)] ?? 0;

      const leftIdx = indices.filter((_, li) => (projs[li] ?? 0) <= thresh);
      const rightIdx = indices.filter((_, li) => (projs[li] ?? 0) > thresh);
      if (leftIdx.length === 0 || rightIdx.length === 0) continue;

      this.splitWeights_[nodeId] = w;
      this.splitThresholds_[nodeId] = thresh;
      this.nodeFeatures_[nodeId] = 0; // mark as internal
      this.leftChild_[nodeId] = this.nNodes_;
      queue.push({ nodeId: this.nNodes_++, indices: leftIdx, depth: depth + 1 });
      this.rightChild_[nodeId] = this.nNodes_;
      queue.push({ nodeId: this.nNodes_++, indices: rightIdx, depth: depth + 1 });
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(x => {
      let node = 0;
      for (let d = 0; d < 200; d++) {
        if (this.leftChild_[node] === -1) return this.leafValues_[node] ?? 0;
        const proj = x.reduce((s, v, j) => s + v * (this.splitWeights_[node]![j] ?? 0), 0);
        node = proj <= (this.splitThresholds_[node] ?? 0) ? (this.leftChild_[node] ?? 0) : (this.rightChild_[node] ?? 0);
      }
      return this.leafValues_[node] ?? 0;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    const ssRes = pred.reduce((s, p, i) => s + ((y[i] ?? 0) - p) ** 2, 0);
    const ssTot = Array.from(y).reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ssRes / (ssTot + 1e-10);
  }
}

export class DecisionTreePruner {
  constructor(private ccp_alpha = 0.0) {}

  prune<T extends { cost_: number; left_?: T | null; right_?: T | null; nLeaves_?: number }>(tree: T): T {
    if (!tree.left_ || !tree.right_) return tree;
    const pruned = this._pruneNode(tree, this.ccp_alpha);
    return pruned as T;
  }

  private _pruneNode(node: Record<string, unknown>, alpha: number): Record<string, unknown> {
    if (!node['left_'] && !node['right_']) return node;
    const left = this._pruneNode(node['left_'] as Record<string, unknown> ?? {}, alpha);
    const right = this._pruneNode(node['right_'] as Record<string, unknown> ?? {}, alpha);
    const cost = (node['cost_'] as number ?? 0) - alpha * ((node['nLeaves_'] as number ?? 1) - 1);
    if (cost <= 0) return { ...node, left_: null, right_: null, isLeaf: true };
    return { ...node, left_: left, right_: right };
  }
}
