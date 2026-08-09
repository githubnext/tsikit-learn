/**
 * Extra Trees (Extremely Randomized Trees) base implementation.
 */

export interface TreeNode {
  feature: number;
  threshold: number;
  value: number;
  left: TreeNode | null;
  right: TreeNode | null;
  isLeaf: boolean;
}

export class ExtraTreeRegressor {
  private root_!: TreeNode;
  private fitted_ = false;

  constructor(private maxDepth = 10, private minSamplesSplit = 2, private minSamplesLeaf = 1) {}

  fit(X: Float64Array[], y: Float64Array): this {
    this.root_ = this._buildTree(X, y, 0);
    this.fitted_ = true;
    return this;
  }

  private _buildTree(X: Float64Array[], y: Float64Array, depth: number): TreeNode {
    const n = X.length, p = X[0]?.length ?? 0;
    const mean = y.reduce((s, v) => s + v, 0) / (n || 1);

    if (n < this.minSamplesSplit || depth >= this.maxDepth) {
      return { feature: 0, threshold: 0, value: mean, left: null, right: null, isLeaf: true };
    }

    // Extra Trees: select random threshold for each random feature
    let bestFeat = 0, bestThresh = 0, bestScore = -Number.POSITIVE_INFINITY;
    const nFeatsToTry = Math.max(1, Math.floor(Math.sqrt(p)));
    const featIndices = Array.from({ length: p }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, nFeatsToTry);

    for (const j of featIndices) {
      const vals = X.map(row => row[j] ?? 0);
      const min = Math.min(...vals), max = Math.max(...vals);
      if (min === max) continue;
      // Random threshold in [min, max]
      const thresh = min + Math.random() * (max - min);
      const leftIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![j] ?? 0) <= thresh);
      const rightIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![j] ?? 0) > thresh);
      if (leftIdx.length < this.minSamplesLeaf || rightIdx.length < this.minSamplesLeaf) continue;
      const score = -this._mse(y, leftIdx) * leftIdx.length / n - this._mse(y, rightIdx) * rightIdx.length / n;
      if (score > bestScore) { bestScore = score; bestFeat = j; bestThresh = thresh; }
    }

    if (bestScore === -Number.POSITIVE_INFINITY) {
      return { feature: 0, threshold: 0, value: mean, left: null, right: null, isLeaf: true };
    }

    const leftIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![bestFeat] ?? 0) <= bestThresh);
    const rightIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![bestFeat] ?? 0) > bestThresh);
    return {
      feature: bestFeat, threshold: bestThresh, value: mean, isLeaf: false,
      left: this._buildTree(leftIdx.map(i => X[i]!), new Float64Array(leftIdx.map(i => y[i] ?? 0)), depth + 1),
      right: this._buildTree(rightIdx.map(i => X[i]!), new Float64Array(rightIdx.map(i => y[i] ?? 0)), depth + 1),
    };
  }

  private _mse(y: Float64Array, idx: number[]): number {
    if (idx.length === 0) return 0;
    const mean = idx.reduce((s, i) => s + (y[i] ?? 0), 0) / idx.length;
    return idx.reduce((s, i) => s + ((y[i] ?? 0) - mean) ** 2, 0) / idx.length;
  }

  private _predictNode(x: Float64Array, node: TreeNode): number {
    if (node.isLeaf || !node.left || !node.right) return node.value;
    return (x[node.feature] ?? 0) <= node.threshold
      ? this._predictNode(x, node.left)
      : this._predictNode(x, node.right);
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(x => this._predictNode(x, this.root_)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    const ss_res = pred.reduce((s, p, i) => s + ((y[i] ?? 0) - p) ** 2, 0);
    const ss_tot = Array.from(y).reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ss_res / (ss_tot + 1e-10);
  }
}

export class ExtraTreeClassifier {
  private root_!: TreeNode;
  private fitted_ = false;
  private classes_!: number[];

  constructor(private maxDepth = 10, private minSamplesSplit = 2, private minSamplesLeaf = 1) {}

  fit(X: Float64Array[], y: Int32Array): this {
    this.classes_ = Array.from(new Set(Array.from(y)));
    this.root_ = this._buildTree(X, y, 0);
    this.fitted_ = true;
    return this;
  }

  private _buildTree(X: Float64Array[], y: Int32Array, depth: number): TreeNode {
    const n = X.length, p = X[0]?.length ?? 0;
    const majority = this._majority(y);
    if (n < this.minSamplesSplit || depth >= this.maxDepth) {
      return { feature: 0, threshold: 0, value: majority, left: null, right: null, isLeaf: true };
    }
    let bestFeat = 0, bestThresh = 0, bestGini = Number.POSITIVE_INFINITY;
    const nFeatsToTry = Math.max(1, Math.floor(Math.sqrt(p)));
    const featIndices = Array.from({ length: p }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, nFeatsToTry);
    for (const j of featIndices) {
      const vals = X.map(row => row[j] ?? 0);
      const min = Math.min(...vals), max = Math.max(...vals);
      if (min === max) continue;
      const thresh = min + Math.random() * (max - min);
      const leftIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![j] ?? 0) <= thresh);
      const rightIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![j] ?? 0) > thresh);
      if (leftIdx.length < this.minSamplesLeaf || rightIdx.length < this.minSamplesLeaf) continue;
      const gini = (this._gini(y, leftIdx) * leftIdx.length + this._gini(y, rightIdx) * rightIdx.length) / n;
      if (gini < bestGini) { bestGini = gini; bestFeat = j; bestThresh = thresh; }
    }
    if (bestGini === Number.POSITIVE_INFINITY) {
      return { feature: 0, threshold: 0, value: majority, left: null, right: null, isLeaf: true };
    }
    const leftIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![bestFeat] ?? 0) <= bestThresh);
    const rightIdx = Array.from({ length: n }, (_, i) => i).filter(i => (X[i]![bestFeat] ?? 0) > bestThresh);
    return {
      feature: bestFeat, threshold: bestThresh, value: majority, isLeaf: false,
      left: this._buildTree(leftIdx.map(i => X[i]!), new Int32Array(leftIdx.map(i => y[i] ?? 0)), depth + 1),
      right: this._buildTree(rightIdx.map(i => X[i]!), new Int32Array(rightIdx.map(i => y[i] ?? 0)), depth + 1),
    };
  }

  private _gini(y: Int32Array, idx: number[]): number {
    if (idx.length === 0) return 0;
    const counts = new Map<number, number>();
    for (const i of idx) counts.set(y[i] ?? 0, (counts.get(y[i] ?? 0) ?? 0) + 1);
    return 1 - [...counts.values()].reduce((s, c) => s + (c / idx.length) ** 2, 0);
  }

  private _majority(y: Int32Array): number {
    const counts = new Map<number, number>();
    Array.from(y).forEach(v => counts.set(v, (counts.get(v) ?? 0) + 1));
    return [...counts.entries()].reduce((best, [c, n]) => n > best.n ? { c, n } : best, { c: 0, n: -1 }).c;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let node = this.root_;
      for (let d = 0; d < 100; d++) {
        if (node.isLeaf || !node.left || !node.right) return node.value;
        node = (x[node.feature] ?? 0) <= node.threshold ? node.left : node.right;
      }
      return node.value;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    return pred.filter((v, i) => v === y[i]).length / pred.length;
  }

  get classes(): number[] { return this.classes_; }
}
