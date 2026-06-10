/**
 * CatBoost-style gradient boosting with categorical features.
 */

export class HistogramGradientBoosting {
  private trees_!: Array<{ thresholds: Float64Array; values: Float64Array; feature: Int32Array; left: Int32Array; right: Int32Array }>;
  private learningRate: number;
  private maxLeafNodes: number;
  private fitted_ = false;
  private nClasses_ = 0;

  constructor(
    private maxIter = 100,
    private maxDepth = 3,
    learningRate = 0.1,
    private l2Regularization = 1.0,
    maxLeafNodes = 31
  ) {
    this.learningRate = learningRate;
    this.maxLeafNodes = maxLeafNodes;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.trees_ = [];
    let pred = new Float64Array(n);
    const shrinkage = this.learningRate;

    // Build histograms (discretize features)
    const nBins = 256;
    const binEdges = Array.from({ length: p }, (_, j) => {
      const vals = X.map(row => row[j] ?? 0).sort((a, b) => a - b);
      return new Float64Array(nBins + 1).map((_, b) => vals[Math.floor(b * n / nBins)] ?? 0);
    });

    const binX = X.map(row => new Int32Array(p).map((_, j) => {
      const edges = binEdges[j]!;
      let bin = 0;
      for (let b = 1; b < edges.length; b++) if ((row[j] ?? 0) >= (edges[b] ?? 0)) bin = b;
      return Math.min(bin, nBins - 1);
    }));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const residuals = new Float64Array(n).map((_, i) => (y[i] ?? 0) - (pred[i] ?? 0));
      const tree = this._buildTree(binX, residuals, binEdges, p);
      for (let i = 0; i < n; i++) pred[i] = (pred[i] ?? 0) + shrinkage * this._predict1(binX[i]!, tree);
      this.trees_.push(tree);
    }
    this.fitted_ = true;
    return this;
  }

  private _buildTree(X: Int32Array[], residuals: Float64Array, binEdges: Float64Array[], p: number): {
    thresholds: Float64Array; values: Float64Array; feature: Int32Array; left: Int32Array; right: Int32Array;
  } {
    const maxNodes = this.maxLeafNodes * 2 + 1;
    const thresholds = new Float64Array(maxNodes).fill(-1);
    const values = new Float64Array(maxNodes);
    const feature = new Int32Array(maxNodes).fill(-1);
    const left = new Int32Array(maxNodes).fill(-1);
    const right = new Int32Array(maxNodes).fill(-1);

    const n = X.length;
    const nodeIndices: number[][] = [Array.from({ length: n }, (_, i) => i)];
    let nodeCount = 1;

    for (let depth = 0; depth < this.maxDepth && nodeCount < this.maxLeafNodes; depth++) {
      const currentNodes = [...nodeIndices];
      for (let nodeIdx = 0; nodeIdx < currentNodes.length; nodeIdx++) {
        const indices = currentNodes[nodeIdx];
        if (!indices || indices.length < 2) { continue; }
        // Find best split
        let bestGain = 0, bestFeat = 0, bestBin = 0;
        for (let j = 0; j < p; j++) {
          for (let bin = 0; bin < 255; bin++) {
            const left_ = indices.filter(i => (X[i]![j] ?? 0) <= bin);
            const right_ = indices.filter(i => (X[i]![j] ?? 0) > bin);
            if (left_.length === 0 || right_.length === 0) continue;
            const gain = this._splitGain(residuals, left_, right_);
            if (gain > bestGain) { bestGain = gain; bestFeat = j; bestBin = bin; }
          }
        }
        if (bestGain > 0) {
          const nid = nodeCount > 0 ? nodeIdx : 0;
          feature[nid] = bestFeat;
          thresholds[nid] = binEdges[bestFeat]![bestBin] ?? 0;
          const leftIdx = indices.filter(i => (X[i]![bestFeat] ?? 0) <= bestBin);
          const rightIdx = indices.filter(i => (X[i]![bestFeat] ?? 0) > bestBin);
          left[nid] = nodeCount;
          nodeIndices.push(leftIdx);
          right[nid] = nodeCount + 1;
          nodeIndices.push(rightIdx);
          nodeCount += 2;
        } else {
          values[nodeIdx] = residuals.length > 0 ? indices.reduce((s, i) => s + (residuals[i] ?? 0), 0) / (indices.length + this.l2Regularization) : 0;
        }
      }
    }
    // Set leaf values
    for (let n_ = 0; n_ < nodeCount; n_++) {
      if (left[n_] === -1 && nodeIndices[n_]) {
        const idx = nodeIndices[n_]!;
        values[n_] = idx.reduce((s, i) => s + (residuals[i] ?? 0), 0) / (idx.length + this.l2Regularization);
      }
    }
    return { thresholds, values, feature, left, right };
  }

  private _splitGain(r: Float64Array, l: number[], ri: number[]): number {
    const sumSq = (idx: number[]) => {
      const s = idx.reduce((acc, i) => acc + (r[i] ?? 0), 0);
      return s * s / (idx.length + this.l2Regularization);
    };
    const total = l.reduce((s, i) => s + (r[i] ?? 0), 0) + ri.reduce((s, i) => s + (r[i] ?? 0), 0);
    const n = l.length + ri.length;
    return sumSq(l) + sumSq(ri) - total * total / (n + this.l2Regularization);
  }

  private _predict1(x: Int32Array, tree: { thresholds: Float64Array; values: Float64Array; feature: Int32Array; left: Int32Array; right: Int32Array }): number {
    let node = 0;
    for (let d = 0; d < 50; d++) {
      if (tree.feature[node] === -1 || tree.left[node] === -1) return tree.values[node] ?? 0;
      const f = tree.feature[node] ?? 0;
      const go = (x[f] ?? 0) <= (tree.thresholds[node] ?? 0);
      node = go ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0);
    }
    return tree.values[node] ?? 0;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0;
    return new Float64Array(X.map(row => this.trees_.reduce((s, tree) => {
      const binRow = new Int32Array(p).map((_, j) => Math.min(Math.round(row[j] ?? 0), 255));
      return s + this.learningRate * this._predict1(binRow, tree);
    }, 0)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    const ss_res = pred.reduce((s, p, i) => s + ((y[i] ?? 0) - p) ** 2, 0);
    const ss_tot = Array.from(y).reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ss_res / (ss_tot + 1e-10);
    void this.nClasses_;
  }
}
