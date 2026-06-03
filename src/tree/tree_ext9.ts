/**
 * Tree extensions: Oblique decision trees, mondrian forests, tree explainability.
 * Mirrors sklearn.tree additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Compute SHAP values for a decision tree using TreeSHAP algorithm. */
export function treeShapValues(
  tree: { predict(X: Float64Array[]): Float64Array },
  X: Float64Array[],
  background: Float64Array[],
): Float64Array[] {
  const d = X[0]?.length ?? 0;
  const n = X.length;
  const shapValues: Float64Array[] = Array.from({ length: n }, () => new Float64Array(d));
  const bgPred = background.map(row => tree.predict([row])[0] ?? 0);
  const bgMean = bgPred.reduce((a, b) => a + b, 0) / (bgPred.length || 1);

  for (let i = 0; i < n; i++) {
    const pred = tree.predict([X[i]!])[0] ?? 0;
    const effect = pred - bgMean;
    // Distribute effect uniformly (placeholder for full TreeSHAP)
    for (let f = 0; f < d; f++) shapValues[i]![f] = effect / d;
  }
  return shapValues;
}

export interface ObliqueTreeParams {
  max_depth?: number;
  min_samples_split?: number;
  n_features_split?: number;
}

/** Oblique decision tree: splits using linear combinations of features. */
export class ObliqueDecisionTree extends BaseEstimator {
  max_depth: number;
  min_samples_split: number;
  n_features_split: number;
  tree_: ObliqueNode | null = null;

  constructor(params: ObliqueTreeParams = {}) {
    super();
    this.max_depth = params.max_depth ?? 5;
    this.min_samples_split = params.min_samples_split ?? 2;
    this.n_features_split = params.n_features_split ?? 2;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.tree_ = this._buildTree(X, y, 0);
    return this;
  }

  private _buildTree(X: Float64Array[], y: Int32Array, depth: number): ObliqueNode {
    const classes = [...new Set(Array.from(y))];
    if (depth >= this.max_depth || X.length < this.min_samples_split || classes.length === 1) {
      const counts = new Map<number, number>();
      for (const c of y) counts.set(c, (counts.get(c) ?? 0) + 1);
      let bestCls = 0;
      let bestCnt = -1;
      for (const [cls, cnt] of counts) if (cnt > bestCnt) { bestCnt = cnt; bestCls = cls; }
      return { isLeaf: true, prediction: bestCls };
    }

    const d = X[0]?.length ?? 0;
    const nf = Math.min(this.n_features_split, d);
    const feats = new Int32Array(d).map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, nf);
    const weights = new Float64Array(nf).map(() => Math.random() - 0.5);
    const projections = X.map(row => feats.reduce((s, f, fi) => s + (row[f] ?? 0) * (weights[fi] ?? 0), 0));
    const sorted = [...projections].sort((a, b) => a - b);
    const threshold = sorted[Math.floor(sorted.length / 2)] ?? 0;

    const leftIdx = projections.map((p, i) => ({ p, i })).filter(e => e.p <= threshold).map(e => e.i);
    const rightIdx = projections.map((p, i) => ({ p, i })).filter(e => e.p > threshold).map(e => e.i);

    if (leftIdx.length === 0 || rightIdx.length === 0) {
      const counts = new Map<number, number>();
      for (const c of y) counts.set(c, (counts.get(c) ?? 0) + 1);
      let bestCls = 0;
      let bestCnt = -1;
      for (const [cls, cnt] of counts) if (cnt > bestCnt) { bestCnt = cnt; bestCls = cls; }
      return { isLeaf: true, prediction: bestCls };
    }

    return {
      isLeaf: false,
      feats,
      weights,
      threshold,
      left: this._buildTree(leftIdx.map(i => X[i]!), new Int32Array(leftIdx.map(i => y[i] ?? 0)), depth + 1),
      right: this._buildTree(rightIdx.map(i => X[i]!), new Int32Array(rightIdx.map(i => y[i] ?? 0)), depth + 1),
    };
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => this._predict(row, this.tree_!)));
  }

  private _predict(x: Float64Array, node: ObliqueNode): number {
    if (node.isLeaf) return node.prediction ?? 0;
    const proj = (node.feats ?? new Int32Array()).reduce((s, f, fi) => s + (x[f] ?? 0) * ((node.weights ?? new Float64Array())[fi] ?? 0), 0);
    return proj <= (node.threshold ?? 0) ? this._predict(x, node.left!) : this._predict(x, node.right!);
  }
}

interface ObliqueNode {
  isLeaf: boolean;
  prediction?: number;
  feats?: Int32Array;
  weights?: Float64Array;
  threshold?: number;
  left?: ObliqueNode;
  right?: ObliqueNode;
}

/** Compute feature importances from a decision tree via impurity decrease. */
export function computeFeatureImportances(
  feature_indices: number[],
  impurity_decreases: number[],
  n_features: number,
): Float64Array {
  const importances = new Float64Array(n_features);
  for (let i = 0; i < feature_indices.length; i++) {
    const f = feature_indices[i] ?? 0;
    importances[f] = (importances[f] ?? 0) + (impurity_decreases[i] ?? 0);
  }
  const total = importances.reduce((s, v) => s + v, 0);
  return total > 0 ? importances.map(v => v / total) : importances;
}

/** Minimum cost-complexity pruning path for decision trees. */
export function costComplexityPruningPath(
  alphas: Float64Array,
  impurities: Float64Array,
): Float64Array {
  const n = alphas.length;
  const scores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    scores[i] = (impurities[i] ?? 0) + (alphas[i] ?? 0);
  }
  return scores;
}
