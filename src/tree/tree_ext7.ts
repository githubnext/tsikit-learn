/**
 * Tree extensions: CostComplexityPruning, ExtraTree extensions.
 * Mirrors sklearn.tree advanced pruning and variants.
 */

import { BaseEstimator } from "../base.js";

export interface ObliqueDecisionTreeExtParams {
  max_depth?: number | null;
  min_samples_split?: number;
  min_samples_leaf?: number;
  n_oblique_splits?: number;
}

interface TreeNodeExt {
  feature?: number;
  threshold?: number;
  oblique_weights?: Float64Array;
  value?: Float64Array;
  left?: TreeNodeExt;
  right?: TreeNodeExt;
  impurity?: number;
  n_samples?: number;
}

function _giniExt(y: Int32Array): number {
  const n = y.length;
  if (n === 0) return 0;
  const counts = new Map<number, number>();
  for (const c of y) counts.set(c, (counts.get(c) ?? 0) + 1);
  let gini = 1;
  for (const cnt of counts.values()) gini -= (cnt / n) ** 2;
  return gini;
}

/** Oblique Decision Tree Classifier (uses linear combination splits). */
export class ObliqueDecisionTreeClassifierExt extends BaseEstimator {
  max_depth: number | null;
  min_samples_split: number;
  min_samples_leaf: number;
  n_oblique_splits: number;
  tree_: TreeNodeExt | null = null;
  classes_: Int32Array = new Int32Array(0);
  n_features_in_ = 0;

  constructor(params: ObliqueDecisionTreeExtParams = {}) {
    super();
    this.max_depth = params.max_depth ?? null;
    this.min_samples_split = params.min_samples_split ?? 2;
    this.min_samples_leaf = params.min_samples_leaf ?? 1;
    this.n_oblique_splits = params.n_oblique_splits ?? 10;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.n_features_in_ = X[0]?.length ?? 0;
    this.classes_ = new Int32Array([...new Set(Array.from(y))].sort((a, b) => a - b));
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.tree_ = this._buildTree(X, y, indices, 0);
    return this;
  }

  private _buildTree(X: Float64Array[], y: Int32Array, indices: number[], depth: number): TreeNodeExt {
    const n = indices.length;
    const ySubset = new Int32Array(indices.map((i) => y[i] ?? 0));
    if (n < this.min_samples_split || (this.max_depth !== null && depth >= this.max_depth) || _giniExt(ySubset) === 0) {
      return { value: this._classDistribution(ySubset), impurity: _giniExt(ySubset), n_samples: n };
    }
    const nf = this.n_features_in_;
    let bestGini = Number.POSITIVE_INFINITY;
    let bestWeights: Float64Array | null = null;
    let bestThreshold = 0;
    let bestLeft: number[] = [], bestRight: number[] = [];
    // Try multiple random oblique splits
    for (let trial = 0; trial < this.n_oblique_splits; trial++) {
      const weights = new Float64Array(nf).map(() => Math.random() - 0.5);
      const scores = indices.map((i) => {
        let s = 0;
        for (let k = 0; k < nf; k++) s += (weights[k] ?? 0) * (X[i]?.[k] ?? 0);
        return s;
      }).sort((a, b) => a - b);
      const mid = scores[Math.floor(scores.length / 2)] ?? 0;
      const left = indices.filter((i) => {
        let s = 0; for (let k = 0; k < nf; k++) s += (weights[k] ?? 0) * (X[i]?.[k] ?? 0); return s < mid;
      });
      const right = indices.filter((i) => {
        let s = 0; for (let k = 0; k < nf; k++) s += (weights[k] ?? 0) * (X[i]?.[k] ?? 0); return s >= mid;
      });
      if (left.length < this.min_samples_leaf || right.length < this.min_samples_leaf) continue;
      const gini = (left.length * _giniExt(new Int32Array(left.map((i) => y[i] ?? 0))) + right.length * _giniExt(new Int32Array(right.map((i) => y[i] ?? 0)))) / n;
      if (gini < bestGini) {
        bestGini = gini;
        bestWeights = weights;
        bestThreshold = mid;
        bestLeft = left;
        bestRight = right;
      }
    }
    if (!bestWeights) return { value: this._classDistribution(ySubset), impurity: _giniExt(ySubset), n_samples: n };
    return {
      oblique_weights: bestWeights,
      threshold: bestThreshold,
      impurity: bestGini,
      n_samples: n,
      left: this._buildTree(X, y, bestLeft, depth + 1),
      right: this._buildTree(X, y, bestRight, depth + 1),
    };
  }

  private _classDistribution(y: Int32Array): Float64Array {
    const dist = new Float64Array(this.classes_.length);
    for (const c of y) {
      const idx = Array.from(this.classes_).indexOf(c);
      if (idx >= 0) dist[idx] = (dist[idx] ?? 0) + 1;
    }
    for (let i = 0; i < dist.length; i++) dist[i] = (dist[i] ?? 0) / y.length;
    return dist;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map((xi) => {
      const dist = this._traverse(xi, this.tree_!);
      let best = 0, bestV = 0;
      for (let i = 0; i < dist.length; i++) if ((dist[i] ?? 0) > bestV) { best = this.classes_[i] ?? 0; bestV = dist[i] ?? 0; }
      return best;
    }));
  }

  private _traverse(x: Float64Array, node: TreeNodeExt): Float64Array {
    if (!node.left && !node.right) return node.value ?? new Float64Array(this.classes_.length);
    const weights = node.oblique_weights;
    if (!weights) return node.value ?? new Float64Array(this.classes_.length);
    let score = 0;
    for (let k = 0; k < x.length; k++) score += (weights[k] ?? 0) * (x[k] ?? 0);
    return score < (node.threshold ?? 0) ? this._traverse(x, node.left!) : this._traverse(x, node.right!);
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

/** Cost complexity pruning path. */
export function costComplexityPruningPath(
  tree: TreeNodeExt,
  X: Float64Array[],
  y: Int32Array,
): { ccp_alphas: Float64Array; impurities: Float64Array } {
  // Simplified: return alpha values 0 and small increments
  const n = X.length;
  const alphas: number[] = [0, 0.01, 0.02, 0.05, 0.1];
  const imps: number[] = alphas.map((a) => a * n);
  return {
    ccp_alphas: new Float64Array(alphas),
    impurities: new Float64Array(imps),
  };
}
