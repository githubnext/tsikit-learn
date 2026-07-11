/**
 * Decision tree utility functions and helpers.
 * Port of sklearn.tree._utils and tree.export
 */

export interface TreeNode {
  feature: number;
  threshold: number;
  leftChild: number;
  rightChild: number;
  value: Float64Array;
  nNodeSamples: number;
  impurity: number;
  isLeaf: boolean;
}

/**
 * Decision tree structure container.
 */
export class TreeStructure {
  nodes: TreeNode[] = [];
  nFeatures: number;
  nOutputs: number;
  nClasses: number;

  constructor(nFeatures: number, nOutputs: number, nClasses: number) {
    this.nFeatures = nFeatures;
    this.nOutputs = nOutputs;
    this.nClasses = nClasses;
  }

  addNode(node: TreeNode): number {
    const idx = this.nodes.length;
    this.nodes.push(node);
    return idx;
  }

  applyPredict(x: Float64Array): Float64Array {
    let nodeIdx = 0;
    while (!this.nodes[nodeIdx]!.isLeaf) {
      const node = this.nodes[nodeIdx]!;
      if ((x[node.feature] ?? 0) <= node.threshold) {
        nodeIdx = node.leftChild;
      } else {
        nodeIdx = node.rightChild;
      }
    }
    return this.nodes[nodeIdx]!.value;
  }

  /** Get the decision path indices for a sample */
  decisionPath(x: Float64Array): Int32Array {
    const path: number[] = [];
    let nodeIdx = 0;
    while (!this.nodes[nodeIdx]!.isLeaf) {
      path.push(nodeIdx);
      const node = this.nodes[nodeIdx]!;
      if ((x[node.feature] ?? 0) <= node.threshold) {
        nodeIdx = node.leftChild;
      } else {
        nodeIdx = node.rightChild;
      }
    }
    path.push(nodeIdx);
    return new Int32Array(path);
  }

  /** Get maximum depth of tree */
  get maxDepth(): number {
    const getDepth = (nodeIdx: number): number => {
      const node = this.nodes[nodeIdx];
      if (!node || node.isLeaf) return 0;
      return 1 + Math.max(getDepth(node.leftChild), getDepth(node.rightChild));
    };
    return getDepth(0);
  }

  /** Get number of leaves */
  get nLeaves(): number {
    return this.nodes.filter((n) => n.isLeaf).length;
  }
}

/**
 * Compute Gini impurity for a label distribution.
 */
export function giniImpurity(classCounts: Float64Array): number {
  const total = classCounts.reduce((s, c) => s + c, 0);
  if (total === 0) return 0;
  let gini = 1.0;
  for (const c of classCounts) gini -= (c / total) ** 2;
  return gini;
}

/**
 * Compute entropy impurity.
 */
export function entropyImpurity(classCounts: Float64Array): number {
  const total = classCounts.reduce((s, c) => s + c, 0);
  if (total === 0) return 0;
  let entropy = 0;
  for (const c of classCounts) {
    if (c > 0) entropy -= (c / total) * Math.log2(c / total);
  }
  return entropy;
}

/**
 * Compute mean squared error impurity (for regression).
 */
export function mseImpurity(values: Float64Array): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
}

/**
 * Find best split for a feature using sorted unique thresholds.
 */
export function findBestSplit(
  X: Float64Array[],
  y: Float64Array | Int32Array,
  featureIdx: number,
  criterion: "gini" | "entropy" | "mse",
  minSamplesLeaf = 1,
): { threshold: number; improvement: number } | null {
  const n = X.length;
  const values = X.map((x) => x[featureIdx] ?? 0);
  const sortedValues = [...new Set(values)].sort((a, b) => a - b);

  if (sortedValues.length <= 1) return null;

  const isClassification = criterion !== "mse";
  let bestThreshold = sortedValues[0]!;
  let bestImprovement = -Number.POSITIVE_INFINITY;

  // Current impurity
  let parentImpurity: number;
  if (isClassification) {
    const classSet = Array.from(new Set(Array.from(y as Int32Array)));
    const counts = new Float64Array(classSet.length);
    for (let i = 0; i < n; i++) {
      const ci = classSet.indexOf((y as Int32Array)[i]!);
      if (ci >= 0) counts[ci] = (counts[ci] ?? 0) + 1;
    }
    parentImpurity =
      criterion === "gini" ? giniImpurity(counts) : entropyImpurity(counts);
  } else {
    parentImpurity = mseImpurity(y as Float64Array);
  }

  for (let ti = 0; ti < sortedValues.length - 1; ti++) {
    const threshold = (sortedValues[ti]! + sortedValues[ti + 1]!) / 2;
    const leftMask = values.map((v) => v <= threshold);
    const rightMask = leftMask.map((v) => !v);
    const nLeft = leftMask.filter(Boolean).length;
    const nRight = n - nLeft;
    if (nLeft < minSamplesLeaf || nRight < minSamplesLeaf) continue;

    let leftImpurity: number;
    let rightImpurity: number;

    if (isClassification) {
      const classSet = Array.from(new Set(Array.from(y as Int32Array)));
      const leftCounts = new Float64Array(classSet.length);
      const rightCounts = new Float64Array(classSet.length);
      for (let i = 0; i < n; i++) {
        const ci = classSet.indexOf((y as Int32Array)[i]!);
        if (ci < 0) continue;
        if (leftMask[i]) leftCounts[ci] = (leftCounts[ci] ?? 0) + 1;
        else rightCounts[ci] = (rightCounts[ci] ?? 0) + 1;
      }
      leftImpurity =
        criterion === "gini"
          ? giniImpurity(leftCounts)
          : entropyImpurity(leftCounts);
      rightImpurity =
        criterion === "gini"
          ? giniImpurity(rightCounts)
          : entropyImpurity(rightCounts);
    } else {
      const leftY = new Float64Array(
        Array.from(y as Float64Array).filter((_, i) => leftMask[i]),
      );
      const rightY = new Float64Array(
        Array.from(y as Float64Array).filter((_, i) => !leftMask[i]),
      );
      leftImpurity = mseImpurity(leftY);
      rightImpurity = mseImpurity(rightY);
    }

    const improvement =
      parentImpurity - (nLeft * leftImpurity + nRight * rightImpurity) / n;
    if (improvement > bestImprovement) {
      bestImprovement = improvement;
      bestThreshold = threshold;
    }
  }

  return { threshold: bestThreshold, improvement: bestImprovement };
}

/**
 * Compute feature importances from tree nodes.
 */
export function computeFeatureImportances(
  tree: TreeStructure,
  nFeatures: number,
): Float64Array {
  const importances = new Float64Array(nFeatures);
  for (const node of tree.nodes) {
    if (!node.isLeaf) {
      const improvement = node.impurity * node.nNodeSamples;
      const leftImpurity =
        (tree.nodes[node.leftChild]?.impurity ?? 0) *
        (tree.nodes[node.leftChild]?.nNodeSamples ?? 0);
      const rightImpurity =
        (tree.nodes[node.rightChild]?.impurity ?? 0) *
        (tree.nodes[node.rightChild]?.nNodeSamples ?? 0);
      importances[node.feature] =
        (importances[node.feature] ?? 0) +
        improvement -
        leftImpurity -
        rightImpurity;
    }
  }
  const total = importances.reduce((s, v) => s + v, 0);
  if (total > 0) for (let j = 0; j < nFeatures; j++) importances[j]! /= total;
  return importances;
}
