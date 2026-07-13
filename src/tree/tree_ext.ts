/**
 * Decision tree utilities — splitting criteria, tree pruning, and tree export.
 */

export type CriterionType = "gini" | "entropy" | "mse" | "mae";

export function giniImpurity(counts: Int32Array): number {
  const total = counts.reduce((s, v) => s + v, 0);
  if (total === 0) return 0;
  let sum = 0;
  for (const c of counts) sum += (c / total) ** 2;
  return 1 - sum;
}

export function entropy(counts: Int32Array): number {
  const total = counts.reduce((s, v) => s + v, 0);
  if (total === 0) return 0;
  let h = 0;
  for (const c of counts) {
    if (c > 0) h -= (c / total) * Math.log2(c / total);
  }
  return h;
}

export function mseCriterion(y: Float64Array): number {
  if (y.length === 0) return 0;
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  return y.reduce((s, v) => s + (v - mean) ** 2, 0) / y.length;
}

export function maeCriterion(y: Float64Array): number {
  if (y.length === 0) return 0;
  const sorted = Array.from(y).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  return y.reduce((s, v) => s + Math.abs(v - median), 0) / y.length;
}

export interface TreeNode {
  isLeaf: boolean;
  value: number;
  nSamples: number;
  impurity: number;
  featureIdx?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  depth: number;
  nodeId: number;
}

export function buildDecisionTree(
  X: Float64Array[],
  y: Float64Array | Int32Array,
  maxDepth = 5,
  minSamplesSplit = 2,
  minSamplesLeaf = 1,
  criterion: CriterionType = "gini",
  nClasses = 2,
  depth = 0,
  nodeId = { current: 0 },
): TreeNode {
  const id = nodeId.current++;
  const n = X.length;

  const value = y instanceof Float64Array
    ? y.reduce((s, v) => s + v, 0) / Math.max(n, 1)
    : (() => {
      const counts = new Array(nClasses).fill(0);
      for (const label of y) counts[label]++;
      return counts.indexOf(Math.max(...counts));
    })();

  const impurity = criterion === "mse" || criterion === "mae"
    ? (y instanceof Float64Array ? (criterion === "mse" ? mseCriterion(y) : maeCriterion(y)) : 0)
    : (() => {
      const counts = new Int32Array(nClasses);
      for (const label of y) counts[label as number]!++;
      return criterion === "gini" ? giniImpurity(counts) : entropy(counts);
    })();

  const baseNode: TreeNode = { isLeaf: true, value, nSamples: n, impurity, depth, nodeId: id };

  if (depth >= maxDepth || n < minSamplesSplit) return baseNode;

  const p = X[0]?.length ?? 0;
  let bestGain = -Number.POSITIVE_INFINITY, bestFeat = -1, bestThresh = 0;

  for (let feat = 0; feat < p; feat++) {
    const vals = Array.from({ length: n }, (_, i) => ({ x: X[i]?.[feat] ?? 0, y: y[i] })).sort((a, b) => a.x - b.x);
    for (let split = minSamplesLeaf; split < n - minSamplesLeaf; split++) {
      const thresh = ((vals[split - 1]?.x ?? 0) + (vals[split]?.x ?? 0)) / 2;
      const leftY = vals.slice(0, split).map((v) => v.y);
      const rightY = vals.slice(split).map((v) => v.y);

      let leftImpurity: number, rightImpurity: number;
      if (criterion === "mse" || criterion === "mae") {
        const lF = Float64Array.from(leftY as number[]);
        const rF = Float64Array.from(rightY as number[]);
        leftImpurity = criterion === "mse" ? mseCriterion(lF) : maeCriterion(lF);
        rightImpurity = criterion === "mse" ? mseCriterion(rF) : maeCriterion(rF);
      } else {
        const lC = new Int32Array(nClasses);
        const rC = new Int32Array(nClasses);
        for (const v of leftY) lC[v as number]!++;
        for (const v of rightY) rC[v as number]!++;
        leftImpurity = criterion === "gini" ? giniImpurity(lC) : entropy(lC);
        rightImpurity = criterion === "gini" ? giniImpurity(rC) : entropy(rC);
      }

      const gain = impurity - (leftY.length * leftImpurity + rightY.length * rightImpurity) / n;
      if (gain > bestGain) { bestGain = gain; bestFeat = feat; bestThresh = thresh; }
    }
  }

  if (bestFeat < 0 || bestGain <= 0) return baseNode;

  const leftMask = X.map((row) => (row[bestFeat] ?? 0) <= bestThresh);
  const rightMask = leftMask.map((v) => !v);
  const XLeft = X.filter((_, i) => leftMask[i]);
  const XRight = X.filter((_, i) => rightMask[i]);
  const yLeft = y instanceof Float64Array
    ? Float64Array.from(Array.from(y).filter((_, i) => leftMask[i]))
    : Int32Array.from(Array.from(y).filter((_, i) => leftMask[i]));
  const yRight = y instanceof Float64Array
    ? Float64Array.from(Array.from(y).filter((_, i) => rightMask[i]))
    : Int32Array.from(Array.from(y).filter((_, i) => rightMask[i]));

  if (XLeft.length < minSamplesLeaf || XRight.length < minSamplesLeaf) return baseNode;

  return {
    ...baseNode,
    isLeaf: false,
    featureIdx: bestFeat,
    threshold: bestThresh,
    left: buildDecisionTree(XLeft, yLeft, maxDepth, minSamplesSplit, minSamplesLeaf, criterion, nClasses, depth + 1, nodeId),
    right: buildDecisionTree(XRight, yRight, maxDepth, minSamplesSplit, minSamplesLeaf, criterion, nClasses, depth + 1, nodeId),
  };
}

export function predictTreeNode(node: TreeNode, x: Float64Array): number {
  if (node.isLeaf || !node.left || !node.right) return node.value;
  const feat = node.featureIdx ?? 0;
  const thresh = node.threshold ?? 0;
  return (x[feat] ?? 0) <= thresh ? predictTreeNode(node.left, x) : predictTreeNode(node.right, x);
}

export function pruneReducedError(node: TreeNode, Xval: Float64Array[], yval: Int32Array): TreeNode {
  if (node.isLeaf) return node;
  const pruned: TreeNode = { ...node, isLeaf: true, left: undefined, right: undefined };
  const prunedError = Xval.filter((_, i) => Math.round(predictTreeNode(pruned, Xval[i] as Float64Array)) !== (yval[i] ?? 0)).length;
  const fullError = Xval.filter((_, i) => Math.round(predictTreeNode(node, Xval[i] as Float64Array)) !== (yval[i] ?? 0)).length;
  return prunedError <= fullError ? pruned : node;
}

export function exportTreeToText(node: TreeNode, featureNames?: string[], depth = 0): string {
  const indent = "  ".repeat(depth);
  if (node.isLeaf) return `${indent}Leaf: value=${node.value.toFixed(4)}, samples=${node.nSamples}\n`;
  const feat = featureNames?.[node.featureIdx ?? 0] ?? `feature_${node.featureIdx ?? 0}`;
  let text = `${indent}${feat} <= ${(node.threshold ?? 0).toFixed(4)} (impurity=${node.impurity.toFixed(4)}, samples=${node.nSamples})\n`;
  text += exportTreeToText(node.left as TreeNode, featureNames, depth + 1);
  text += `${indent}else:\n`;
  text += exportTreeToText(node.right as TreeNode, featureNames, depth + 1);
  return text;
}

export function getTreeDepth(node: TreeNode): number {
  if (node.isLeaf) return 0;
  return 1 + Math.max(getTreeDepth(node.left as TreeNode), getTreeDepth(node.right as TreeNode));
}

export function countLeaves(node: TreeNode): number {
  if (node.isLeaf) return 1;
  return countLeaves(node.left as TreeNode) + countLeaves(node.right as TreeNode);
}
