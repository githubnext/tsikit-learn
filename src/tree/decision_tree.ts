/**
 * Decision Tree Classifier and Regressor.
 * Mirrors sklearn.tree.DecisionTreeClassifier / DecisionTreeRegressor.
 */

import { NotFittedError } from "../exceptions.js";

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  value: Float64Array;
  isLeaf: boolean;
}

function giniImpurity(y: number[]): number {
  const counts = new Map<number, number>();
  for (const label of y) counts.set(label, (counts.get(label) ?? 0) + 1);
  let impurity = 1;
  for (const count of counts.values()) {
    impurity -= (count / y.length) ** 2;
  }
  return impurity;
}

function mse(y: number[]): number {
  if (y.length === 0) return 0;
  const mean = y.reduce((a, b) => a + b, 0) / y.length;
  return y.reduce((s, v) => s + (v - mean) ** 2, 0) / y.length;
}

function classificationLeafValue(y: number[]): Float64Array {
  const counts = new Map<number, number>();
  for (const label of y) counts.set(label, (counts.get(label) ?? 0) + 1);
  let best = 0;
  let bestCount = 0;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestCount = count;
      best = label;
    }
  }
  return new Float64Array([best]);
}

function regressionLeafValue(y: number[]): Float64Array {
  return new Float64Array([y.reduce((a, b) => a + b, 0) / y.length]);
}

function buildTree(
  X: Float64Array[],
  y: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  criterion: "gini" | "mse",
): TreeNode {
  const leafValue =
    criterion === "gini"
      ? classificationLeafValue(y)
      : regressionLeafValue(y);

  if (
    depth >= maxDepth ||
    y.length < minSamplesSplit ||
    new Set(y).size === 1
  ) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: leafValue, isLeaf: true };
  }

  const nFeatures = (X[0] ?? new Float64Array(0)).length;
  let bestGain = -Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;

  const currentImpurity = criterion === "gini" ? giniImpurity(y) : mse(y);

  for (let j = 0; j < nFeatures; j++) {
    const vals = X.map((xi) => xi[j] ?? 0);
    const sorted = Array.from(new Set(vals)).sort((a, b) => a - b);
    for (let ti = 0; ti < sorted.length - 1; ti++) {
      const threshold = ((sorted[ti] ?? 0) + (sorted[ti + 1] ?? 0)) / 2;
      const leftY: number[] = [];
      const rightY: number[] = [];
      for (let i = 0; i < X.length; i++) {
        ((X[i] ?? new Float64Array(0))[j] ?? 0) <= threshold
          ? leftY.push(y[i] ?? 0)
          : rightY.push(y[i] ?? 0);
      }
      if (leftY.length === 0 || rightY.length === 0) continue;

      const n = y.length;
      const leftImpurity = criterion === "gini" ? giniImpurity(leftY) : mse(leftY);
      const rightImpurity = criterion === "gini" ? giniImpurity(rightY) : mse(rightY);
      const gain =
        currentImpurity -
        (leftY.length / n) * leftImpurity -
        (rightY.length / n) * rightImpurity;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = j;
        bestThreshold = threshold;
      }
    }
  }

  if (bestGain <= 0) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: leafValue, isLeaf: true };
  }

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (let i = 0; i < X.length; i++) {
    ((X[i] ?? new Float64Array(0))[bestFeature] ?? 0) <= bestThreshold
      ? leftIdx.push(i)
      : rightIdx.push(i);
  }

  const leftX = leftIdx.map((i) => X[i] ?? new Float64Array(0));
  const leftY = leftIdx.map((i) => y[i] ?? 0);
  const rightX = rightIdx.map((i) => X[i] ?? new Float64Array(0));
  const rightY = rightIdx.map((i) => y[i] ?? 0);

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(leftX, leftY, depth + 1, maxDepth, minSamplesSplit, criterion),
    right: buildTree(rightX, rightY, depth + 1, maxDepth, minSamplesSplit, criterion),
    value: leafValue,
    isLeaf: false,
  };
}

function predict1(node: TreeNode, x: Float64Array): number {
  if (node.isLeaf) return node.value[0] ?? 0;
  return (x[node.featureIndex] ?? 0) <= node.threshold
    ? predict1(node.left as TreeNode, x)
    : predict1(node.right as TreeNode, x);
}

export class DecisionTreeClassifier {
  maxDepth: number;
  minSamplesSplit: number;
  criterion: string;

  tree_: TreeNode | null = null;
  classes_: Float64Array | null = null;
  nFeatures_: number = 0;

  constructor(
    options: {
      maxDepth?: number;
      minSamplesSplit?: number;
      criterion?: string;
    } = {},
  ) {
    this.maxDepth = options.maxDepth ?? Infinity;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.criterion = options.criterion ?? "gini";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.nFeatures_ = (X[0] ?? new Float64Array(0)).length;
    this.classes_ = new Float64Array(
      Array.from(new Set(Array.from(y))).sort((a, b) => a - b),
    );
    this.tree_ = buildTree(
      X,
      Array.from(y),
      0,
      this.maxDepth,
      this.minSamplesSplit,
      "gini",
    );
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.tree_ === null) throw new NotFittedError("DecisionTreeClassifier");
    return new Float64Array(X.map((xi) => predict1(this.tree_ as TreeNode, xi)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (this.tree_ === null || this.classes_ === null)
      throw new NotFittedError("DecisionTreeClassifier");
    const classes = this.classes_;
    return X.map((xi) => {
      const pred = predict1(this.tree_ as TreeNode, xi);
      const proba = new Float64Array(classes.length);
      const idx = Array.from(classes).indexOf(pred);
      if (idx >= 0) proba[idx] = 1;
      return proba;
    });
  }
}

export class DecisionTreeRegressor {
  maxDepth: number;
  minSamplesSplit: number;

  tree_: TreeNode | null = null;
  nFeatures_: number = 0;

  constructor(
    options: { maxDepth?: number; minSamplesSplit?: number } = {},
  ) {
    this.maxDepth = options.maxDepth ?? Infinity;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.nFeatures_ = (X[0] ?? new Float64Array(0)).length;
    this.tree_ = buildTree(
      X,
      Array.from(y),
      0,
      this.maxDepth,
      this.minSamplesSplit,
      "mse",
    );
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.tree_ === null) throw new NotFittedError("DecisionTreeRegressor");
    return new Float64Array(X.map((xi) => predict1(this.tree_ as TreeNode, xi)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}
