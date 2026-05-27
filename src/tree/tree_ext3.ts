/**
 * Additional tree utilities: ExtraTreeClassifier, ExtraTreeRegressor.
 * Mirrors sklearn.tree extra tree estimators.
 */

import { NotFittedError } from "../exceptions.js";

interface TreeLeaf {
  value: number | Int32Array;
  nSamples: number;
}

interface TreeSplit {
  featureIndex: number;
  threshold: number;
  left: ExtraTreeNode;
  right: ExtraTreeNode;
}

type ExtraTreeNode = TreeLeaf | TreeSplit;

function isSplit(node: ExtraTreeNode): node is TreeSplit {
  return "featureIndex" in node;
}

function predictNode(node: ExtraTreeNode, x: Float64Array): number {
  if (!isSplit(node)) {
    const v = node.value;
    return typeof v === "number" ? v : (v[0] ?? 0);
  }
  return (x[node.featureIndex] ?? 0) <= node.threshold
    ? predictNode(node.left, x)
    : predictNode(node.right, x);
}

function buildExtraRegTree(
  X: Float64Array[],
  y: Float64Array,
  maxDepth: number,
  minSamplesLeaf: number,
  nFeaturesToTry: number,
  rng: { next: () => number },
): ExtraTreeNode {
  const n = X.length;
  if (n <= minSamplesLeaf || maxDepth === 0) {
    let sum = 0;
    for (const yi of y) sum += yi;
    return { value: n > 0 ? sum / n : 0, nSamples: n };
  }

  const nFeatures = X[0]?.length ?? 0;
  const featuresToTry: number[] = [];
  const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);
  for (let i = allFeatures.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = allFeatures[i];
    allFeatures[i] = allFeatures[j] ?? 0;
    allFeatures[j] = tmp ?? 0;
  }
  featuresToTry.push(...allFeatures.slice(0, Math.min(nFeaturesToTry, nFeatures)));

  let bestGain = -Number.POSITIVE_INFINITY;
  let bestFeature = featuresToTry[0] ?? 0;
  let bestThreshold = 0;

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  let totalVar = 0;
  for (const yi of y) totalVar += (yi - yMean) ** 2;

  for (const j of featuresToTry) {
    const minVal = Math.min(...X.map((row) => row[j] ?? 0));
    const maxVal = Math.max(...X.map((row) => row[j] ?? 0));
    if (minVal === maxVal) continue;

    // Random threshold
    const threshold = minVal + rng.next() * (maxVal - minVal);

    let leftSum = 0;
    let rightSum = 0;
    let leftN = 0;
    let rightN = 0;
    for (let i = 0; i < n; i++) {
      if ((X[i]?.[j] ?? 0) <= threshold) {
        leftSum += y[i] ?? 0;
        leftN++;
      } else {
        rightSum += y[i] ?? 0;
        rightN++;
      }
    }
    if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) continue;

    const leftMean = leftSum / leftN;
    const rightMean = rightSum / rightN;
    let leftVar = 0;
    let rightVar = 0;
    for (let i = 0; i < n; i++) {
      if ((X[i]?.[j] ?? 0) <= threshold) leftVar += ((y[i] ?? 0) - leftMean) ** 2;
      else rightVar += ((y[i] ?? 0) - rightMean) ** 2;
    }
    const gain = totalVar - leftVar - rightVar;
    if (gain > bestGain) {
      bestGain = gain;
      bestFeature = j;
      bestThreshold = threshold;
    }
  }

  if (bestGain <= 0) {
    let sum = 0;
    for (const yi of y) sum += yi;
    return { value: n > 0 ? sum / n : 0, nSamples: n };
  }

  const leftX: Float64Array[] = [];
  const leftY: number[] = [];
  const rightX: Float64Array[] = [];
  const rightY: number[] = [];

  for (let i = 0; i < n; i++) {
    if ((X[i]?.[bestFeature] ?? 0) <= bestThreshold) {
      leftX.push(X[i]!);
      leftY.push(y[i] ?? 0);
    } else {
      rightX.push(X[i]!);
      rightY.push(y[i] ?? 0);
    }
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildExtraRegTree(leftX, new Float64Array(leftY), maxDepth - 1, minSamplesLeaf, nFeaturesToTry, rng),
    right: buildExtraRegTree(rightX, new Float64Array(rightY), maxDepth - 1, minSamplesLeaf, nFeaturesToTry, rng),
  };
}

export class ExtraTreeRegressor {
  maxDepth: number;
  minSamplesLeaf: number;
  maxFeatures: number | "auto" | "sqrt" | "log2" | null;
  randomState: number;

  private tree_: ExtraTreeNode | null = null;
  featureImportances_: Float64Array | null = null;

  constructor(
    options: {
      maxDepth?: number;
      minSamplesLeaf?: number;
      maxFeatures?: number | "auto" | "sqrt" | "log2" | null;
      randomState?: number;
    } = {},
  ) {
    this.maxDepth = options.maxDepth ?? Number.MAX_SAFE_INTEGER;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nFeatures = X[0]?.length ?? 0;
    let nFeaturesToTry = nFeatures;
    if (this.maxFeatures === "sqrt" || this.maxFeatures === "auto") {
      nFeaturesToTry = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    } else if (this.maxFeatures === "log2") {
      nFeaturesToTry = Math.max(1, Math.round(Math.log2(nFeatures)));
    } else if (typeof this.maxFeatures === "number") {
      nFeaturesToTry = this.maxFeatures <= 1
        ? Math.max(1, Math.round(this.maxFeatures * nFeatures))
        : Math.round(this.maxFeatures);
    }

    let rngState = this.randomState;
    const rng = {
      next: (): number => {
        rngState = (rngState * 1664525 + 1013904223) >>> 0;
        return rngState / 4294967296;
      },
    };

    this.tree_ = buildExtraRegTree(X, y, this.maxDepth, this.minSamplesLeaf, nFeaturesToTry, rng);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.tree_) throw new NotFittedError("ExtraTreeRegressor is not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      out[i] = predictNode(this.tree_, X[i] ?? new Float64Array(0));
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
