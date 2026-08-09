/**
 * Gradient Boosting extensions: HistGradientBoostingClassifier/Regressor.
 * Mirrors sklearn.ensemble.HistGradientBoosting*.
 */

import { NotFittedError } from "../exceptions.js";

interface Leaf {
  value: number;
  count: number;
}

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | Leaf;
  right: TreeNode | Leaf;
}

type Node = TreeNode | Leaf;

function isLeaf(node: Node): node is Leaf {
  return "value" in node && !("featureIndex" in node);
}

function buildTree(
  X: Float64Array[],
  residuals: Float64Array,
  maxDepth: number,
  minSamplesLeaf: number,
): Node {
  if (X.length <= minSamplesLeaf || maxDepth === 0) {
    let sum = 0;
    for (const r of residuals) sum += r;
    return { value: X.length > 0 ? sum / X.length : 0, count: X.length };
  }

  const nFeatures = X[0]?.length ?? 0;
  let bestGain = -Number.POSITIVE_INFINITY;
  let bestFeature = 0;
  let bestThreshold = 0;

  // Try splitting on each feature
  for (let j = 0; j < nFeatures; j++) {
    const vals = X.map((row, i) => ({ v: row[j] ?? 0, r: residuals[i] ?? 0 }));
    vals.sort((a, b) => a.v - b.v);

    let sumLeft = 0;
    let sumRight = 0;
    for (const vr of vals) sumRight += vr.r;
    const totalMean = sumRight / vals.length;
    let totalSS = 0;
    for (const vr of vals) totalSS += (vr.r - totalMean) ** 2;

    for (let split = minSamplesLeaf; split <= vals.length - minSamplesLeaf; split++) {
      sumLeft += vals[split - 1]?.r ?? 0;
      sumRight -= vals[split - 1]?.r ?? 0;
      const meanL = sumLeft / split;
      const meanR = sumRight / (vals.length - split);
      let ssL = 0;
      let ssR = 0;
      for (let k = 0; k < split; k++) ssL += ((vals[k]?.r ?? 0) - meanL) ** 2;
      for (let k = split; k < vals.length; k++) ssR += ((vals[k]?.r ?? 0) - meanR) ** 2;
      const gain = totalSS - ssL - ssR;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = j;
        bestThreshold = ((vals[split - 1]?.v ?? 0) + (vals[split]?.v ?? 0)) / 2;
      }
    }
  }

  if (bestGain <= 0) {
    let sum = 0;
    for (const r of residuals) sum += r;
    return { value: X.length > 0 ? sum / X.length : 0, count: X.length };
  }

  const leftX: Float64Array[] = [];
  const leftR: number[] = [];
  const rightX: Float64Array[] = [];
  const rightR: number[] = [];

  for (let i = 0; i < X.length; i++) {
    if ((X[i]?.[bestFeature] ?? 0) <= bestThreshold) {
      leftX.push(X[i]!);
      leftR.push(residuals[i] ?? 0);
    } else {
      rightX.push(X[i]!);
      rightR.push(residuals[i] ?? 0);
    }
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(leftX, new Float64Array(leftR), maxDepth - 1, minSamplesLeaf),
    right: buildTree(rightX, new Float64Array(rightR), maxDepth - 1, minSamplesLeaf),
  };
}

function predictTree(node: Node, x: Float64Array): number {
  if (isLeaf(node)) return node.value;
  const v = x[node.featureIndex] ?? 0;
  return v <= node.threshold
    ? predictTree(node.left, x)
    : predictTree(node.right, x);
}

export class HistGradientBoostingRegressor {
  learningRate: number;
  maxIter: number;
  maxDepth: number;
  minSamplesLeaf: number;
  l2Regularization: number;

  private trees_: Node[] = [];
  private initialPrediction_ = 0;
  private isFitted_ = false;

  constructor(
    options: {
      learningRate?: number;
      maxIter?: number;
      maxDepth?: number;
      minSamplesLeaf?: number;
      l2Regularization?: number;
    } = {},
  ) {
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 100;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 1;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = y.length;
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= n;
    this.initialPrediction_ = yMean;

    const F = new Float64Array(n).fill(yMean);
    this.trees_ = [];

    for (let iter = 0; iter < this.maxIter; iter++) {
      const residuals = new Float64Array(n);
      for (let i = 0; i < n; i++) residuals[i] = (y[i] ?? 0) - (F[i] ?? 0);

      const tree = buildTree(X, residuals, this.maxDepth, Math.max(1, this.minSamplesLeaf));
      this.trees_.push(tree);

      for (let i = 0; i < n; i++) {
        F[i] = (F[i] ?? 0) + this.learningRate * predictTree(tree, X[i] ?? new Float64Array(0));
      }
    }

    this.isFitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.isFitted_) throw new NotFittedError("HistGradientBoostingRegressor is not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let pred = this.initialPrediction_;
      for (const tree of this.trees_) {
        pred += this.learningRate * predictTree(tree, X[i] ?? new Float64Array(0));
      }
      out[i] = pred;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= y.length;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}

export class HistGradientBoostingClassifier {
  learningRate: number;
  maxIter: number;
  maxDepth: number;
  minSamplesLeaf: number;

  private regressors_: HistGradientBoostingRegressor[] = [];
  private classes_: number[] = [];
  private isFitted_ = false;

  constructor(
    options: {
      learningRate?: number;
      maxIter?: number;
      maxDepth?: number;
      minSamplesLeaf?: number;
    } = {},
  ) {
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 100;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.classes_ = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.regressors_ = [];

    if (this.classes_.length === 2) {
      // Binary classification
      const yBin = new Float64Array(y.length);
      for (let i = 0; i < y.length; i++) yBin[i] = (y[i] ?? 0) === (this.classes_[1] ?? 1) ? 1 : 0;
      const reg = new HistGradientBoostingRegressor({
        learningRate: this.learningRate,
        maxIter: this.maxIter,
        maxDepth: this.maxDepth,
        minSamplesLeaf: this.minSamplesLeaf,
      });
      reg.fit(X, yBin);
      this.regressors_.push(reg);
    } else {
      // OvR
      for (const c of this.classes_) {
        const yBin = new Float64Array(y.length);
        for (let i = 0; i < y.length; i++) yBin[i] = (y[i] ?? 0) === c ? 1 : 0;
        const reg = new HistGradientBoostingRegressor({
          learningRate: this.learningRate,
          maxIter: this.maxIter,
          maxDepth: this.maxDepth,
          minSamplesLeaf: this.minSamplesLeaf,
        });
        reg.fit(X, yBin);
        this.regressors_.push(reg);
      }
    }

    this.isFitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.isFitted_) throw new NotFittedError("HistGradientBoostingClassifier is not fitted");
    const labels = new Int32Array(X.length);
    if (this.classes_.length === 2) {
      const scores = this.regressors_[0]!.predict(X);
      for (let i = 0; i < X.length; i++) {
        labels[i] = (scores[i] ?? 0) >= 0.5 ? (this.classes_[1] ?? 1) : (this.classes_[0] ?? 0);
      }
    } else {
      const scores = this.regressors_.map((r) => r.predict(X));
      for (let i = 0; i < X.length; i++) {
        let bestClass = 0;
        let bestScore = -Number.POSITIVE_INFINITY;
        for (let c = 0; c < this.classes_.length; c++) {
          const s = scores[c]?.[i] ?? 0;
          if (s > bestScore) {
            bestScore = s;
            bestClass = this.classes_[c] ?? c;
          }
        }
        labels[i] = bestClass;
      }
    }
    return labels;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if ((y[i] ?? 0) === (yPred[i] ?? 0)) correct++;
    }
    return correct / y.length;
  }
}
