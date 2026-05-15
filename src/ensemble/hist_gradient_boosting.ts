/**
 * HistGradientBoostingClassifier and HistGradientBoostingRegressor.
 * Mirrors sklearn.ensemble.HistGradientBoostingClassifier/Regressor.
 */

import { NotFittedError } from "../exceptions.js";

export interface HistGradientBoostingOptions {
  loss?: string;
  learningRate?: number;
  maxIter?: number;
  maxLeafNodes?: number;
  maxDepth?: number | null;
  minSamplesLeaf?: number;
  l2Regularization?: number;
  maxBins?: number;
  validationFraction?: number | null;
  nIterNoChange?: number;
  tol?: number;
  randomState?: number;
}

interface HistNode {
  featureIndex: number;
  threshold: number;
  left: HistNode | null;
  right: HistNode | null;
  value: number;
  isLeaf: boolean;
}

function buildTree(
  X: Float64Array[],
  gradients: Float64Array,
  hessians: Float64Array,
  maxLeafNodes: number,
  minSamplesLeaf: number,
  maxDepth: number,
  l2Reg: number,
  indices: Int32Array,
  depth: number
): HistNode {
  const n = indices.length;
  const p = X[0]?.length ?? 0;

  let sumG = 0, sumH = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i]!;
    sumG += gradients[idx] ?? 0;
    sumH += hessians[idx] ?? 0;
  }
  const leafValue = -sumG / (sumH + l2Reg);

  if (n < 2 * minSamplesLeaf || depth >= maxDepth || maxLeafNodes <= 1) {
    return { featureIndex: 0, threshold: 0, left: null, right: null, value: leafValue, isLeaf: true };
  }

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIdx: Int32Array | null = null;
  let bestRightIdx: Int32Array | null = null;

  for (let j = 0; j < p; j++) {
    const vals = Array.from(indices).map((i) => ({ v: X[i]![j] ?? 0, i }));
    vals.sort((a, b) => a.v - b.v);

    let leftG = 0, leftH = 0;
    for (let t = 0; t < n - 1; t++) {
      const idx = vals[t]!.i;
      leftG += gradients[idx] ?? 0;
      leftH += hessians[idx] ?? 0;
      const rightG = sumG - leftG;
      const rightH = sumH - leftH;

      if (leftH + l2Reg < 1e-6 || rightH + l2Reg < 1e-6) continue;
      if (t + 1 < minSamplesLeaf || n - t - 1 < minSamplesLeaf) continue;
      if ((vals[t]!.v) === (vals[t + 1]!.v)) continue;

      const gain = leftG * leftG / (leftH + l2Reg) + rightG * rightG / (rightH + l2Reg) - sumG * sumG / (sumH + l2Reg);
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = j;
        bestThreshold = (vals[t]!.v + vals[t + 1]!.v) / 2;
        const leftIdxArr = new Int32Array(vals.slice(0, t + 1).map((v) => v.i));
        const rightIdxArr = new Int32Array(vals.slice(t + 1).map((v) => v.i));
        bestLeftIdx = leftIdxArr;
        bestRightIdx = rightIdxArr;
      }
    }
  }

  if (bestFeature < 0 || !bestLeftIdx || !bestRightIdx) {
    return { featureIndex: 0, threshold: 0, left: null, right: null, value: leafValue, isLeaf: true };
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(X, gradients, hessians, maxLeafNodes - 1, minSamplesLeaf, maxDepth, l2Reg, bestLeftIdx, depth + 1),
    right: buildTree(X, gradients, hessians, maxLeafNodes - 1, minSamplesLeaf, maxDepth, l2Reg, bestRightIdx, depth + 1),
    value: leafValue,
    isLeaf: false,
  };
}

function predictTree(node: HistNode, x: Float64Array): number {
  if (node.isLeaf) return node.value;
  const v = x[node.featureIndex] ?? 0;
  if (v <= node.threshold) return node.left ? predictTree(node.left, x) : node.value;
  return node.right ? predictTree(node.right, x) : node.value;
}

export class HistGradientBoostingRegressor {
  learningRate: number;
  maxIter: number;
  maxLeafNodes: number;
  maxDepth: number;
  minSamplesLeaf: number;
  l2Regularization: number;
  maxBins: number;
  tol: number;
  randomState: number;
  nIter_: number = 0;

  private _trees: HistNode[] = [];
  private _baseScore: number = 0;

  constructor(options: Partial<HistGradientBoostingOptions> = {}) {
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 100;
    this.maxLeafNodes = options.maxLeafNodes ?? 31;
    this.maxDepth = options.maxDepth ?? 5;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 1.0;
    this.maxBins = options.maxBins ?? 255;
    this.tol = options.tol ?? 1e-7;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this._baseScore = 0;
    for (let i = 0; i < n; i++) this._baseScore += y[i] ?? 0;
    this._baseScore /= n;

    const F = new Float64Array(n).fill(this._baseScore);
    this._trees = [];

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Gradients and hessians (MSE loss)
      const gradients = new Float64Array(n);
      const hessians = new Float64Array(n).fill(1.0);
      for (let i = 0; i < n; i++) gradients[i]! = (F[i] ?? 0) - (y[i] ?? 0);

      const indices = new Int32Array(n).map((_, i) => i);
      const tree = buildTree(X, gradients, hessians, this.maxLeafNodes, this.minSamplesLeaf, this.maxDepth, this.l2Regularization, indices, 0);
      this._trees.push(tree);

      for (let i = 0; i < n; i++) F[i]! += this.learningRate * predictTree(tree, X[i]!);
      this.nIter_ = iter + 1;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this._trees.length === 0) throw new NotFittedError("HistGradientBoostingRegressor is not fitted");
    const n = X.length;
    const out = new Float64Array(n).fill(this._baseScore);
    for (const tree of this._trees) {
      for (let i = 0; i < n; i++) out[i]! += this.learningRate * predictTree(tree, X[i]!);
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let ssTot = 0, ssRes = 0, yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}

export class HistGradientBoostingClassifier {
  learningRate: number;
  maxIter: number;
  maxLeafNodes: number;
  maxDepth: number;
  minSamplesLeaf: number;
  l2Regularization: number;
  maxBins: number;
  tol: number;
  randomState: number;
  nIter_: number = 0;

  private _trees: HistNode[] = [];
  private _baseScore: number = 0;
  private _classes: Int32Array | null = null;

  constructor(options: Partial<HistGradientBoostingOptions> = {}) {
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 100;
    this.maxLeafNodes = options.maxLeafNodes ?? 31;
    this.maxDepth = options.maxDepth ?? 5;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 1.0;
    this.maxBins = options.maxBins ?? 255;
    this.tol = options.tol ?? 1e-7;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classSet = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this._classes = new Int32Array(classSet);

    // Binary classification: encode as {-1, 1}, use log-loss gradients
    const yBin = new Float64Array(n);
    for (let i = 0; i < n; i++) yBin[i]! = (y[i] ?? 0) === (classSet[1] ?? 1) ? 1 : 0;

    // Base score: log-odds of class 1
    let p1 = 0;
    for (let i = 0; i < n; i++) p1 += yBin[i] ?? 0;
    p1 = Math.max(1e-6, Math.min(1 - 1e-6, p1 / n));
    this._baseScore = Math.log(p1 / (1 - p1));

    // F(x) = raw score
    const F = new Float64Array(n).fill(this._baseScore);
    this._trees = [];

    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const gradients = new Float64Array(n);
      const hessians = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const prob = sigmoid(F[i] ?? 0);
        gradients[i]! = prob - (yBin[i] ?? 0);
        hessians[i]! = Math.max(1e-6, prob * (1 - prob));
      }

      const indices = new Int32Array(n).map((_, i) => i);
      const tree = buildTree(X, gradients, hessians, this.maxLeafNodes, this.minSamplesLeaf, this.maxDepth, this.l2Regularization, indices, 0);
      this._trees.push(tree);

      for (let i = 0; i < n; i++) F[i]! += this.learningRate * predictTree(tree, X[i]!);
      this.nIter_ = iter + 1;
    }
    return this;
  }

  private _rawScore(X: Float64Array[]): Float64Array {
    if (this._trees.length === 0) throw new NotFittedError("HistGradientBoostingClassifier is not fitted");
    const n = X.length;
    const out = new Float64Array(n).fill(this._baseScore);
    for (const tree of this._trees) {
      for (let i = 0; i < n; i++) out[i]! += this.learningRate * predictTree(tree, X[i]!);
    }
    return out;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    const raw = this._rawScore(X);
    return raw.map((f) => {
      const p1 = 1 / (1 + Math.exp(-f));
      return new Float64Array([1 - p1, p1]);
    });
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this._classes) throw new NotFittedError("HistGradientBoostingClassifier is not fitted");
    const raw = this._rawScore(X);
    const out = new Int32Array(raw.length);
    const c0 = this._classes[0] ?? 0;
    const c1 = this._classes[1] ?? 1;
    for (let i = 0; i < raw.length; i++) out[i]! = (raw[i] ?? 0) > 0 ? c1 : c0;
    return out;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if ((pred[i] ?? 0) === (y[i] ?? 0)) correct++;
    return correct / y.length;
  }

  get classes_(): Int32Array {
    if (!this._classes) throw new NotFittedError("HistGradientBoostingClassifier is not fitted");
    return this._classes;
  }
}
