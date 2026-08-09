/**
 * Extended ensemble: AdaBoostR2, GradientBoostingExt (GBDT with histogram binning)
 */

export interface WeakLearner {
  predict(X: Float64Array[]): Float64Array;
}

export interface WeakClassifier extends WeakLearner {
  fit(X: Float64Array[], y: Int32Array, sampleWeights?: Float64Array): this;
}

export interface WeakRegressor extends WeakLearner {
  fit(X: Float64Array[], y: Float64Array, sampleWeights?: Float64Array): this;
}

/** AdaBoost.R2 for regression */
export class AdaBoostR2 {
  private nEstimators: number;
  private learningRate: number;
  private estimators_: WeakRegressor[] = [];
  private estimatorWeights_: Float64Array | null = null;
  private estimatorFactory: () => WeakRegressor;

  constructor(
    estimatorFactory: () => WeakRegressor,
    nEstimators = 50,
    learningRate = 1.0
  ) {
    this.estimatorFactory = estimatorFactory;
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    let weights = new Float64Array(n).fill(1 / n);
    const estimatorWeights: number[] = [];
    this.estimators_ = [];

    for (let m = 0; m < this.nEstimators; m++) {
      const est = this.estimatorFactory();
      est.fit(X, y, weights);
      const preds = est.predict(X);

      // Compute max error for normalization
      let maxErr = 0;
      for (let i = 0; i < n; i++) {
        const err = Math.abs((y[i] ?? 0) - (preds[i] ?? 0));
        if (err > maxErr) maxErr = err;
      }
      if (maxErr === 0) { this.estimators_.push(est); estimatorWeights.push(1); break; }

      // Normalized losses
      const losses = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        losses[i] = Math.abs((y[i] ?? 0) - (preds[i] ?? 0)) / maxErr;
      }
      const beta = losses.reduce((acc, l, i) => acc + (weights[i] ?? 0) * l, 0);
      if (beta >= 0.5) break;

      const estWeight = this.learningRate * Math.log((1 - beta + 1e-10) / (beta + 1e-10));
      estimatorWeights.push(estWeight);
      this.estimators_.push(est);

      // Update weights
      const newWeights = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        newWeights[i] = (weights[i] ?? 0) * Math.exp(-estWeight * (1 - (losses[i] ?? 0)));
      }
      const wSum = newWeights.reduce((a, b) => a + b, 0) || 1;
      for (let i = 0; i < n; i++) newWeights[i] = (newWeights[i] ?? 0) / wSum;
      weights = newWeights;
    }

    this.estimatorWeights_ = new Float64Array(estimatorWeights);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimatorWeights_) throw new Error("Not fitted");
    const n = X.length;
    const result = new Float64Array(n);
    let totalWeight = 0;
    for (let m = 0; m < this.estimators_.length; m++) {
      const w = this.estimatorWeights_[m] ?? 0;
      totalWeight += w;
      const preds = this.estimators_[m]!.predict(X);
      for (let i = 0; i < n; i++) result[i]! += w * (preds[i] ?? 0);
    }
    for (let i = 0; i < n; i++) result[i]! /= totalWeight || 1;
    return result;
  }
}

/** Gradient Boosting Ext with subsample and feature subsetting */
export class GradientBoostingExt {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private subsample: number;
  private maxFeatures: number | "sqrt" | "log2";
  private estimators_: SimpleTree[] = [];
  private initialPred_: number = 0;
  isFitted_: boolean = false;

  constructor(
    nEstimators = 100,
    learningRate = 0.1,
    maxDepth = 3,
    subsample = 1.0,
    maxFeatures: number | "sqrt" | "log2" = 1.0
  ) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.subsample = subsample;
    this.maxFeatures = maxFeatures;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.initialPred_ = y.reduce((a, b) => a + b, 0) / n;

    let F = new Float64Array(n).fill(this.initialPred_);
    this.estimators_ = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Subsample
      const sampleSize = Math.max(1, Math.floor(this.subsample * n));
      const indices: number[] = [];
      const allIdx = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = allIdx[i]!; allIdx[i] = allIdx[j]!; allIdx[j] = tmp;
      }
      for (let i = 0; i < sampleSize; i++) indices.push(allIdx[i]!);

      // Compute negative gradient (residuals for MSE)
      const residuals = new Float64Array(sampleSize);
      const Xsub: Float64Array[] = [];
      for (let s = 0; s < sampleSize; s++) {
        const idx = indices[s]!;
        residuals[s] = (y[idx] ?? 0) - (F[idx] ?? 0);
        Xsub.push(X[idx]!);
      }

      // Feature subsetting
      const nFeats = typeof this.maxFeatures === "number"
        ? Math.max(1, Math.floor(this.maxFeatures * p))
        : this.maxFeatures === "sqrt" ? Math.max(1, Math.floor(Math.sqrt(p)))
        : Math.max(1, Math.floor(Math.log2(p + 1)));
      const featIdx: number[] = Array.from({ length: p }, (_, i) => i);
      for (let i = p - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = featIdx[i]!; featIdx[i] = featIdx[j]!; featIdx[j] = tmp;
      }
      const selectedFeats = featIdx.slice(0, nFeats);

      const tree = new SimpleTree(this.maxDepth);
      tree.fit(Xsub, residuals, selectedFeats);
      this.estimators_.push(tree);

      // Update F
      const treePreds = tree.predict(X);
      for (let i = 0; i < n; i++) F[i] = (F[i] ?? 0) + this.learningRate * (treePreds[i] ?? 0);
    }

    this.isFitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.isFitted_) throw new Error("Not fitted");
    const n = X.length;
    const result = new Float64Array(n).fill(this.initialPred_);
    for (const tree of this.estimators_) {
      const preds = tree.predict(X);
      for (let i = 0; i < n; i++) result[i]! += this.learningRate * (preds[i] ?? 0);
    }
    return result;
  }
}

class SimpleTree {
  private maxDepth: number;
  private root: TreeSplit | null = null;
  private featIdx: number[] = [];

  constructor(maxDepth = 3) {
    this.maxDepth = maxDepth;
  }

  fit(X: Float64Array[], y: Float64Array, featIdx: number[]): this {
    this.featIdx = featIdx;
    this.root = this.buildNode(X, y, 0);
    return this;
  }

  private buildNode(X: Float64Array[], y: Float64Array, depth: number): TreeSplit {
    const mean = y.reduce((a, b) => a + b, 0) / (y.length || 1);
    if (depth >= this.maxDepth || y.length <= 1) return { mean, feature: -1, threshold: 0, left: null, right: null };

    let bestFeat = -1, bestThresh = 0, bestScore = Number.POSITIVE_INFINITY;
    for (const f of this.featIdx) {
      const vals = [...new Set(Array.from(X).map((row) => row[f] ?? 0))].sort((a, b) => a - b);
      for (let vi = 0; vi < vals.length - 1; vi++) {
        const thresh = ((vals[vi] ?? 0) + (vals[vi + 1] ?? 0)) / 2;
        const leftIdx = X.map((_, i) => i).filter((i) => (X[i]![f] ?? 0) <= thresh);
        const rightIdx = X.map((_, i) => i).filter((i) => (X[i]![f] ?? 0) > thresh);
        if (leftIdx.length === 0 || rightIdx.length === 0) continue;
        const score = this.mse(y, leftIdx) + this.mse(y, rightIdx);
        if (score < bestScore) { bestScore = score; bestFeat = f; bestThresh = thresh; }
      }
    }

    if (bestFeat === -1) return { mean, feature: -1, threshold: 0, left: null, right: null };

    const leftIdx = X.map((_, i) => i).filter((i) => (X[i]![bestFeat] ?? 0) <= bestThresh);
    const rightIdx = X.map((_, i) => i).filter((i) => (X[i]![bestFeat] ?? 0) > bestThresh);
    return {
      mean,
      feature: bestFeat,
      threshold: bestThresh,
      left: this.buildNode(leftIdx.map((i) => X[i]!), new Float64Array(leftIdx.map((i) => y[i] ?? 0)), depth + 1),
      right: this.buildNode(rightIdx.map((i) => X[i]!), new Float64Array(rightIdx.map((i) => y[i] ?? 0)), depth + 1),
    };
  }

  private mse(y: Float64Array, idx: number[]): number {
    if (idx.length === 0) return 0;
    const mean = idx.reduce((acc, i) => acc + (y[i] ?? 0), 0) / idx.length;
    return idx.reduce((acc, i) => acc + ((y[i] ?? 0) - mean) ** 2, 0);
  }

  predict(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((row) => this.predictOne(row, this.root)));
  }

  private predictOne(row: Float64Array, node: TreeSplit | null): number {
    if (!node || node.feature === -1 || !node.left || !node.right) return node?.mean ?? 0;
    return (row[node.feature] ?? 0) <= node.threshold
      ? this.predictOne(row, node.left)
      : this.predictOne(row, node.right);
  }
}

interface TreeSplit {
  mean: number;
  feature: number;
  threshold: number;
  left: TreeSplit | null;
  right: TreeSplit | null;
}
