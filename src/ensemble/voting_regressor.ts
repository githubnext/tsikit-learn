/**
 * VotingRegressor and additional ensemble utilities.
 * Mirrors sklearn.ensemble.VotingRegressor.
 */

export interface VotingRegressorOptions {
  estimators: Array<[string, { fit(X: Float64Array[], y: Float64Array): unknown; predict(X: Float64Array[]): Float64Array }]>;
  weights?: Float64Array | null;
  nJobs?: number | null;
  verbose?: boolean;
}

/**
 * Soft Voting/Majority Rule classifier for regression.
 * Returns the weighted average of predictions.
 */
export class VotingRegressor {
  estimators: Array<[string, { fit(X: Float64Array[], y: Float64Array): unknown; predict(X: Float64Array[]): Float64Array }]>;
  weights: Float64Array | null;
  verbose: boolean;
  private fitted_: boolean = false;

  constructor(options: VotingRegressorOptions) {
    this.estimators = options.estimators;
    this.weights = options.weights ?? null;
    this.verbose = options.verbose ?? false;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    for (const [, est] of this.estimators) {
      est.fit(X, y);
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error("VotingRegressor not fitted");
    const nSamples = X.length;
    const nEst = this.estimators.length;
    const allPreds: Float64Array[] = this.estimators.map(([, est]) => est.predict(X));

    const result = new Float64Array(nSamples);
    const totalWeight = this.weights
      ? Array.from(this.weights).reduce((s, v) => s + v, 0)
      : nEst;

    for (let i = 0; i < nSamples; i++) {
      let pred = 0;
      for (let e = 0; e < nEst; e++) {
        const w = this.weights ? (this.weights[e] ?? 1) : 1;
        pred += (allPreds[e]?.[i] ?? 0) * w;
      }
      result[i] = pred / (totalWeight || 1);
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((s, v) => s + v, 0) / y.length;
    let ss_res = 0, ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot < 1e-10 ? 1 : 1 - ss_res / ss_tot;
  }

  getParams(): Record<string, unknown> {
    return { estimators: this.estimators, weights: this.weights };
  }
}

export interface IsolationForestOptions {
  nEstimators?: number;
  maxSamples?: number | "auto";
  contamination?: number | "auto";
  maxFeatures?: number;
  bootstrap?: boolean;
  randomState?: number | null;
  verbose?: number;
}

/**
 * Isolation Forest for anomaly detection.
 */
export class IsolationForest {
  nEstimators: number;
  maxSamples: number | "auto";
  contamination: number | "auto";
  maxFeatures: number;
  bootstrap: boolean;
  randomState: number | null;

  private trees_: Array<{ nodes: IFNode[] }> | null = null;
  private maxDepth_: number = 0;
  private nSamplesFit_: number = 0;
  offset_: number = 0;

  constructor(options: IsolationForestOptions = {}) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxSamples = options.maxSamples ?? "auto";
    this.contamination = options.contamination ?? "auto";
    this.maxFeatures = options.maxFeatures ?? 1;
    this.bootstrap = options.bootstrap ?? false;
    this.randomState = options.randomState ?? null;
  }

  fit(X: Float64Array[]): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nSamplesFit_ = nSamples;

    const sampleSize = this.maxSamples === "auto"
      ? Math.min(256, nSamples)
      : this.maxSamples;
    this.maxDepth_ = Math.ceil(Math.log2(sampleSize + 1));

    let seed = this.randomState ?? 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0xffffffff;
    }
    function randInt(n: number): number { return Math.floor(rand() * n); }

    this.trees_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      // Sample subset
      const idx: number[] = [];
      if (this.bootstrap) {
        for (let i = 0; i < sampleSize; i++) idx.push(randInt(nSamples));
      } else {
        const perm = Array.from({ length: nSamples }, (_, i) => i);
        for (let i = nSamples - 1; i > 0; i--) {
          const j = randInt(i + 1);
          const tmp = perm[i]!; perm[i] = perm[j]!; perm[j] = tmp;
        }
        idx.push(...perm.slice(0, sampleSize));
      }
      const Xsub = idx.map(i => X[i]!);
      this.trees_.push({ nodes: this._buildTree(Xsub, 0, nFeatures, rand) });
    }

    // Compute offset for contamination
    const scores = Array.from({ length: nSamples }, (_, i) => this._scoreOne(X[i]!));
    scores.sort((a, b) => a - b);
    if (this.contamination === "auto") {
      this.offset_ = -0.5;
    } else {
      const idx = Math.floor((1 - this.contamination) * nSamples);
      this.offset_ = -(scores[Math.min(idx, nSamples - 1)] ?? 0);
    }

    return this;
  }

  private _buildTree(X: Float64Array[], depth: number, nFeatures: number, rand: () => number): IFNode[] {
    const nodes: IFNode[] = [];
    this._buildNode(X, depth, nFeatures, rand, nodes);
    return nodes;
  }

  private _buildNode(X: Float64Array[], depth: number, nFeatures: number, rand: () => number, nodes: IFNode[]): number {
    const nodeIdx = nodes.length;
    nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, size: X.length });

    if (X.length <= 1 || depth >= this.maxDepth_) {
      return nodeIdx;
    }

    const featureIdx = Math.floor(rand() * nFeatures);
    let min = X[0]?.[featureIdx] ?? 0;
    let max = min;
    for (const row of X) {
      const v = row[featureIdx] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    if (min === max) return nodeIdx;

    const threshold = min + rand() * (max - min);
    nodes[nodeIdx]!.feature = featureIdx;
    nodes[nodeIdx]!.threshold = threshold;

    const left = X.filter(row => (row[featureIdx] ?? 0) < threshold);
    const right = X.filter(row => (row[featureIdx] ?? 0) >= threshold);

    nodes[nodeIdx]!.left = this._buildNode(left, depth + 1, nFeatures, rand, nodes);
    nodes[nodeIdx]!.right = this._buildNode(right, depth + 1, nFeatures, rand, nodes);

    return nodeIdx;
  }

  private _scoreOne(x: Float64Array): number {
    if (!this.trees_) return 0;
    let totalPathLength = 0;
    for (const tree of this.trees_) {
      totalPathLength += this._pathLength(x, tree.nodes, 0, 0);
    }
    const avgPathLength = totalPathLength / this.trees_.length;
    const normFactor = _cFactor(this.nSamplesFit_);
    return -(2 ** (-avgPathLength / normFactor));
  }

  private _pathLength(x: Float64Array, nodes: IFNode[], nodeIdx: number, depth: number): number {
    const node = nodes[nodeIdx];
    if (!node || node.feature === -1 || node.left === -1 || node.right === -1) {
      return depth + _cFactor(node?.size ?? 1);
    }
    if ((x[node.feature] ?? 0) < node.threshold) {
      return this._pathLength(x, nodes, node.left, depth + 1);
    }
    return this._pathLength(x, nodes, node.right, depth + 1);
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map(x => this._scoreOne(x) + this.offset_));
  }

  predict(X: Float64Array[]): Int32Array {
    const scores = this.decisionFunction(X);
    return new Int32Array(scores.map(s => s < 0 ? -1 : 1));
  }

  scoreAnomalies(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map(x => this._scoreOne(x)));
  }
}

interface IFNode {
  feature: number;
  threshold: number;
  left: number;
  right: number;
  size: number;
}

function _cFactor(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
}
