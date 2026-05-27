/**
 * Extended ensemble methods: IsolationForest utilities, AdaBoost helpers,
 * voting utilities, and ensemble calibration.
 */

/** Isolation Tree node structure. */
interface IsolationNode {
  feature: number;
  threshold: number;
  left?: IsolationNode;
  right?: IsolationNode;
  size: number;
  isLeaf: boolean;
}

/** Build a single isolation tree. */
function buildIsolationTree(X: Float64Array[], maxDepth: number, depth = 0): IsolationNode {
  const n = X.length;
  if (n <= 1 || depth >= maxDepth) {
    return { feature: 0, threshold: 0, size: n, isLeaf: true };
  }
  const d = X[0]?.length ?? 0;
  const feature = Math.floor(Math.random() * d);
  let minV = Number.POSITIVE_INFINITY;
  let maxV = Number.NEGATIVE_INFINITY;
  for (const xi of X) {
    const v = xi[feature] ?? 0;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  if (minV >= maxV) return { feature, threshold: minV, size: n, isLeaf: true };
  const threshold = minV + Math.random() * (maxV - minV);
  const left = X.filter((xi) => (xi[feature] ?? 0) < threshold);
  const right = X.filter((xi) => (xi[feature] ?? 0) >= threshold);
  return {
    feature,
    threshold,
    left: buildIsolationTree(left, maxDepth, depth + 1),
    right: buildIsolationTree(right, maxDepth, depth + 1),
    size: n,
    isLeaf: false,
  };
}

/** Compute path length in an isolation tree for a sample. */
function pathLength(x: Float64Array, node: IsolationNode, depth = 0): number {
  if (node.isLeaf) return depth + cFactor(node.size);
  const v = x[node.feature] ?? 0;
  if (v < node.threshold) {
    return node.left ? pathLength(x, node.left, depth + 1) : depth + 1;
  }
  return node.right ? pathLength(x, node.right, depth + 1) : depth + 1;
}

/** Average path length of unsuccessful search in BST. */
function cFactor(n: number): number {
  if (n <= 1) return 0;
  return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
}

/** Extended Isolation Forest with anomaly score computation. */
export class IsolationForestScorer {
  private trees: IsolationNode[] = [];
  private nEstimators: number;
  private maxSamples: number;
  private maxDepth: number;
  private avgPathLength_ = 1;

  constructor(nEstimators = 100, maxSamples = 256) {
    this.nEstimators = nEstimators;
    this.maxSamples = maxSamples;
    this.maxDepth = Math.ceil(Math.log2(Math.max(maxSamples, 2)));
  }

  fit(X: Float64Array[]): this {
    this.trees = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const idx = Array.from({ length: Math.min(this.maxSamples, X.length) }, () =>
        Math.floor(Math.random() * X.length));
      const sample = idx.map((i) => X[i] ?? new Float64Array(0));
      this.trees.push(buildIsolationTree(sample, this.maxDepth));
    }
    this.avgPathLength_ = cFactor(Math.min(this.maxSamples, X.length));
    return this;
  }

  /** Anomaly score: < 0.5 is normal, > 0.5 is anomalous. */
  scoreAnomaly(X: Float64Array[]): Float64Array {
    const scores = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      if (xi === undefined) continue;
      let avgLen = 0;
      for (const tree of this.trees) avgLen += pathLength(xi, tree);
      avgLen /= this.trees.length;
      scores[i] = -(2 ** (-avgLen / (this.avgPathLength_ + 1e-10)));
    }
    return scores;
  }

  predict(X: Float64Array[], threshold = -0.5): Int32Array {
    const scores = this.scoreAnomaly(X);
    return Int32Array.from(scores.map((s) => (s < threshold ? -1 : 1)));
  }
}

/** SAMME.R AdaBoost weight update. */
export function sammeRWeightUpdate(
  yTrue: Int32Array,
  classProbas: Float64Array[],
  nClasses: number,
): Float64Array {
  const n = yTrue.length;
  const weights = new Float64Array(n).fill(1 / n);
  for (let i = 0; i < n; i++) {
    const proba = classProbas[i];
    if (proba === undefined) continue;
    const yi = yTrue[i] ?? 0;
    const pyi = Math.max(proba[yi] ?? 1e-10, 1e-10);
    let sum = 0;
    for (let c = 0; c < nClasses; c++) sum += Math.log(Math.max(proba[c] ?? 1e-10, 1e-10));
    weights[i] *= Math.exp(-((nClasses - 1) / nClasses) * (Math.log(pyi) - sum / nClasses));
  }
  // Normalize
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => w / (total + 1e-10));
}

/** Compute sample weights for Bagging. */
export function baggingWeights(n: number, maxSamples: number): Int32Array {
  const counts = new Int32Array(n);
  for (let i = 0; i < maxSamples; i++) {
    counts[Math.floor(Math.random() * n)]++;
  }
  return counts;
}

/** Ensemble diversity: ambiguity decomposition. */
export interface AmbiguityDecomp {
  ensembleMSE: number;
  ambiguity: number;
  averageMSE: number;
}

export function ambiguityDecomposition(
  predictions: Float64Array[],  // one row per estimator
  yTrue: Float64Array,
): AmbiguityDecomp {
  const n = yTrue.length;
  const k = predictions.length;
  const ensemble = new Float64Array(n);
  for (const pred of predictions) {
    for (let i = 0; i < n; i++) ensemble[i] = (ensemble[i] ?? 0) + (pred[i] ?? 0) / k;
  }
  let ensembleMSE = 0;
  for (let i = 0; i < n; i++) ensembleMSE += ((ensemble[i] ?? 0) - (yTrue[i] ?? 0)) ** 2;
  ensembleMSE /= n;

  let avgMSE = 0;
  for (const pred of predictions) {
    for (let i = 0; i < n; i++) avgMSE += ((pred[i] ?? 0) - (yTrue[i] ?? 0)) ** 2;
  }
  avgMSE /= n * k;

  return { ensembleMSE, ambiguity: avgMSE - ensembleMSE, averageMSE: avgMSE };
}
