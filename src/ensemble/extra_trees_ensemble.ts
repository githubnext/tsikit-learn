/**
 * Extremely Randomized Trees (Extra-Trees) ensemble utilities.
 * Mirrors scikit-learn's ensemble.ExtraTreesClassifier/ExtraTreesRegressor.
 */

export interface ExtraTreesOptions {
  nEstimators?: number;
  maxDepth?: number;
  minSamplesSplit?: number;
  maxFeatures?: "sqrt" | "log2" | number | null;
  randomState?: number;
  bootstrap?: boolean;
}

interface ETNode {
  feature: number;
  threshold: number;
  left: ETNode | null;
  right: ETNode | null;
  value: number;
  isLeaf: boolean;
}

function buildExtraTree(
  X: Float64Array[],
  y: Float64Array,
  maxDepth: number,
  minSamplesSplit: number,
  maxFeatures: number,
  rng: () => number,
): ETNode {
  const n = X.length;
  const mean = y.reduce((s, v) => s + v, 0) / n;

  if (n < minSamplesSplit || maxDepth === 0) {
    return { feature: -1, threshold: 0, left: null, right: null, value: mean, isLeaf: true };
  }

  const nFeatures = X[0]?.length ?? 0;
  // Random feature subset
  const featureIndices = Array.from({ length: nFeatures }, (_, i) => i)
    .sort(() => rng() - 0.5)
    .slice(0, maxFeatures);

  let bestFeature = -1;
  let bestThreshold = 0;
  let bestScore = -Number.POSITIVE_INFINITY;

  for (const fi of featureIndices) {
    const values = X.map((row) => row[fi] ?? 0);
    const mn = Math.min(...values);
    const mx = Math.max(...values);
    if (mx - mn < 1e-10) continue;
    // Extremely random: choose threshold uniformly
    const threshold = mn + rng() * (mx - mn);
    const leftY = y.filter((_, i) => (X[i]?.[fi] ?? 0) <= threshold);
    const rightY = y.filter((_, i) => (X[i]?.[fi] ?? 0) > threshold);
    if (leftY.length === 0 || rightY.length === 0) continue;
    const lMean = leftY.reduce((s, v) => s + v, 0) / leftY.length;
    const rMean = rightY.reduce((s, v) => s + v, 0) / rightY.length;
    const score =
      -(leftY.reduce((s, v) => s + (v - lMean) ** 2, 0) +
        rightY.reduce((s, v) => s + (v - rMean) ** 2, 0));
    if (score > bestScore) {
      bestScore = score;
      bestFeature = fi;
      bestThreshold = threshold;
    }
  }

  if (bestFeature === -1) {
    return { feature: -1, threshold: 0, left: null, right: null, value: mean, isLeaf: true };
  }

  const leftMask = X.map((row) => (row[bestFeature] ?? 0) <= bestThreshold);
  const XLeft = X.filter((_, i) => leftMask[i]);
  const yLeft = y.filter((_, i) => leftMask[i]);
  const XRight = X.filter((_, i) => !leftMask[i]);
  const yRight = y.filter((_, i) => !leftMask[i]);

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildExtraTree(XLeft, yLeft, maxDepth - 1, minSamplesSplit, maxFeatures, rng),
    right: buildExtraTree(XRight, yRight, maxDepth - 1, minSamplesSplit, maxFeatures, rng),
    value: mean,
    isLeaf: false,
  };
}

function predictTree(node: ETNode, x: Float64Array): number {
  if (node.isLeaf) return node.value;
  const goLeft = (x[node.feature] ?? 0) <= node.threshold;
  return predictTree(goLeft ? node.left! : node.right!, x);
}

export class ExtraTreesRegressorExt {
  readonly nEstimators: number;
  readonly maxDepth: number;
  readonly minSamplesSplit: number;
  readonly maxFeatures: "sqrt" | "log2" | number | null;
  readonly randomState: number;
  readonly bootstrap: boolean;

  private _trees: ETNode[] = [];

  constructor(options: ExtraTreesOptions = {}) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.randomState = options.randomState ?? 42;
    this.bootstrap = options.bootstrap ?? false;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nFeatures = X[0]?.length ?? 0;
    const maxF =
      this.maxFeatures === "sqrt"
        ? Math.max(1, Math.floor(Math.sqrt(nFeatures)))
        : this.maxFeatures === "log2"
          ? Math.max(1, Math.floor(Math.log2(nFeatures)))
          : this.maxFeatures === null
            ? nFeatures
            : Math.max(1, Math.min(nFeatures, Math.floor(nFeatures * (this.maxFeatures as number))));

    let seed = this.randomState;
    const rng = (): number => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0xffffffff;
    };

    this._trees = Array.from({ length: this.nEstimators }, () => {
      let Xb = X, yb = y;
      if (this.bootstrap) {
        const n = X.length;
        const idx = Array.from({ length: n }, () => Math.floor(rng() * n));
        Xb = idx.map((i) => X[i]!);
        yb = Float64Array.from(idx, (i) => y[i]!);
      }
      return buildExtraTree(Xb, yb, this.maxDepth, this.minSamplesSplit, maxF, rng);
    });
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return Float64Array.from(X, (xi) => {
      const preds = this._trees.map((t) => predictTree(t, xi));
      return preds.reduce((s, v) => s + v, 0) / preds.length;
    });
  }
}
