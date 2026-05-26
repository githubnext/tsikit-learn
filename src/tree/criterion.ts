/**
 * Tree splitting criteria and splitter utilities.
 */

export interface SplitRecord {
  featureIndex: number;
  threshold: number;
  impurityImprovement: number;
  leftSize: number;
  rightSize: number;
}

export function giniImpurity(classCounts: Int32Array, total: number): number {
  if (total === 0) return 0;
  let g = 1;
  for (const c of classCounts) g -= (c / total) ** 2;
  return g;
}

export function entropyImpurity(classCounts: Int32Array, total: number): number {
  if (total === 0) return 0;
  let h = 0;
  for (const c of classCounts) {
    if (c > 0) h -= (c / total) * Math.log2(c / total);
  }
  return h;
}

export function mseCriterion(y: Float64Array): number {
  const n = y.length;
  if (n === 0) return 0;
  const mean = y.reduce((a, b) => a + b, 0) / n;
  return y.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
}

export function maeCriterion(y: Float64Array): number {
  const n = y.length;
  if (n === 0) return 0;
  const sorted = new Float64Array(y).sort();
  const median = n % 2 === 0 ? ((sorted[n / 2 - 1] ?? 0) + (sorted[n / 2] ?? 0)) / 2 : (sorted[Math.floor(n / 2)] ?? 0);
  return y.reduce((s, v) => s + Math.abs(v - median), 0) / n;
}

export class BestSplitter {
  findBestSplit(
    X: Float64Array[],
    y: Float64Array | Int32Array,
    featureIndices?: number[],
    criterion: "gini" | "entropy" | "mse" | "mae" = "mse"
  ): SplitRecord | null {
    const nF = X[0]?.length ?? 0;
    const features = featureIndices ?? Array.from({ length: nF }, (_, i) => i);
    let bestSplit: SplitRecord | null = null;
    const parentImpurity = this._impurity(y, criterion);

    for (const fi of features) {
      const values = X.map((x) => x[fi] ?? 0);
      const thresholds = [...new Set(values)].sort((a, b) => a - b);
      for (let ti = 0; ti < thresholds.length - 1; ti++) {
        const t = ((thresholds[ti] ?? 0) + (thresholds[ti + 1] ?? 0)) / 2;
        const leftIdx = values.map((v, i) => v <= t ? i : -1).filter((i) => i >= 0);
        const rightIdx = values.map((v, i) => v > t ? i : -1).filter((i) => i >= 0);
        if (leftIdx.length === 0 || rightIdx.length === 0) continue;
        const yLeft = this._subset(y, leftIdx);
        const yRight = this._subset(y, rightIdx);
        const n = y.length;
        const improvement = parentImpurity
          - leftIdx.length / n * this._impurity(yLeft, criterion)
          - rightIdx.length / n * this._impurity(yRight, criterion);
        if (bestSplit === null || improvement > bestSplit.impurityImprovement) {
          bestSplit = { featureIndex: fi, threshold: t, impurityImprovement: improvement, leftSize: leftIdx.length, rightSize: rightIdx.length };
        }
      }
    }
    return bestSplit;
  }

  private _subset(y: Float64Array | Int32Array, indices: number[]): Float64Array | Int32Array {
    if (y instanceof Int32Array) return new Int32Array(indices.map((i) => y[i] ?? 0));
    return new Float64Array(indices.map((i) => y[i] ?? 0));
  }

  private _impurity(y: Float64Array | Int32Array, criterion: "gini" | "entropy" | "mse" | "mae"): number {
    if (criterion === "mse") return mseCriterion(y instanceof Float64Array ? y : new Float64Array(y));
    if (criterion === "mae") return maeCriterion(y instanceof Float64Array ? y : new Float64Array(y));
    const counts = new Map<number, number>();
    for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
    const classCounts = new Int32Array([...counts.values()]);
    if (criterion === "gini") return giniImpurity(classCounts, y.length);
    return entropyImpurity(classCounts, y.length);
  }
}

export class RandomSplitter {
  constructor(private readonly maxFeatures: number | "sqrt" | "log2" | "auto" = "sqrt", private readonly seed = 42) {}

  findBestSplit(
    X: Float64Array[],
    y: Float64Array | Int32Array,
    criterion: "gini" | "entropy" | "mse" | "mae" = "mse"
  ): SplitRecord | null {
    const nF = X[0]?.length ?? 0;
    const k = this.maxFeatures === "sqrt" ? Math.ceil(Math.sqrt(nF))
      : this.maxFeatures === "log2" ? Math.ceil(Math.log2(Math.max(nF, 2)))
      : this.maxFeatures === "auto" ? Math.ceil(Math.sqrt(nF))
      : Math.min(this.maxFeatures, nF);
    const rng = this._seededRng(this.seed);
    const allFeatures = Array.from({ length: nF }, (_, i) => i);
    // Shuffle and take k
    for (let i = allFeatures.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [allFeatures[i], allFeatures[j]] = [allFeatures[j]!, allFeatures[i]!];
    }
    const selectedFeatures = allFeatures.slice(0, k);
    const splitter = new BestSplitter();
    return splitter.findBestSplit(X, y, selectedFeatures, criterion);
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class ExtraTreeSplitter {
  constructor(private readonly seed = 42) {}

  findRandomSplit(
    X: Float64Array[],
    y: Float64Array | Int32Array,
    criterion: "gini" | "entropy" | "mse" | "mae" = "mse"
  ): SplitRecord | null {
    const nF = X[0]?.length ?? 0;
    if (nF === 0) return null;
    const rng = this._seededRng(this.seed);
    const fi = Math.floor(rng() * nF);
    const values = X.map((x) => x[fi] ?? 0);
    const minV = Math.min(...values), maxV = Math.max(...values);
    if (minV >= maxV) return null;
    const t = minV + rng() * (maxV - minV);
    const leftIdx = values.map((v, i) => v <= t ? i : -1).filter((i) => i >= 0);
    const rightIdx = values.map((v, i) => v > t ? i : -1).filter((i) => i >= 0);
    if (leftIdx.length === 0 || rightIdx.length === 0) return null;
    const parentImpurity = this._impurity(y, criterion);
    const yLeft = leftIdx.map((i) => y[i] ?? 0);
    const yRight = rightIdx.map((i) => y[i] ?? 0);
    const yLArr = y instanceof Int32Array ? new Int32Array(yLeft) : new Float64Array(yLeft);
    const yRArr = y instanceof Int32Array ? new Int32Array(yRight) : new Float64Array(yRight);
    const n = y.length;
    const improvement = parentImpurity - leftIdx.length / n * this._impurity(yLArr, criterion) - rightIdx.length / n * this._impurity(yRArr, criterion);
    return { featureIndex: fi, threshold: t, impurityImprovement: improvement, leftSize: leftIdx.length, rightSize: rightIdx.length };
  }

  private _impurity(y: Float64Array | Int32Array, criterion: "gini" | "entropy" | "mse" | "mae"): number {
    if (criterion === "mse") return mseCriterion(y instanceof Float64Array ? y : new Float64Array(y));
    if (criterion === "mae") return maeCriterion(y instanceof Float64Array ? y : new Float64Array(y));
    const counts = new Map<number, number>();
    for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
    const classCounts = new Int32Array([...counts.values()]);
    if (criterion === "gini") return giniImpurity(classCounts, y.length);
    return entropyImpurity(classCounts, y.length);
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}
