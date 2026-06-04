/**
 * Ensemble extensions: ExtraTrees, Rotation Forest, Feature Importance via permutation
 */

export class ExtraTreesClassifierExt {
  private trees_: Array<{ tree: ExtraTreeNode; features: number[] }> = [];
  private classes_: Int32Array = new Int32Array(0);
  private fitted_ = false;

  constructor(
    private nEstimators: number = 100,
    private maxDepth: number = 10,
    private minSamplesSplit: number = 2,
    private maxFeatures: number | 'sqrt' | 'log2' = 'sqrt',
    private randomState: number = 42
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    this.classes_ = new Int32Array([...new Set(Array.from(y))].sort((a, b) => a - b));
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

    const n = X.length, p = X[0]?.length ?? 0;
    const nFeatures = this._nFeatures(p);

    this.trees_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const features: number[] = [];
      while (features.length < nFeatures) {
        const f = Math.floor(rand() * p);
        if (!features.includes(f)) features.push(f);
      }
      const subX = X.map(row => new Float64Array(features.map(f => row[f] ?? 0)));
      const tree = this._buildTree(subX, y, 0, rand);
      this.trees_.push({ tree, features });
    }
    this.fitted_ = true;
    return this;
  }

  private _nFeatures(p: number): number {
    if (typeof this.maxFeatures === 'number') return this.maxFeatures;
    if (this.maxFeatures === 'sqrt') return Math.max(1, Math.round(Math.sqrt(p)));
    return Math.max(1, Math.round(Math.log2(p)));
  }

  private _buildTree(X: Float64Array[], y: Int32Array, depth: number, rand: () => number): ExtraTreeNode {
    const n = X.length;
    if (depth >= this.maxDepth || n < this.minSamplesSplit || this._isPure(y)) {
      return { leaf: true, classValue: this._majorityClass(y) };
    }
    const p = X[0]?.length ?? 0;
    let bestFeature = 0, bestThreshold = 0, bestGain = -1;
    for (let j = 0; j < p; j++) {
      const vals = X.map(row => row[j] ?? 0);
      const minV = Math.min(...vals), maxV = Math.max(...vals);
      const threshold = minV + rand() * (maxV - minV); // Extra-trees: random threshold
      const gain = this._informationGain(y, X, j, threshold);
      if (gain > bestGain) { bestGain = gain; bestFeature = j; bestThreshold = threshold; }
    }
    const leftIdx: number[] = [], rightIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((X[i]?.[bestFeature] ?? 0) <= bestThreshold) leftIdx.push(i); else rightIdx.push(i);
    }
    if (leftIdx.length === 0 || rightIdx.length === 0) return { leaf: true, classValue: this._majorityClass(y) };
    const leftX = leftIdx.map(i => X[i]!), leftY = new Int32Array(leftIdx.map(i => y[i] ?? 0));
    const rightX = rightIdx.map(i => X[i]!), rightY = new Int32Array(rightIdx.map(i => y[i] ?? 0));
    return {
      leaf: false, feature: bestFeature, threshold: bestThreshold,
      left: this._buildTree(leftX, leftY, depth + 1, rand),
      right: this._buildTree(rightX, rightY, depth + 1, rand)
    };
  }

  private _isPure(y: Int32Array): boolean {
    return new Set(Array.from(y)).size === 1;
  }

  private _majorityClass(y: Int32Array): number {
    const counts = new Map<number, number>();
    for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
    let best = y[0] ?? 0, bestCount = 0;
    for (const [k, c] of counts) if (c > bestCount) { best = k; bestCount = c; }
    return best;
  }

  private _informationGain(y: Int32Array, X: Float64Array[], j: number, threshold: number): number {
    const leftIdx = [], rightIdx = [];
    for (let i = 0; i < y.length; i++) {
      if ((X[i]?.[j] ?? 0) <= threshold) leftIdx.push(i); else rightIdx.push(i);
    }
    if (leftIdx.length === 0 || rightIdx.length === 0) return 0;
    const leftY = new Int32Array(leftIdx.map(i => y[i] ?? 0));
    const rightY = new Int32Array(rightIdx.map(i => y[i] ?? 0));
    return this._gini(y) - (leftY.length / y.length) * this._gini(leftY) - (rightY.length / y.length) * this._gini(rightY);
  }

  private _gini(y: Int32Array): number {
    const counts = new Map<number, number>();
    for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
    let g = 1;
    for (const c of counts.values()) g -= (c / y.length) ** 2;
    return g;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(row => {
      const votes = new Map<number, number>();
      for (const { tree, features } of this.trees_) {
        const subRow = new Float64Array(features.map(f => row[f] ?? 0));
        const pred = this._predictTree(tree, subRow);
        votes.set(pred, (votes.get(pred) ?? 0) + 1);
      }
      let best = 0, bestCount = 0;
      for (const [k, c] of votes) if (c > bestCount) { best = k; bestCount = c; }
      return best;
    }));
  }

  private _predictTree(node: ExtraTreeNode, row: Float64Array): number {
    if (node.leaf) return node.classValue ?? 0;
    if ((row[node.feature ?? 0] ?? 0) <= (node.threshold ?? 0)) return this._predictTree(node.left!, row);
    return this._predictTree(node.right!, row);
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const nClasses = this.classes_.length;
    return X.map(row => {
      const votes = new Float64Array(nClasses);
      for (const { tree, features } of this.trees_) {
        const subRow = new Float64Array(features.map(f => row[f] ?? 0));
        const pred = this._predictTree(tree, subRow);
        const idx = Array.from(this.classes_).indexOf(pred);
        if (idx >= 0) votes[idx] = (votes[idx] ?? 0) + 1;
      }
      const total = this.nEstimators;
      return new Float64Array(votes.map(v => v / total));
    });
  }
}

interface ExtraTreeNode {
  leaf: boolean;
  classValue?: number;
  feature?: number;
  threshold?: number;
  left?: ExtraTreeNode;
  right?: ExtraTreeNode;
}

export class ExtraTreesRegressorExt {
  private trees_: Array<{ tree: ExtraTreeRegNode; features: number[] }> = [];
  private fitted_ = false;

  constructor(
    private nEstimators: number = 100,
    private maxDepth: number = 10,
    private minSamplesSplit: number = 2,
    private maxFeatures: number | 'sqrt' | 'log2' = 'sqrt',
    private randomState: number = 42
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
    const p = X[0]?.length ?? 0;
    const nFeatures = typeof this.maxFeatures === 'number' ? this.maxFeatures : Math.max(1, Math.round(Math.sqrt(p)));

    this.trees_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const features: number[] = [];
      while (features.length < nFeatures) {
        const f = Math.floor(rand() * p);
        if (!features.includes(f)) features.push(f);
      }
      const subX = X.map(row => new Float64Array(features.map(f => row[f] ?? 0)));
      const tree = this._buildTree(subX, y, 0, rand);
      this.trees_.push({ tree, features });
    }
    this.fitted_ = true;
    return this;
  }

  private _buildTree(X: Float64Array[], y: Float64Array, depth: number, rand: () => number): ExtraTreeRegNode {
    const n = X.length;
    const mean = y.reduce((s, v) => s + v, 0) / n;
    if (depth >= this.maxDepth || n < this.minSamplesSplit) return { leaf: true, value: mean };

    const p = X[0]?.length ?? 0;
    let bestFeature = 0, bestThreshold = 0, bestVar = Number.POSITIVE_INFINITY;
    for (let j = 0; j < p; j++) {
      const vals = X.map(row => row[j] ?? 0);
      const minV = Math.min(...vals), maxV = Math.max(...vals);
      const threshold = minV + rand() * (maxV - minV);
      let varL = 0, varR = 0, nL = 0, nR = 0, mL = 0, mR = 0;
      for (let i = 0; i < n; i++) {
        if ((X[i]?.[j] ?? 0) <= threshold) { mL += y[i] ?? 0; nL++; } else { mR += y[i] ?? 0; nR++; }
      }
      if (nL === 0 || nR === 0) continue;
      mL /= nL; mR /= nR;
      for (let i = 0; i < n; i++) {
        if ((X[i]?.[j] ?? 0) <= threshold) varL += ((y[i] ?? 0) - mL) ** 2;
        else varR += ((y[i] ?? 0) - mR) ** 2;
      }
      const totalVar = varL + varR;
      if (totalVar < bestVar) { bestVar = totalVar; bestFeature = j; bestThreshold = threshold; }
    }
    const leftIdx: number[] = [], rightIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((X[i]?.[bestFeature] ?? 0) <= bestThreshold) leftIdx.push(i); else rightIdx.push(i);
    }
    if (leftIdx.length === 0 || rightIdx.length === 0) return { leaf: true, value: mean };
    return {
      leaf: false, feature: bestFeature, threshold: bestThreshold,
      left: this._buildTree(leftIdx.map(i => X[i]!), new Float64Array(leftIdx.map(i => y[i] ?? 0)), depth + 1, rand),
      right: this._buildTree(rightIdx.map(i => X[i]!), new Float64Array(rightIdx.map(i => y[i] ?? 0)), depth + 1, rand)
    };
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => {
      let sum = 0;
      for (const { tree, features } of this.trees_) {
        const subRow = new Float64Array(features.map(f => row[f] ?? 0));
        sum += this._predictTree(tree, subRow);
      }
      return sum / this.nEstimators;
    }));
  }

  private _predictTree(node: ExtraTreeRegNode, row: Float64Array): number {
    if (node.leaf) return node.value ?? 0;
    if ((row[node.feature ?? 0] ?? 0) <= (node.threshold ?? 0)) return this._predictTree(node.left!, row);
    return this._predictTree(node.right!, row);
  }
}

interface ExtraTreeRegNode {
  leaf: boolean;
  value?: number;
  feature?: number;
  threshold?: number;
  left?: ExtraTreeRegNode;
  right?: ExtraTreeRegNode;
}

export function permutationImportance(
  model: { predict: (X: Float64Array[]) => Float64Array | Int32Array },
  X: Float64Array[],
  y: Float64Array | Int32Array,
  scoreFn: (yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number,
  nRepeats: number = 5,
  randomState: number = 42
): { importances: Float64Array; importancesMean: Float64Array; importancesStd: Float64Array } {
  const p = X[0]?.length ?? 0;
  const baseScore = scoreFn(y, model.predict(X));
  const importances = new Float64Array(p * nRepeats);
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

  for (let j = 0; j < p; j++) {
    for (let r = 0; r < nRepeats; r++) {
      const Xperm = X.map(row => row.slice());
      // Shuffle column j
      const colVals = Xperm.map(row => row[j] ?? 0);
      for (let i = colVals.length - 1; i > 0; i--) {
        const k = Math.floor(rand() * (i + 1));
        const tmp = colVals[i]!; colVals[i] = colVals[k]!; colVals[k] = tmp;
      }
      for (let i = 0; i < Xperm.length; i++) Xperm[i]![j] = colVals[i] ?? 0;
      const permScore = scoreFn(y, model.predict(Xperm));
      importances[j * nRepeats + r] = baseScore - permScore;
    }
  }

  const mean = new Float64Array(p);
  const std = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const vals = Array.from({ length: nRepeats }, (_, r) => importances[j * nRepeats + r] ?? 0);
    mean[j] = vals.reduce((s, v) => s + v, 0) / nRepeats;
    std[j] = Math.sqrt(vals.reduce((s, v) => s + (v - (mean[j] ?? 0)) ** 2, 0) / nRepeats);
  }
  return { importances, importancesMean: mean, importancesStd: std };
}
