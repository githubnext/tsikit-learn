/**
 * Ensemble learning base and ExtraTreesBooster extension.
 */

export interface BaseEstimatorExt {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  predict(X: Float64Array[]): Float64Array | Int32Array;
}

export abstract class EnsembleBase implements BaseEstimatorExt {
  estimators_: BaseEstimatorExt[] = [];
  weights_: Float64Array | null = null;

  abstract fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  abstract predict(X: Float64Array[]): Float64Array | Int32Array;

  get nEstimators(): number {
    return this.estimators_.length;
  }
}

export class VotingClassifier extends EnsembleBase {
  estimators: Array<[string, BaseEstimatorExt]>;
  voting: "hard" | "soft";
  nClasses: number;

  constructor(estimators: Array<[string, BaseEstimatorExt]>, voting: "hard" | "soft" = "hard", nClasses = 2) {
    super();
    this.estimators = estimators;
    this.voting = voting;
    this.nClasses = nClasses;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    this.estimators_ = this.estimators.map(([, est]) => est.fit(X, y));
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const votes: Int32Array[] = this.estimators_.map((est) => est.predict(X) as Int32Array);
    return Int32Array.from({ length: n }, (_, i) => {
      const counts = new Int32Array(this.nClasses);
      for (const v of votes) counts[v[i] ?? 0]++;
      let best = 0, bestCount = 0;
      for (let c = 0; c < this.nClasses; c++) if ((counts[c] ?? 0) > bestCount) { bestCount = counts[c] ?? 0; best = c; }
      return best;
    });
  }
}

export class VotingRegressor extends EnsembleBase {
  estimators: Array<[string, BaseEstimatorExt]>;

  constructor(estimators: Array<[string, BaseEstimatorExt]>) {
    super();
    this.estimators = estimators;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    this.estimators_ = this.estimators.map(([, est]) => est.fit(X, y));
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const n = X.length;
    const preds = this.estimators_.map((est) => est.predict(X) as Float64Array);
    return Float64Array.from({ length: n }, (_, i) => preds.reduce((s, p) => s + (p[i] ?? 0), 0) / preds.length);
  }
}

export class ExtraTreesBooster extends EnsembleBase {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  private _fitEstimators: number = 0;

  constructor(nEstimators = 100, maxDepth = 5, minSamplesSplit = 2) {
    super();
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  private _sampleFeatures(p: number): number[] {
    const nFeatures = Math.max(1, Math.round(Math.sqrt(p)));
    const indices = Array.from({ length: p }, (_, i) => i);
    for (let i = p - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = indices[i]; indices[i] = indices[j] as number; indices[j] = t as number;
    }
    return indices.slice(0, nFeatures);
  }

  private _buildTree(
    X: Float64Array[], y: Float64Array, depth = 0
  ): { leaf: boolean; value?: number; featureIdx?: number; threshold?: number; left?: unknown; right?: unknown } {
    if (depth >= this.maxDepth || X.length < this.minSamplesSplit) {
      const value = y.reduce((s, v) => s + v, 0) / Math.max(y.length, 1);
      return { leaf: true, value };
    }
    const p = X[0]?.length ?? 0;
    const featureCandidates = this._sampleFeatures(p);
    let bestFeat = 0, bestThresh = 0, bestGain = -Number.POSITIVE_INFINITY;

    for (const feat of featureCandidates) {
      const vals = X.map((row) => row[feat] ?? 0);
      const mn = Math.min(...vals), mx = Math.max(...vals);
      if (mn === mx) continue;
      // Randomly select threshold (extra trees style)
      const thresh = mn + Math.random() * (mx - mn);
      const leftY = y.filter((_, i) => (X[i]?.[feat] ?? 0) <= thresh);
      const rightY = y.filter((_, i) => (X[i]?.[feat] ?? 0) > thresh);
      if (leftY.length === 0 || rightY.length === 0) continue;
      const leftMean = leftY.reduce((s, v) => s + v, 0) / leftY.length;
      const rightMean = rightY.reduce((s, v) => s + v, 0) / rightY.length;
      const gain = leftY.reduce((s, v) => s + v * v, 0) / leftY.length - leftMean ** 2
        + rightY.reduce((s, v) => s + v * v, 0) / rightY.length - rightMean ** 2;
      if (-gain > bestGain) { bestGain = -gain; bestFeat = feat; bestThresh = thresh; }
    }

    if (bestGain === -Number.POSITIVE_INFINITY) {
      return { leaf: true, value: y.reduce((s, v) => s + v, 0) / Math.max(y.length, 1) };
    }

    const leftMask = X.map((row) => (row[bestFeat] ?? 0) <= bestThresh);
    const leftX = X.filter((_, i) => leftMask[i]);
    const rightX = X.filter((_, i) => !leftMask[i]);
    const leftY2 = y.filter((_, i) => leftMask[i]);
    const rightY2 = y.filter((_, i) => !leftMask[i]);

    return {
      leaf: false,
      featureIdx: bestFeat,
      threshold: bestThresh,
      left: this._buildTree(leftX, leftY2, depth + 1),
      right: this._buildTree(rightX, rightY2, depth + 1),
    };
  }

  private _predictTree(tree: ReturnType<ExtraTreesBooster["_buildTree"]>, x: Float64Array): number {
    if (tree.leaf) return tree.value ?? 0;
    const feat = tree.featureIdx ?? 0;
    if ((x[feat] ?? 0) <= (tree.threshold ?? 0)) {
      return this._predictTree(tree.left as ReturnType<ExtraTreesBooster["_buildTree"]>, x);
    }
    return this._predictTree(tree.right as ReturnType<ExtraTreesBooster["_buildTree"]>, x);
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const yF = y instanceof Float64Array ? y : Float64Array.from(y);
    this.estimators_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const tree = this._buildTree(X, yF);
      this.estimators_.push({
        fit: (_X: Float64Array[], _y: Float64Array | Int32Array) => ({} as BaseEstimatorExt),
        predict: (Xnew: Float64Array[]) => Float64Array.from(Xnew, (row) => this._predictTree(tree, row)),
      });
    }
    this._fitEstimators = this.nEstimators;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_.length === 0) throw new Error("Not fitted");
    const preds = this.estimators_.map((est) => est.predict(X) as Float64Array);
    return Float64Array.from({ length: X.length }, (_, i) => preds.reduce((s, p) => s + (p[i] ?? 0), 0) / preds.length);
  }
}

export class BaggingRegressor extends EnsembleBase {
  baseEstimatorFactory: () => BaseEstimatorExt;
  nEstimators: number;
  maxSamples: number;
  maxFeatures: number;
  private _nFeatures: number = 0;

  constructor(
    baseEstimatorFactory: () => BaseEstimatorExt,
    nEstimators = 10,
    maxSamples = 1.0,
    maxFeatures = 1.0,
  ) {
    super();
    this.baseEstimatorFactory = baseEstimatorFactory;
    this.nEstimators = nEstimators;
    this.maxSamples = maxSamples;
    this.maxFeatures = maxFeatures;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this._nFeatures = p;
    const nSamples = Math.max(1, Math.round(this.maxSamples * n));
    const nFeatures = Math.max(1, Math.round(this.maxFeatures * p));
    this.estimators_ = Array.from({ length: this.nEstimators }, () => {
      const est = this.baseEstimatorFactory();
      const sampleIdx = Array.from({ length: nSamples }, () => Math.floor(Math.random() * n));
      const featIdx = Array.from({ length: p }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, nFeatures);
      const Xboot = sampleIdx.map((i) => Float64Array.from(featIdx, (j) => (X[i]?.[j] ?? 0)));
      const yboot = y instanceof Int32Array
        ? Int32Array.from(sampleIdx, (i) => y[i] ?? 0)
        : Float64Array.from(sampleIdx, (i) => y[i] ?? 0);
      est.fit(Xboot, yboot);
      return { fit: (Xn: Float64Array[], yn: Float64Array | Int32Array) => est.fit(Xn.map((row) => Float64Array.from(featIdx, (j) => row[j] ?? 0)), yn), predict: (Xn: Float64Array[]) => est.predict(Xn.map((row) => Float64Array.from(featIdx, (j) => row[j] ?? 0))) };
    });
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const preds = this.estimators_.map((est) => est.predict(X) as Float64Array);
    return Float64Array.from({ length: X.length }, (_, i) => preds.reduce((s, p) => s + (p[i] ?? 0), 0) / preds.length);
  }
}
