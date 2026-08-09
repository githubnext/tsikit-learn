/**
 * Stacking ensembles and blending meta-learners.
 */

export interface StackingEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  predict(X: Float64Array[]): Float64Array | Int32Array;
  score?(X: Float64Array[], y: Float64Array | Int32Array): number;
}

export class StackingRegressor {
  private fittedBaseEstimators_: StackingEstimator[] = [];
  private fittedMetaEstimator_!: StackingEstimator;
  private fitted_ = false;

  constructor(
    private baseEstimators: StackingEstimator[],
    private metaEstimator: StackingEstimator,
    private cv = 5,
    private passthrough = false
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, nBase = this.baseEstimators.length;
    const metaFeatures = Array.from({ length: n }, () => new Float64Array(nBase));
    const foldSize = Math.floor(n / this.cv);

    // Generate out-of-fold predictions for meta-features
    for (let b = 0; b < nBase; b++) {
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize, end = fold === this.cv - 1 ? n : start + foldSize;
        const trainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
        const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
        const XTr = trainIdx.map(i => X[i]!);
        const yTr = new Float64Array(trainIdx.map(i => y[i] ?? 0));
        const XTe = testIdx.map(i => X[i]!);
        const cloned = this._cloneEstimator(this.baseEstimators[b]!);
        cloned.fit(XTr, yTr);
        const preds = cloned.predict(XTe) as Float64Array;
        for (let i = 0; i < testIdx.length; i++) metaFeatures[testIdx[i]!]![b] = preds[i] ?? 0;
      }
    }

    // Fit all base estimators on full data
    this.fittedBaseEstimators_ = this.baseEstimators.map(est => {
      const cloned = this._cloneEstimator(est);
      cloned.fit(X, y);
      return cloned;
    });

    // Build meta-features with passthrough
    const metaX = this.passthrough
      ? metaFeatures.map((mf, i) => new Float64Array([...mf, ...(X[i] ?? [])]))
      : metaFeatures;

    this.fittedMetaEstimator_ = this._cloneEstimator(this.metaEstimator);
    this.fittedMetaEstimator_.fit(metaX, y);
    this.fitted_ = true;
    return this;
  }

  private _cloneEstimator(est: StackingEstimator): StackingEstimator {
    // Shallow clone — create a new instance of the same class
    return Object.create(Object.getPrototypeOf(est), Object.getOwnPropertyDescriptors(est));
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const metaFeatures = X.map(xi => {
      const mf = new Float64Array(this.fittedBaseEstimators_.length);
      for (let b = 0; b < this.fittedBaseEstimators_.length; b++) {
        const pred = this.fittedBaseEstimators_[b]!.predict([xi]) as Float64Array;
        mf[b] = pred[0] ?? 0;
      }
      return this.passthrough ? new Float64Array([...mf, ...xi]) : mf;
    });
    return this.fittedMetaEstimator_.predict(metaFeatures) as Float64Array;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    const ssRes = pred.reduce((s, p, i) => s + ((y[i] ?? 0) - p) ** 2, 0);
    const ssTot = Array.from(y).reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ssRes / (ssTot + 1e-10);
  }
}

export class BlendingRegressor {
  private fittedBase_: StackingEstimator[] = [];
  private fittedMeta_!: StackingEstimator;
  private fitted_ = false;

  constructor(
    private baseEstimators: StackingEstimator[],
    private metaEstimator: StackingEstimator,
    private holdoutFraction = 0.2
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, nHold = Math.floor(n * this.holdoutFraction);
    const holdIdx = Array.from({ length: nHold }, (_, i) => n - nHold + i);
    const trainIdx = Array.from({ length: n - nHold }, (_, i) => i);
    const XTrain = trainIdx.map(i => X[i]!), yTrain = new Float64Array(trainIdx.map(i => y[i] ?? 0));
    const XHold = holdIdx.map(i => X[i]!), yHold = new Float64Array(holdIdx.map(i => y[i] ?? 0));

    this.fittedBase_ = this.baseEstimators.map(est => {
      const e = Object.create(Object.getPrototypeOf(est), Object.getOwnPropertyDescriptors(est));
      e.fit(XTrain, yTrain);
      return e;
    });

    const metaX = XHold.map(xi => new Float64Array(this.fittedBase_.map(e => {
      const p = e.predict([xi]) as Float64Array;
      return p[0] ?? 0;
    })));
    this.fittedMeta_ = Object.create(Object.getPrototypeOf(this.metaEstimator), Object.getOwnPropertyDescriptors(this.metaEstimator));
    this.fittedMeta_.fit(metaX, yHold);
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const metaX = X.map(xi => new Float64Array(this.fittedBase_.map(e => { const p = e.predict([xi]) as Float64Array; return p[0] ?? 0; })));
    return this.fittedMeta_.predict(metaX) as Float64Array;
  }
}
