/**
 * Recursive Feature Elimination (RFE) and RFECV.
 */

export interface RFEEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  coef?: Float64Array;
  featureImportances?: Float64Array;
  score?(X: Float64Array[], y: Float64Array | Int32Array): number;
}

export class RFE {
  private ranking_!: Int32Array;
  private supportMask_!: boolean[];
  private fitted_ = false;

  constructor(
    private estimator: RFEEstimator,
    private nFeaturesToSelect: number | null = null,
    private step = 1
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const p = X[0]?.length ?? 0;
    const nSelect = this.nFeaturesToSelect ?? Math.max(1, Math.floor(p / 2));
    let features = Array.from({ length: p }, (_, i) => i);
    const ranking = new Int32Array(p).fill(1);
    let rank = 1;

    while (features.length > nSelect) {
      const Xsub = X.map(row => new Float64Array(features.map(j => row[j] ?? 0)));
      this.estimator.fit(Xsub, y);
      const importances = (this.estimator.featureImportances ?? this.estimator.coef);
      if (!importances) break;
      const absImps = Array.from(importances).map(v => Math.abs(v));
      const nRemove = Math.min(
        typeof this.step === 'number' && this.step >= 1 ? Math.floor(this.step) : Math.max(1, Math.floor(features.length * this.step)),
        features.length - nSelect
      );
      const sorted = absImps.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const removeLocal = new Set(sorted.slice(0, nRemove).map(x => x.i));
      const toRemove = new Set(features.filter((_, li) => removeLocal.has(li)));
      rank++;
      for (const f of toRemove) ranking[f] = rank;
      features = features.filter(f => !toRemove.has(f));
    }

    this.ranking_ = ranking;
    this.supportMask_ = Array.from({ length: p }, (_, i) => ranking[i] === 1);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.filter((_, j) => this.supportMask_[j])));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] { return this.fit(X, y).transform(X); }
  get ranking(): Int32Array { return this.ranking_; }
  get support(): boolean[] { return this.supportMask_; }
}

export class RFECV extends RFE {
  private cvScores_!: Float64Array;

  constructor(
    estimator: RFEEstimator,
    private minFeaturesToSelect = 1,
    private cv = 5,
    step = 1
  ) {
    super(estimator, null, step);
  }

  fitWithCV(X: Float64Array[], y: Float64Array | Int32Array): this {
    const p = X[0]?.length ?? 0, n = X.length;
    const nFeaturesArr = Array.from({ length: p - this.minFeaturesToSelect + 1 }, (_, i) => p - i);
    this.cvScores_ = new Float64Array(nFeaturesArr.length);

    for (let fi = 0; fi < nFeaturesArr.length; fi++) {
      const nF = nFeaturesArr[fi]!;
      const foldSize = Math.floor(n / this.cv);
      let totalScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize, end = fold === this.cv - 1 ? n : start + foldSize;
        const trainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
        const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
        const XTr = trainIdx.map(i => X[i]!), yTr = y instanceof Int32Array ? new Int32Array(trainIdx.map(i => y[i] ?? 0)) : new Float64Array(trainIdx.map(i => (y as Float64Array)[i] ?? 0));
        const XTe = testIdx.map(i => X[i]!), yTe = y instanceof Int32Array ? new Int32Array(testIdx.map(i => y[i] ?? 0)) : new Float64Array(testIdx.map(i => (y as Float64Array)[i] ?? 0));
        const rfe = new RFE(this['estimator' as keyof typeof this] as RFEEstimator, nF, this['step' as keyof typeof this] as number);
        rfe.fit(XTr, yTr);
        const XTrSel = rfe.transform(XTr), XTeSel = rfe.transform(XTe);
        (this['estimator' as keyof typeof this] as RFEEstimator).fit(XTrSel, yTr);
        totalScore += (this['estimator' as keyof typeof this] as RFEEstimator).score?.(XTeSel, yTe) ?? 0;
      }
      this.cvScores_[fi] = totalScore / this.cv;
    }
    const bestIdx = Array.from(this.cvScores_).reduce((best, v, i) => v > (this.cvScores_[best] ?? 0) ? i : best, 0);
    this['nFeaturesToSelect' as keyof typeof this] = nFeaturesArr[bestIdx] as unknown as this[keyof this];
    return this.fit(X, y);
  }

  get cvScores(): Float64Array { return this.cvScores_; }
}
