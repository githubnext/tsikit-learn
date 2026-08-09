/**
 * SelectFromModel, GenericUnivariateSelect, and variance threshold extensions.
 */

export interface FeatureEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  coef?: Float64Array;
  featureImportances?: Float64Array;
}

export class SelectFromModel {
  private selectedMask_!: boolean[];
  private fitted_ = false;

  constructor(
    private estimator: FeatureEstimator,
    private threshold: number | 'mean' | 'median' = 'mean',
    private maxFeatures: number | null = null,
    private prefit = false
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    if (!this.prefit) this.estimator.fit(X, y);
    const importances = this.estimator.featureImportances ?? this.estimator.coef;
    if (!importances) throw new Error('Estimator must have coef_ or feature_importances_');
    const absImps = new Float64Array(importances.map(v => Math.abs(v)));
    let thresh: number;
    if (this.threshold === 'mean') thresh = absImps.reduce((s, v) => s + v, 0) / absImps.length;
    else if (this.threshold === 'median') {
      const sorted = [...absImps].sort((a, b) => a - b);
      thresh = sorted[Math.floor(sorted.length / 2)] ?? 0;
    } else thresh = this.threshold;
    this.selectedMask_ = Array.from(absImps).map(v => v >= thresh);
    if (this.maxFeatures !== null) {
      const ranked = Array.from(absImps).map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
      const keep = new Set(ranked.slice(0, this.maxFeatures).map(x => x.i));
      this.selectedMask_ = this.selectedMask_.map((sel, i) => sel && keep.has(i));
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.filter((_, j) => this.selectedMask_[j])));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] { return this.fit(X, y).transform(X); }
  get selectedMask(): boolean[] { return this.selectedMask_; }
  getSupportMask(): boolean[] { return this.selectedMask_; }
}

export class VarianceThresholdExt {
  private variances_!: Float64Array;
  private mask_!: boolean[];
  private fitted_ = false;

  constructor(private threshold = 0.0) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.variances_ = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const mean = X.reduce((s, row) => s + (row[j] ?? 0), 0) / n;
      this.variances_[j] = X.reduce((s, row) => s + ((row[j] ?? 0) - mean) ** 2, 0) / n;
    }
    this.mask_ = Array.from(this.variances_).map(v => v > this.threshold);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.filter((_, j) => this.mask_[j])));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get variances(): Float64Array { return this.variances_; }
}
