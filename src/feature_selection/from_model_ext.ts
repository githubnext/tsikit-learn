/**
 * Feature selection from model: SelectFromModel, VarianceThreshold, SelectPercentile.
 */

export interface FittedEstimatorWithImportance {
  featureImportances_?: Float64Array;
  coef_?: Float64Array | Float64Array[];
}

export class SelectFromModel {
  private threshold_: number | null = null;
  private mask_: boolean[] = [];
  private nFeatures = 0;

  constructor(
    private readonly estimator: FittedEstimatorWithImportance,
    private readonly threshold: number | "mean" | "median" = "mean",
    private readonly maxFeatures?: number
  ) {}

  fit(X: Float64Array[], _y?: unknown): this {
    this.nFeatures = X[0]?.length ?? 0;
    const importances = this._getImportances();
    if (importances === null) {
      this.mask_ = new Array(this.nFeatures).fill(true) as boolean[];
      return this;
    }
    let thresh: number;
    if (this.threshold === "mean") {
      thresh = importances.reduce((a, b) => a + b, 0) / importances.length;
    } else if (this.threshold === "median") {
      const sorted = [...importances].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      thresh = sorted.length % 2 === 0 ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2 : (sorted[mid] ?? 0);
    } else {
      thresh = this.threshold;
    }
    this.threshold_ = thresh;
    this.mask_ = Array.from(importances, (v) => v >= thresh);
    if (this.maxFeatures !== undefined) {
      const idxScores = Array.from(importances, (v, i) => ({ i, v })).sort((a, b) => b.v - a.v);
      const keep = new Set(idxScores.slice(0, this.maxFeatures).map((s) => s.i));
      this.mask_ = this.mask_.map((m, i) => m && keep.has(i));
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const selectedIdx = this.mask_.map((m, i) => m ? i : -1).filter((i) => i >= 0);
    return X.map((x) => new Float64Array(selectedIdx.map((i) => x[i] ?? 0)));
  }

  getSupport(): boolean[] { return [...this.mask_]; }

  private _getImportances(): Float64Array | null {
    if (this.estimator.featureImportances_) return this.estimator.featureImportances_;
    if (this.estimator.coef_) {
      const c = this.estimator.coef_;
      if (c instanceof Float64Array) return new Float64Array(c.map(Math.abs));
      // 2D coef: take max over rows
      const nF = c[0]?.length ?? 0;
      const result = new Float64Array(nF);
      for (const row of c) for (let i = 0; i < row.length; i++) result[i] = Math.max(result[i] ?? 0, Math.abs(row[i] ?? 0));
      return result;
    }
    return null;
  }
}

export class VarianceThreshold {
  private variances_: Float64Array = new Float64Array(0);
  private mask_: boolean[] = [];

  constructor(private readonly threshold = 0.0) {}

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.variances_ = new Float64Array(nFeatures);
    const means = new Float64Array(nFeatures);
    const n = X.length;
    for (const x of X) for (let j = 0; j < nFeatures; j++) means[j] = (means[j] ?? 0) + (x[j] ?? 0) / n;
    for (const x of X) for (let j = 0; j < nFeatures; j++) {
      const d = (x[j] ?? 0) - (means[j] ?? 0);
      this.variances_[j] = (this.variances_[j] ?? 0) + d * d / n;
    }
    this.mask_ = Array.from(this.variances_, (v) => v > this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const selectedIdx = this.mask_.map((m, i) => m ? i : -1).filter((i) => i >= 0);
    return X.map((x) => new Float64Array(selectedIdx.map((i) => x[i] ?? 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  getSupport(): boolean[] { return [...this.mask_]; }
}

export class SelectPercentile {
  private scores_: Float64Array = new Float64Array(0);
  private mask_: boolean[] = [];

  constructor(
    private readonly scoreFn: (X: Float64Array[], y: Int32Array | Float64Array) => Float64Array,
    private readonly percentile = 10
  ) {}

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    this.scores_ = this.scoreFn(X, y);
    const nFeatures = this.scores_.length;
    const threshold = this.percentile / 100;
    const sorted = [...this.scores_].sort((a, b) => b - a);
    const cutoff = sorted[Math.floor(threshold * nFeatures)] ?? 0;
    this.mask_ = Array.from(this.scores_, (v) => v >= cutoff);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const selectedIdx = this.mask_.map((m, i) => m ? i : -1).filter((i) => i >= 0);
    return X.map((x) => new Float64Array(selectedIdx.map((i) => x[i] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Int32Array | Float64Array): Float64Array[] { return this.fit(X, y).transform(X); }
  getSupport(): boolean[] { return [...this.mask_]; }
}
