/**
 * BorutaSelector and PermutationImportanceSelector — feature selection via importance.
 */

export interface EstimatorWithImportances {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  featureImportances_: Float64Array | null;
}

export class BorutaSelector {
  estimator: EstimatorWithImportances;
  maxIter: number;
  alpha: number;
  twoStepSelection: boolean;
  private confirmed_: Uint8Array | null = null;
  private tentative_: Uint8Array | null = null;
  private rejected_: Uint8Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(
    estimator: EstimatorWithImportances,
    maxIter = 100,
    alpha = 0.05,
    twoStepSelection = false,
  ) {
    this.estimator = estimator;
    this.maxIter = maxIter;
    this.alpha = alpha;
    this.twoStepSelection = twoStepSelection;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const hitCounts = new Float64Array(p);
    const totalTrials = Math.min(this.maxIter, 50);

    for (let iter = 0; iter < totalTrials; iter++) {
      // Create shadow features (randomly permuted copies)
      const XAug = X.map((row) => {
        const aug = new Float64Array(2 * p);
        aug.set(row);
        return aug;
      });
      // Fill shadow features with shuffled values
      for (let j = 0; j < p; j++) {
        const col = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0);
        for (let i = n - 1; i > 0; i--) {
          const swapIdx = Math.floor(Math.random() * (i + 1));
          const tmp = col[i]; col[i] = col[swapIdx] as number; col[swapIdx] = tmp as number;
        }
        for (let i = 0; i < n; i++) (XAug[i] as Float64Array)[p + j] = col[i] ?? 0;
      }

      this.estimator.fit(XAug, y);
      const importances = this.estimator.featureImportances_ ?? new Float64Array(2 * p);
      const shadowMax = Math.max(...Array.from(importances.slice(p)));

      for (let j = 0; j < p; j++) {
        if ((importances[j] ?? 0) > shadowMax) hitCounts[j]!++;
      }
    }

    // Binomial test approximation: expected successes = totalTrials * 0.5
    const threshold = totalTrials * 0.5 + 1.96 * Math.sqrt(totalTrials * 0.25);
    this.confirmed_ = hitCounts.map((h) => h > threshold ? 1 : 0);
    this.tentative_ = hitCounts.map((h, j) => h > totalTrials * 0.3 && (this.confirmed_?.[j] ?? 0) === 0 ? 1 : 0);
    this.rejected_ = hitCounts.map((h, j) => (this.confirmed_?.[j] ?? 0) === 0 && (this.tentative_?.[j] ?? 0) === 0 ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    const support = this.confirmed_ ?? new Uint8Array(this.nFeaturesIn_);
    if (indices) return Int32Array.from(Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0));
    return support;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support() as Uint8Array;
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }
}

export class PermutationImportanceSelector {
  estimator: { fit: (X: Float64Array[], y: Float64Array | Int32Array) => unknown; score: (X: Float64Array[], y: Float64Array | Int32Array) => number };
  nRepeats: number;
  randomState: number;
  threshold: number;
  private support_: Uint8Array | null = null;
  importances_: Float64Array | null = null;
  importancesMean_: Float64Array | null = null;
  importancesStd_: Float64Array | null = null;

  constructor(
    estimator: { fit: (X: Float64Array[], y: Float64Array | Int32Array) => unknown; score: (X: Float64Array[], y: Float64Array | Int32Array) => number },
    nRepeats = 5,
    randomState = 42,
    threshold = 0.0,
  ) {
    this.estimator = estimator;
    this.nRepeats = nRepeats;
    this.randomState = randomState;
    this.threshold = threshold;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const p = X[0]?.length ?? 0;
    const baseScore = this.estimator.score(X, y);
    const allImportances: Float64Array[] = Array.from({ length: p }, () => new Float64Array(this.nRepeats));

    for (let j = 0; j < p; j++) {
      for (let r = 0; r < this.nRepeats; r++) {
        const Xperm = X.map((row) => new Float64Array(row));
        const colVals = X.map((row) => row[j] ?? 0);
        for (let i = colVals.length - 1; i > 0; i--) {
          const swapIdx = Math.floor(Math.random() * (i + 1));
          const tmp = colVals[i]; colVals[i] = colVals[swapIdx] as number; colVals[swapIdx] = tmp as number;
        }
        for (let i = 0; i < X.length; i++) (Xperm[i] as Float64Array)[j] = colVals[i] ?? 0;
        const permScore = this.estimator.score(Xperm, y);
        (allImportances[j] as Float64Array)[r] = baseScore - permScore;
      }
    }

    this.importancesMean_ = Float64Array.from({ length: p }, (_, j) => {
      const imp = allImportances[j] as Float64Array;
      return imp.reduce((a, b) => a + b, 0) / this.nRepeats;
    });
    this.importancesStd_ = Float64Array.from({ length: p }, (_, j) => {
      const imp = allImportances[j] as Float64Array;
      const mean = imp.reduce((a, b) => a + b, 0) / this.nRepeats;
      return Math.sqrt(imp.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(this.nRepeats - 1, 1));
    });
    this.importances_ = this.importancesMean_;
    this.support_ = this.importancesMean_.map((v) => v > this.threshold ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    const support = this.support_ ?? new Uint8Array(0);
    if (indices) return Int32Array.from(Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0));
    return support;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support() as Uint8Array;
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }
}
