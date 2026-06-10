/**
 * Regularized LDA, Heteroscedastic LDA, and shrinkage estimators for discriminant analysis.
 */

export class RegularizedLDA {
  private classMeans_!: Map<number, Float64Array>;
  private classCovariances_!: Map<number, Float64Array[]>;
  private classPriors_!: Map<number, number>;
  private classes_!: Int32Array;
  private fitted_ = false;

  constructor(
    private regularization = 0.1,
    private solver: 'lsqr' | 'eigen' = 'lsqr'
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Int32Array(classSet);
    this.classMeans_ = new Map();
    this.classCovariances_ = new Map();
    this.classPriors_ = new Map();

    for (const c of classSet) {
      const indices = Array.from(y).map((v, i) => v === c ? i : -1).filter(i => i >= 0);
      const prior = indices.length / n;
      const mean = new Float64Array(p).map((_, j) =>
        indices.reduce((s, i) => s + (X[i]![j] ?? 0), 0) / indices.length
      );
      // Class-specific covariance with regularization
      const cov = Array.from({ length: p }, (_, j) =>
        new Float64Array(p).map((_, k) => {
          const c1 = indices.reduce((s, i) => s + ((X[i]![j] ?? 0) - (mean[j] ?? 0)) * ((X[i]![k] ?? 0) - (mean[k] ?? 0)), 0) / (indices.length - 1 || 1);
          return c1 + (j === k ? this.regularization : 0);
        })
      );
      this.classMeans_.set(c, mean);
      this.classCovariances_.set(c, cov);
      this.classPriors_.set(c, prior);
    }
    this.fitted_ = true;
    void this.solver;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let bestClass = this.classes_[0]!, bestScore = -Number.POSITIVE_INFINITY;
      for (const c of this.classes_) {
        const mean = this.classMeans_.get(c)!;
        const cov = this.classCovariances_.get(c)!;
        const prior = this.classPriors_.get(c) ?? 0;
        // Mahalanobis distance + log prior
        const diff = new Float64Array(x.map((v, j) => v - (mean[j] ?? 0)));
        const covInv = this._invertCov(cov);
        const mahal = diff.reduce((s, v, j) => s + v * covInv[j]!.reduce((cs, cv, k) => cs + cv * (diff[k] ?? 0), 0), 0);
        const logDetCov = this._logDet(cov);
        const score = -0.5 * mahal - 0.5 * logDetCov + Math.log(prior + 1e-10);
        if (score > bestScore) { bestScore = score; bestClass = c; }
      }
      return bestClass;
    }));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(x => {
      const scores = new Float64Array(this.classes_.length).map((_, ci) => {
        const c = this.classes_[ci]!;
        const mean = this.classMeans_.get(c)!;
        const cov = this.classCovariances_.get(c)!;
        const prior = this.classPriors_.get(c) ?? 0;
        const diff = new Float64Array(x.map((v, j) => v - (mean[j] ?? 0)));
        const covInv = this._invertCov(cov);
        const mahal = diff.reduce((s, v, j) => s + v * covInv[j]!.reduce((cs, cv, k) => cs + cv * (diff[k] ?? 0), 0), 0);
        const logDetCov = this._logDet(cov);
        return -0.5 * mahal - 0.5 * logDetCov + Math.log(prior + 1e-10);
      });
      const maxScore = Math.max(...scores);
      const exps = new Float64Array(scores.map(s => Math.exp(s - maxScore)));
      const sumExps = exps.reduce((s, v) => s + v, 0);
      return new Float64Array(exps.map(v => v / (sumExps + 1e-10)));
    });
  }

  private _invertCov(cov: Float64Array[]): Float64Array[] {
    const p = cov.length;
    const aug = cov.map((row, i) => {
      const r = Array.from(row) as number[];
      for (let j = 0; j < p; j++) r.push(i === j ? 1 : 0);
      return r;
    });
    for (let col = 0; col < p; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j < 2 * p; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (piv + 1e-10);
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j < 2 * p; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => aug[i]![p + j] ?? 0));
  }

  private _logDet(cov: Float64Array[]): number {
    const p = cov.length;
    let logDet = 0;
    const L = cov.map(row => new Float64Array(row));
    for (let col = 0; col < p; col++) {
      const v = L[col]![col] ?? 0;
      if (v <= 0) return 0;
      logDet += Math.log(v);
      const sqrtV = Math.sqrt(v);
      for (let row = col; row < p; row++) L[row]![col] = (L[row]![col] ?? 0) / sqrtV;
      for (let row = col + 1; row < p; row++) {
        const f = L[row]![col] ?? 0;
        for (let k = col; k < p; k++) L[row]![k] = (L[row]![k] ?? 0) - f * (L[col]![k] ?? 0);
      }
    }
    return 2 * logDet;
  }

  get classes(): Int32Array { return this.classes_; }
}

export function shrinkageLDA(
  X: Float64Array[],
  y: Int32Array,
  shrinkageAlpha = 0.1
): RegularizedLDA {
  const lda = new RegularizedLDA(shrinkageAlpha);
  lda.fit(X, y);
  return lda;
}
