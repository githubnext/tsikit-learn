/**
 * Robust covariance estimators: MinCovDet, EllipticEnvelope.
 */

export class MinCovDet {
  private location_!: Float64Array;
  private covariance_!: Float64Array[];
  private precision_!: Float64Array[];
  private supportFraction: number;
  private fitted_ = false;

  constructor(supportFraction = 0.7, private maxIter = 1000, private tol = 1e-8) {
    this.supportFraction = supportFraction;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const h = Math.max(p + 1, Math.floor(n * this.supportFraction));
    
    // Random initialization
    let bestDet = Number.POSITIVE_INFINITY;
    let bestSupport: number[] = [];

    for (let trial = 0; trial < 10; trial++) {
      const perm = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
      let support = perm.slice(0, Math.min(p + 1, n));

      for (let iter = 0; iter < this.maxIter; iter++) {
        const Xs = support.map(i => X[i]!);
        const loc = this._mean(Xs, p);
        const cov = this._cov(Xs, loc, p);
        // Compute Mahalanobis distances
        const prec = this._invertMat(cov, p);
        const dists = X.map(xi => this._mahal2(xi, loc, prec));
        const sorted = dists.map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d);
        const newSupport = sorted.slice(0, h).map(x => x.i);
        if (newSupport.every((v, i) => v === support[i])) { support = newSupport; break; }
        support = newSupport;
      }
      const Xs = support.map(i => X[i]!);
      const loc = this._mean(Xs, p);
      const cov = this._cov(Xs, loc, p);
      const det = this._det(cov, p);
      if (det < bestDet) { bestDet = det; bestSupport = support; }
    }

    const Xs = bestSupport.map(i => X[i]!);
    this.location_ = this._mean(Xs, p);
    this.covariance_ = this._cov(Xs, this.location_, p);
    // Correction factor
    const cf = this._correctionFactor(n, h, p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) this.covariance_[i]![j] = (this.covariance_[i]![j] ?? 0) * cf;
    this.precision_ = this._invertMat(this.covariance_, p);
    this.fitted_ = true;
    return this;
  }

  private _mean(X: Float64Array[], p: number): Float64Array {
    const n = X.length;
    return new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[j] ?? 0), 0) / (n || 1));
  }

  private _cov(X: Float64Array[], mean: Float64Array, p: number): Float64Array[] {
    const n = X.length;
    return Array.from({ length: p }, (_, i) =>
      new Float64Array(p).map((_, j) => X.reduce((s, row) => s + ((row[i] ?? 0) - (mean[i] ?? 0)) * ((row[j] ?? 0) - (mean[j] ?? 0)), 0) / (n - 1 || 1))
    );
  }

  private _mahal2(x: Float64Array, mean: Float64Array, prec: Float64Array[]): number {
    const d = x.map((v, i) => v - (mean[i] ?? 0));
    return d.reduce((s, v, i) => s + v * prec[i]!.reduce((ss, pij, j) => ss + pij * (d[j] ?? 0), 0), 0);
  }

  private _det(A: Float64Array[], p: number): number {
    const M = A.map(row => new Float64Array(row));
    let det = 1;
    for (let k = 0; k < p; k++) {
      let pivot = M[k]![k] ?? 0;
      if (Math.abs(pivot) < 1e-12) return 0;
      det *= pivot;
      for (let i = k + 1; i < p; i++) {
        const f = (M[i]![k] ?? 0) / pivot;
        for (let j = k; j < p; j++) M[i]![j] = (M[i]![j] ?? 0) - f * (M[k]![j] ?? 0);
      }
    }
    return det;
  }

  private _invertMat(A: Float64Array[], p: number): Float64Array[] {
    const aug = A.map((row, i) => [...Array.from(row), ...(Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)))]);
    for (let col = 0; col < p; col++) {
      const piv = aug[col]![col] ?? 1;
      for (let j = col; j < 2 * p; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (piv || 1);
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const f = aug[row]![col] ?? 0;
        for (let j = col; j < 2 * p; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
      }
    }
    return Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => aug[i]![p + j] ?? 0));
  }

  private _correctionFactor(n: number, h: number, p: number): number {
    const q = h / n;
    const phi = Math.exp(-0.5 * p / q);
    return 1 / (2 * q * phi / Math.sqrt(2 * Math.PI) / (2 * phi - 1));
  }

  mahalanobisDistances(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(x => Math.sqrt(Math.max(this._mahal2(x, this.location_, this.precision_), 0))));
  }

  get location(): Float64Array { return this.location_; }
  get covariance(): Float64Array[] { return this.covariance_; }
  get precision(): Float64Array[] { return this.precision_; }
}

export class EllipticEnvelope {
  private mcd_: MinCovDet;
  private threshold_!: number;
  private fitted_ = false;

  constructor(private contamination = 0.1, supportFraction = 0.7) {
    this.mcd_ = new MinCovDet(supportFraction);
  }

  fit(X: Float64Array[]): this {
    this.mcd_.fit(X);
    const dists = this.mcd_.mahalanobisDistances(X);
    const sorted = Array.from(dists).sort((a, b) => a - b);
    const cutoff = Math.floor((1 - this.contamination) * X.length);
    this.threshold_ = sorted[cutoff] ?? sorted[sorted.length - 1] ?? Number.POSITIVE_INFINITY;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const dists = this.mcd_.mahalanobisDistances(X);
    return new Int32Array(dists.map(d => d <= this.threshold_ ? 1 : -1));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    return pred.filter((v, i) => v === y[i]).length / pred.length;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const dists = this.mcd_.mahalanobisDistances(X);
    return new Float64Array(dists.map(d => this.threshold_ - d));
  }
}
