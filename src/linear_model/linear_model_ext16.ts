/**
 * RANSAC Regressor and Orthogonal Matching Pursuit.
 */

export class RANSACRegressor {
  private coef_!: Float64Array;
  private intercept_ = 0;
  private inlierMask_!: boolean[];
  private fitted_ = false;

  constructor(
    private minSamples = 0.1,
    private residualThreshold = 1.0,
    private maxTrials = 100,
    private stopProbability = 0.99
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const minN = typeof this.minSamples === 'number' && this.minSamples < 1
      ? Math.max(p + 1, Math.floor(n * this.minSamples))
      : Math.max(p + 1, Math.floor(this.minSamples as number));
    let bestScore = -1;
    let bestCoef = new Float64Array(p);
    let bestIntercept = 0;
    let bestMask = new Array<boolean>(n).fill(false);

    for (let trial = 0; trial < this.maxTrials; trial++) {
      // Random sample
      const idx: number[] = [];
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [perm[i], perm[j]] = [perm[j]!, perm[i]!];
      }
      idx.push(...perm.slice(0, minN));

      const Xs = idx.map(i => X[i]!);
      const ys = new Float64Array(idx.map(i => y[i] ?? 0));
      const [coef, intercept] = this._fitOLS(Xs, ys);
      const resids = X.map((row, i) => Math.abs((y[i] ?? 0) - intercept - row.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0)));
      const inliers = resids.map(r => r <= this.residualThreshold);
      const score = inliers.filter(Boolean).length;

      if (score > bestScore) {
        bestScore = score;
        // Refit on inliers
        const inlierX = X.filter((_, i) => inliers[i]);
        const inlierY = new Float64Array(Array.from(y).filter((_, i) => inliers[i]));
        if (inlierX.length >= p + 1) {
          const [c, b] = this._fitOLS(inlierX, inlierY);
          bestCoef = c; bestIntercept = b;
        } else { bestCoef = coef; bestIntercept = intercept; }
        bestMask = inliers;
      }
      // Early stopping
      if (score / n >= this.stopProbability) break;
    }
    this.coef_ = bestCoef;
    this.intercept_ = bestIntercept;
    this.inlierMask_ = bestMask;
    this.fitted_ = true;
    return this;
  }

  private _fitOLS(X: Float64Array[], y: Float64Array): [Float64Array, number] {
    const n = X.length, p = (X[0]?.length ?? 0) + 1;
    const Xa = X.map(row => new Float64Array([1, ...row]));
    const XtX = Array.from({ length: p }, (_, i) =>
      new Float64Array(p).map((_, j) => Xa.reduce((s, row) => s + (row[i] ?? 0) * (row[j] ?? 0), 0))
    );
    const Xty = new Float64Array(p).map((_, j) => Xa.reduce((s, row, i) => s + (row[j] ?? 0) * (y[i] ?? 0), 0));
    for (let i = 0; i < p; i++) XtX[i]![i] = (XtX[i]![i] ?? 0) + 1e-8;
    const augM = XtX.map((row, i) => [...row, Xty[i] ?? 0]);
    for (let col = 0; col < p; col++) {
      const piv = augM[col]![col] ?? 1;
      for (let j = col; j <= p; j++) augM[col]![j] = (augM[col]![j] ?? 0) / piv;
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const f = augM[row]![col] ?? 0;
        for (let j = col; j <= p; j++) augM[row]![j] = (augM[row]![j] ?? 0) - f * (augM[col]![j] ?? 0);
      }
    }
    const w = augM.map(row => row[p] ?? 0);
    return [new Float64Array(w.slice(1)), w[0] ?? 0];
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => this.intercept_ + row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0)));
  }

  get coef(): Float64Array { return this.coef_; }
  get intercept(): number { return this.intercept_; }
  get inlierMask(): boolean[] { return this.inlierMask_; }
}

export class OrthogonalMatchingPursuit {
  private coef_!: Float64Array;
  private fitted_ = false;

  constructor(private nNonzeroCoefs: number | null = null, private tol: number | null = null) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const maxK = this.nNonzeroCoefs ?? Math.min(n, p);
    const active: number[] = [];
    let residual = new Float64Array(y);
    this.coef_ = new Float64Array(p);

    for (let k = 0; k < maxK; k++) {
      // Select atom most correlated with residual
      let bestJ = 0, bestCorr = -1;
      for (let j = 0; j < p; j++) {
        if (active.includes(j)) continue;
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (X[i]![j] ?? 0) * (residual[i] ?? 0);
        corr = Math.abs(corr);
        if (corr > bestCorr) { bestCorr = corr; bestJ = j; }
      }
      active.push(bestJ);
      // OLS on active set
      const Xa = X.map(row => new Float64Array(active.map(j => row[j] ?? 0)));
      const XtX = Array.from({ length: active.length }, (_, i) =>
        new Float64Array(active.length).map((_, jj) => Xa.reduce((s, row) => s + (row[i] ?? 0) * (row[jj] ?? 0), 0))
      );
      const Xty = new Float64Array(active.length).map((_, i) => Xa.reduce((s, row, idx) => s + (row[i] ?? 0) * (y[idx] ?? 0), 0));
      for (let i = 0; i < active.length; i++) XtX[i]![i] = (XtX[i]![i] ?? 0) + 1e-10;
      const augM = XtX.map((row, i) => [...row, Xty[i] ?? 0]);
      for (let col = 0; col < active.length; col++) {
        const piv = augM[col]![col] ?? 1;
        for (let j = col; j <= active.length; j++) augM[col]![j] = (augM[col]![j] ?? 0) / piv;
        for (let row = 0; row < active.length; row++) {
          if (row === col) continue;
          const f = augM[row]![col] ?? 0;
          for (let j = col; j <= active.length; j++) augM[row]![j] = (augM[row]![j] ?? 0) - f * (augM[col]![j] ?? 0);
        }
      }
      const wa = augM.map(row => row[active.length] ?? 0);
      for (let i = 0; i < active.length; i++) this.coef_[active[i]!] = wa[i] ?? 0;
      residual = new Float64Array(X.map((row, i) => (y[i] ?? 0) - active.reduce((s, j, ii) => s + (row[j] ?? 0) * (wa[ii] ?? 0), 0)));
      const res2 = residual.reduce((s, v) => s + v * v, 0);
      if (this.tol !== null && res2 < this.tol) break;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0)));
  }

  get coef(): Float64Array { return this.coef_; }
}
