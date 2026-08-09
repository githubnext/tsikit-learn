/**
 * Quantile Regression and Theil-Sen Estimator.
 */

export class QuantileRegressor {
  private coef_!: Float64Array;
  private intercept_ = 0;
  private fitted_ = false;

  constructor(private quantile = 0.5, private alpha = 1.0, private maxIter = 1000, private tol = 1e-4) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = (X[0]?.length ?? 0) + 1;
    // Augmented X with intercept
    const Xa = X.map(row => new Float64Array([1, ...row]));
    let w = new Float64Array(p);
    // Gradient descent for pinball loss
    let lr = 0.01;
    for (let iter = 0; iter < this.maxIter; iter++) {
      const grad = new Float64Array(p);
      for (let i = 0; i < n; i++) {
        const yHat = Xa[i]!.reduce((s, v, j) => s + v * (w[j] ?? 0), 0);
        const r = (y[i] ?? 0) - yHat;
        const check = r >= 0 ? this.quantile : this.quantile - 1;
        for (let j = 0; j < p; j++) grad[j] = (grad[j] ?? 0) - check * (Xa[i]![j] ?? 0);
      }
      // L1 regularization on non-intercept
      for (let j = 1; j < p; j++) {
        const sign = (w[j] ?? 0) > 0 ? 1 : -1;
        grad[j] = (grad[j] ?? 0) + this.alpha * sign;
      }
      for (let j = 0; j < p; j++) w[j] = (w[j] ?? 0) - lr * (grad[j] ?? 0) / n;
      // Adaptive lr
      if (iter % 100 === 99) lr *= 0.9;
      const gnorm = Math.sqrt(grad.reduce((s, v) => s + v * v, 0));
      if (gnorm < this.tol) break;
    }
    this.intercept_ = w[0] ?? 0;
    this.coef_ = w.slice(1);
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => this.intercept_ + row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0)));
  }

  get coef(): Float64Array { return this.coef_; }
  get intercept(): number { return this.intercept_; }
}

export class TheilSenRegressor {
  private coef_!: Float64Array;
  private intercept_ = 0;
  private fitted_ = false;

  constructor(private maxSubpopulation = 1000) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 1;
    if (p === 1) {
      // Univariate case: median of pairwise slopes
      const slopes: number[] = [];
      const maxPairs = Math.min(this.maxSubpopulation, (n * (n - 1)) / 2);
      for (let i = 0; i < n && slopes.length < maxPairs; i++) {
        for (let j = i + 1; j < n && slopes.length < maxPairs; j++) {
          const dx = (X[j]![0] ?? 0) - (X[i]![0] ?? 0);
          if (Math.abs(dx) > 1e-10) slopes.push(((y[j] ?? 0) - (y[i] ?? 0)) / dx);
        }
      }
      slopes.sort((a, b) => a - b);
      const slope = slopes[Math.floor(slopes.length / 2)] ?? 0;
      this.coef_ = new Float64Array([slope]);
      const residuals = X.map((row, i) => (y[i] ?? 0) - slope * (row[0] ?? 0));
      residuals.sort((a, b) => a - b);
      this.intercept_ = residuals[Math.floor(residuals.length / 2)] ?? 0;
    } else {
      // Multivariate: simplified least-median-squares
      this.coef_ = new Float64Array(p);
      let bestCost = Number.POSITIVE_INFINITY;
      const nTrials = Math.min(this.maxSubpopulation, n);
      for (let trial = 0; trial < nTrials; trial++) {
        const idx = Array.from({ length: p + 1 }, () => Math.floor(Math.random() * n));
        // Fit OLS to small subset
        const Xsub = idx.map(i => X[i]!);
        const ysub = new Float64Array(idx.map(i => y[i] ?? 0));
        const coef = this._ols(Xsub, ysub);
        const cost = Array.from(X).map((row, i) => {
          const yHat = row.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0);
          return Math.abs((y[i] ?? 0) - yHat);
        }).sort((a, b) => a - b)[Math.floor(n / 2)] ?? 0;
        if (cost < bestCost) { bestCost = cost; this.coef_ = coef; }
      }
      const residuals = X.map((row, i) => (y[i] ?? 0) - row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0));
      residuals.sort((a, b) => a - b);
      this.intercept_ = residuals[Math.floor(residuals.length / 2)] ?? 0;
    }
    this.fitted_ = true;
    return this;
  }

  private _ols(X: Float64Array[], y: Float64Array): Float64Array {
    const p = X[0]?.length ?? 1;
    const XtX = Array.from({ length: p }, (_, i) =>
      new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[i] ?? 0) * (row[j] ?? 0), 0))
    );
    const Xty = new Float64Array(p).map((_, j) => X.reduce((s, row, i) => s + (row[j] ?? 0) * (y[i] ?? 0), 0));
    // Add small ridge
    for (let i = 0; i < p; i++) XtX[i]![i] = (XtX[i]![i] ?? 0) + 1e-6;
    // Simple Gauss-Jordan
    const augM = XtX.map((row, i) => [...row, Xty[i] ?? 0]);
    for (let col = 0; col < p; col++) {
      const pivot = augM[col]![col] ?? 1;
      for (let j = col; j <= p; j++) augM[col]![j] = (augM[col]![j] ?? 0) / pivot;
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const f = augM[row]![col] ?? 0;
        for (let j = col; j <= p; j++) augM[row]![j] = (augM[row]![j] ?? 0) - f * (augM[col]![j] ?? 0);
      }
    }
    return new Float64Array(augM.map(row => row[p] ?? 0));
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => this.intercept_ + row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0)));
  }

  get coef(): Float64Array { return this.coef_; }
  get intercept(): number { return this.intercept_; }
}
