/**
 * Bayesian Linear Regression extensions and Automatic Relevance Determination.
 */

export class BayesianRidgeExt {
  private coef_!: Float64Array;
  private alpha_!: number;
  private lambda_!: number;
  private sigma_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private maxIter = 300,
    private tol = 1e-3,
    private alpha1 = 1e-6,
    private alpha2 = 1e-6,
    private lambda1 = 1e-6,
    private lambda2 = 1e-6
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 1;
    this.alpha_ = 1;
    this.lambda_ = 1;

    // XT X
    const XtX = Array.from({ length: p }, (_, i) =>
      new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[i] ?? 0) * (row[j] ?? 0), 0))
    );
    const Xty = new Float64Array(p).map((_, j) => X.reduce((s, row, i) => s + (row[j] ?? 0) * (y[i] ?? 0), 0));

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute posterior covariance: (lambda * XtX + alpha * I)^{-1}
      const A = XtX.map((row, i) => row.map((v, j) => this.lambda_ * v + (i === j ? this.alpha_ : 0)));
      const invA = this._invertSymmetric(A, p);
      const coef = new Float64Array(p).map((_, j) => invA[j]!.reduce((s, v, k) => s + v * this.lambda_ * (Xty[k] ?? 0), 0));

      // Update hyperparameters
      const gamma = this.lambda_ * invA.reduce((s, row, i) => s + row[i]!, 0);
      const alphaNew = (gamma + 2 * this.alpha1) / (coef.reduce((s, v) => s + v * v, 0) + 2 * this.alpha2);
      const resid = X.reduce((s, row, i) => {
        const yHat = row.reduce((ss, v, j) => ss + v * (coef[j] ?? 0), 0);
        return s + ((y[i] ?? 0) - yHat) ** 2;
      }, 0);
      const lambdaNew = (n - gamma + 2 * this.lambda1) / (resid + 2 * this.lambda2);

      if (Math.abs(alphaNew - this.alpha_) < this.tol && Math.abs(lambdaNew - this.lambda_) < this.tol) {
        this.coef_ = coef;
        this.sigma_ = invA;
        break;
      }
      this.alpha_ = alphaNew;
      this.lambda_ = lambdaNew;
      this.coef_ = coef;
      this.sigma_ = invA;
    }
    this.fitted_ = true;
    return this;
  }

  private _invertSymmetric(A: Float64Array[], p: number): Float64Array[] {
    const M = A.map(row => new Float64Array(row));
    const inv = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = 1;
      return row;
    });
    for (let col = 0; col < p; col++) {
      const pivot = M[col]![col] ?? 1;
      for (let j = col; j < p; j++) M[col]![j] = (M[col]![j] ?? 0) / pivot;
      inv[col] = new Float64Array(inv[col]!.map(v => v / pivot));
      for (let row = 0; row < p; row++) {
        if (row === col) continue;
        const factor = M[row]![col] ?? 0;
        for (let j = col; j < p; j++) M[row]![j] = (M[row]![j] ?? 0) - factor * (M[col]![j] ?? 0);
        inv[row] = new Float64Array(inv[row]!.map((v, j) => v - factor * (inv[col]![j] ?? 0)));
      }
    }
    return inv;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(row => row.reduce((s, v, j) => s + v * (this.coef_[j] ?? 0), 0)));
  }

  predictWithStd(X: Float64Array[]): { mean: Float64Array; std: Float64Array } {
    const mean = this.predict(X);
    const std = new Float64Array(X.map(row => {
      const v = this.sigma_.reduce((s, sigmaRow, i) =>
        s + (row[i] ?? 0) * sigmaRow.reduce((ss, sv, j) => ss + sv * (row[j] ?? 0), 0), 0);
      return Math.sqrt(v + 1 / this.alpha_);
    }));
    return { mean, std };
  }

  get coef(): Float64Array { return this.coef_; }
  get alpha(): number { return this.alpha_; }
  get lambda(): number { return this.lambda_; }
}
