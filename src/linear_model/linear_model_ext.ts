/**
 * Extended linear model: BayesianRidgeExt, ARDRegressionExt, HuberRegressorExt
 */

export class BayesianRidgeExt {
  private maxIter: number;
  private tol: number;
  private alpha1: number;
  private alpha2: number;
  private lambda1: number;
  private lambda2: number;
  coef_: Float64Array | null = null;
  alpha_: number = 1;
  lambda_: number = 1;
  interceptFitted_: number = 0;
  fitIntercept: boolean;

  constructor(
    maxIter = 300,
    tol = 1e-3,
    alpha1 = 1e-6,
    alpha2 = 1e-6,
    lambda1 = 1e-6,
    lambda2 = 1e-6,
    fitIntercept = true
  ) {
    this.maxIter = maxIter;
    this.tol = tol;
    this.alpha1 = alpha1;
    this.alpha2 = alpha2;
    this.lambda1 = lambda1;
    this.lambda2 = lambda2;
    this.fitIntercept = fitIntercept;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    let alpha = this.alpha_;
    let lambda = this.lambda_;

    // Mean-center if intercept
    const xMean = new Float64Array(p);
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    if (this.fitIntercept) {
      for (const row of X) for (let j = 0; j < p; j++) xMean[j] += (row[j] ?? 0) / n;
    }

    const Xc = X.map((row) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = (row[j] ?? 0) - (this.fitIntercept ? (xMean[j] ?? 0) : 0);
      return r;
    });
    const yc = y.map((v) => v - (this.fitIntercept ? yMean : 0));

    // Precompute XtX and Xty
    const XtX: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    const Xty = new Float64Array(p);
    for (let s = 0; s < n; s++) {
      for (let i = 0; i < p; i++) {
        Xty[i] += (Xc[s]![i] ?? 0) * (yc[s] ?? 0);
        for (let j = 0; j < p; j++) XtX[i]![j] += (Xc[s]![i] ?? 0) * (Xc[s]![j] ?? 0);
      }
    }

    let coef = new Float64Array(p);
    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute posterior covariance: (alpha * XtX + lambda * I)^-1
      const A: Float64Array[] = XtX.map((row, i) => {
        const r = Float64Array.from(row);
        r[i] = (r[i] ?? 0) * alpha + lambda;
        return r;
      });
      // Solve A * coef = alpha * Xty using simple iteration
      const newCoef = this.solveSystem(A, Xty.map((v) => v * alpha));
      const delta = newCoef.reduce((acc, v, i) => acc + (v - (coef[i] ?? 0)) ** 2, 0);

      // Update alpha and lambda
      const gamma = p - lambda * newCoef.reduce((acc, v, i) => {
        let diag = XtX[i]![i] ?? 0;
        return acc + (diag * alpha) / ((diag * alpha) + lambda);
      }, 0);

      const residNorm2 = yc.reduce((acc, yi, i) => {
        let pred = 0;
        for (let j = 0; j < p; j++) pred += (Xc[i]![j] ?? 0) * (newCoef[j] ?? 0);
        return acc + (yi - pred) ** 2;
      }, 0);

      alpha = (n - gamma + 2 * (this.alpha1 - 1)) / (residNorm2 + 2 * this.alpha2);
      const coefNorm2 = newCoef.reduce((acc, v) => acc + v * v, 0);
      lambda = (gamma + 2 * (this.lambda1 - 1)) / (coefNorm2 + 2 * this.lambda2);

      coef = newCoef;
      if (delta < this.tol) break;
    }

    this.coef_ = coef;
    this.alpha_ = alpha;
    this.lambda_ = lambda;
    if (this.fitIntercept) {
      this.interceptFitted_ = yMean;
      for (let j = 0; j < p; j++) this.interceptFitted_ -= (coef[j] ?? 0) * (xMean[j] ?? 0);
    }
    return this;
  }

  private solveSystem(A: Float64Array[], b: Float64Array): Float64Array {
    const n = A.length;
    const aug = A.map((row, i) => {
      const r = new Float64Array(n + 1);
      for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
      r[n] = b[i] ?? 0;
      return r;
    });
    for (let col = 0; col < n; col++) {
      const pivot = aug[col]![col] ?? 0;
      if (Math.abs(pivot) < 1e-12) continue;
      for (let row = col + 1; row < n; row++) {
        const factor = (aug[row]![col] ?? 0) / pivot;
        for (let j = col; j <= n; j++) aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
      }
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = aug[i]![n] ?? 0;
      for (let j = i + 1; j < n; j++) sum -= (aug[i]![j] ?? 0) * (x[j] ?? 0);
      x[i] = Math.abs(aug[i]![i] ?? 0) < 1e-12 ? 0 : sum / (aug[i]![i] ?? 1);
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const coef = this.coef_;
    return new Float64Array(X.map((row) => {
      let pred = this.interceptFitted_;
      for (let j = 0; j < coef.length; j++) pred += (row[j] ?? 0) * (coef[j] ?? 0);
      return pred;
    }));
  }
}

export class ARDRegressionExt {
  private maxIter: number;
  private tol: number;
  private thresholdLambda: number;
  coef_: Float64Array | null = null;
  lambda_: Float64Array | null = null;
  alpha_: number = 1;
  activeFeatures_: Int32Array | null = null;

  constructor(maxIter = 300, tol = 1e-3, thresholdLambda = 1e4) {
    this.maxIter = maxIter;
    this.tol = tol;
    this.thresholdLambda = thresholdLambda;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    let alpha = 1.0;
    const lambda = new Float64Array(p).fill(1.0);
    let coef = new Float64Array(p);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Posterior
      const A: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
      for (let i = 0; i < p; i++) A[i]![i] = lambda[i] ?? 1;
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
          A[i]![j] = (A[i]![j] ?? 0) + alpha * (X[s]![i] ?? 0) * (X[s]![j] ?? 0);
        }
      }
      const Xty = new Float64Array(p);
      for (let s = 0; s < n; s++) for (let j = 0; j < p; j++) Xty[j] += alpha * (X[s]![j] ?? 0) * (y[s] ?? 0);

      const newCoef = this.solve(A, Xty);
      const delta = newCoef.reduce((acc, v, i) => acc + (v - (coef[i] ?? 0)) ** 2, 0);

      // Update lambda (per-feature precision)
      const postVar = this.diagonalInverse(A);
      for (let j = 0; j < p; j++) {
        const gamma_j = 1 - (lambda[j] ?? 1) * (postVar[j] ?? 0);
        lambda[j] = gamma_j / ((newCoef[j] ?? 0) ** 2 + 1e-10);
      }

      // Update alpha
      let ssRes = 0;
      for (let s = 0; s < n; s++) {
        let pred = 0;
        for (let j = 0; j < p; j++) pred += (X[s]![j] ?? 0) * (newCoef[j] ?? 0);
        ssRes += ((y[s] ?? 0) - pred) ** 2;
      }
      alpha = n / (ssRes + 1e-10);
      coef = newCoef;
      if (delta < this.tol) break;
    }

    // Prune features with large lambda
    const active: number[] = [];
    for (let j = 0; j < p; j++) { if ((lambda[j] ?? 0) < this.thresholdLambda) active.push(j); }
    this.activeFeatures_ = new Int32Array(active);
    this.coef_ = coef;
    this.lambda_ = lambda;
    this.alpha_ = alpha;
    return this;
  }

  private solve(A: Float64Array[], b: Float64Array): Float64Array {
    const n = A.length;
    const aug = A.map((row, i) => { const r = new Float64Array(n + 1); for (let j = 0; j < n; j++) r[j] = row[j] ?? 0; r[n] = b[i] ?? 0; return r; });
    for (let col = 0; col < n; col++) {
      const pivot = aug[col]![col] ?? 1;
      for (let row = col + 1; row < n; row++) { const f = (aug[row]![col] ?? 0) / pivot; for (let j = col; j <= n; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0); }
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = aug[i]![n] ?? 0; for (let j = i + 1; j < n; j++) s -= (aug[i]![j] ?? 0) * (x[j] ?? 0); x[i] = Math.abs(aug[i]![i] ?? 0) < 1e-12 ? 0 : s / (aug[i]![i] ?? 1); }
    return x;
  }

  private diagonalInverse(A: Float64Array[]): Float64Array {
    const n = A.length;
    const diag = new Float64Array(n);
    const aug = A.map((row, i) => { const r = new Float64Array(2 * n); for (let j = 0; j < n; j++) r[j] = row[j] ?? 0; r[n + i] = 1; return r; });
    for (let col = 0; col < n; col++) {
      const pivot = aug[col]![col] ?? 1;
      for (let j = 0; j < 2 * n; j++) aug[col]![j] = (aug[col]![j] ?? 0) / (pivot || 1);
      for (let row = 0; row < n; row++) { if (row === col) continue; const f = aug[row]![col] ?? 0; for (let j = 0; j < 2 * n; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0); }
    }
    for (let i = 0; i < n; i++) diag[i] = aug[i]![n + i] ?? 0;
    return diag;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const coef = this.coef_;
    return new Float64Array(X.map((row) => { let s = 0; for (let j = 0; j < coef.length; j++) s += (row[j] ?? 0) * (coef[j] ?? 0); return s; }));
  }
}

export class HuberRegressorExt {
  private epsilon: number;
  private maxIter: number;
  private alpha: number;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  scale_: number = 1;
  outliers_: Int32Array | null = null;

  constructor(epsilon = 1.35, maxIter = 100, alpha = 1e-4) {
    this.epsilon = epsilon;
    this.maxIter = maxIter;
    this.alpha = alpha;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    let coef = new Float64Array(p);
    let intercept = 0;
    let scale = 1;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const residuals = y.map((yi, i) => {
        let pred = intercept;
        for (let j = 0; j < p; j++) pred += (X[i]![j] ?? 0) * (coef[j] ?? 0);
        return yi - pred;
      });

      // Update scale using MAD
      const absRes = residuals.map(Math.abs).sort((a, b) => a - b);
      scale = (absRes[Math.floor(n / 2)] ?? 1) / 0.6745;
      if (scale < 1e-10) scale = 1e-10;

      // Huber weights
      const weights = new Float64Array(n);
      const outliers = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const r = Math.abs(residuals[i] ?? 0) / scale;
        if (r <= this.epsilon) { weights[i] = 1; }
        else { weights[i] = this.epsilon / r; outliers[i] = 1; }
      }

      // Weighted least squares update (gradient descent step)
      const lr = 0.01;
      const newCoef = new Float64Array(p);
      let newIntercept = 0;
      for (let s = 0; s < n; s++) {
        const w = weights[s] ?? 0;
        const r = residuals[s] ?? 0;
        for (let j = 0; j < p; j++) newCoef[j] += lr * 2 * w * r * (X[s]![j] ?? 0) / n;
        newIntercept += lr * 2 * w * r / n;
      }
      for (let j = 0; j < p; j++) coef[j] += newCoef[j] - this.alpha * (coef[j] ?? 0);
      intercept += newIntercept;

      this.outliers_ = outliers;
    }

    this.coef_ = coef;
    this.intercept_ = intercept;
    this.scale_ = scale;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const coef = this.coef_;
    const intercept = this.intercept_;
    return new Float64Array(X.map((row) => { let s = intercept; for (let j = 0; j < coef.length; j++) s += (row[j] ?? 0) * (coef[j] ?? 0); return s; }));
  }
}
