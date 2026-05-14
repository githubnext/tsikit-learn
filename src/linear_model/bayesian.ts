/**
 * BayesianRidge and ARDRegression.
 * Mirrors sklearn.linear_model.BayesianRidge and ARDRegression.
 */

import { NotFittedError } from "../exceptions.js";

// ─── BayesianRidge ────────────────────────────────────────────────────────────

export interface BayesianRidgeOptions {
  maxIter?: number;
  tol?: number;
  alpha1?: number;
  alpha2?: number;
  lambda1?: number;
  lambda2?: number;
  fitIntercept?: boolean;
  computeScore?: boolean;
}

export class BayesianRidge {
  maxIter: number;
  tol: number;
  alpha1: number;
  alpha2: number;
  lambda1: number;
  lambda2: number;
  fitIntercept: boolean;
  computeScore: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  alpha_: number = 1;
  lambda_: number = 1;
  sigma_: Float64Array[] | null = null;

  constructor(opts: BayesianRidgeOptions = {}) {
    this.maxIter = opts.maxIter ?? 300;
    this.tol = opts.tol ?? 1e-3;
    this.alpha1 = opts.alpha1 ?? 1e-6;
    this.alpha2 = opts.alpha2 ?? 1e-6;
    this.lambda1 = opts.lambda1 ?? 1e-6;
    this.lambda2 = opts.lambda2 ?? 1e-6;
    this.fitIntercept = opts.fitIntercept ?? true;
    this.computeScore = opts.computeScore ?? false;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;

    // Center if fitting intercept
    let Xfit = X;
    let yfit = y;
    let xMean = new Float64Array(d);
    let yMean = 0;

    if (this.fitIntercept) {
      for (const xi of X) for (let j = 0; j < d; j++) xMean[j]! += (xi[j] ?? 0) / n;
      for (let i = 0; i < n; i++) yMean += (y[i] ?? 0) / n;
      Xfit = X.map((xi) => Float64Array.from({ length: d }, (_, j) => (xi[j] ?? 0) - (xMean[j] ?? 0)));
      yfit = Float64Array.from(y, (v) => v - yMean);
    }

    // Gram matrix X^T X
    const XtX: Float64Array[] = Array.from({ length: d }, () => new Float64Array(d));
    for (const xi of Xfit) {
      for (let i = 0; i < d; i++) {
        for (let j = i; j < d; j++) {
          XtX[i]![j]! += (xi[i] ?? 0) * (xi[j] ?? 0);
          if (i !== j) XtX[j]![i]! += (xi[i] ?? 0) * (xi[j] ?? 0);
        }
      }
    }

    let alpha = this.alpha_;
    let lambda = this.lambda_;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sigma = (lambda * X^T X + alpha * I)^{-1}
      const A: Float64Array[] = XtX.map((row, i) =>
        Float64Array.from(row, (v, j) => lambda * v + (i === j ? alpha : 0)),
      );

      // Solve for coef using Gaussian elimination
      const coef = this.solveLinear(A, this.xtYDot(Xfit, yfit, d, lambda));

      const alphaOld = alpha;
      const lambdaOld = lambda;

      // gamma = sum(lambda_i / (alpha + lambda_i)) via trace
      // Approximate: gamma = d - alpha * trace(Sigma)
      const residuals = Float64Array.from({ length: n }, (_, i) => {
        let pred = 0;
        for (let j = 0; j < d; j++) pred += (coef[j] ?? 0) * ((Xfit[i] as Float64Array)[j] ?? 0);
        return (yfit[i] ?? 0) - pred;
      });

      const ssRes = residuals.reduce((s, v) => s + v * v, 0);
      const ssCoef = coef.reduce((s, v) => s + v * v, 0);

      alpha = (this.alpha1 + n / 2) / (this.alpha2 + ssRes / 2);
      lambda = (this.lambda1 + d / 2) / (this.lambda2 + ssCoef / 2);

      if (Math.abs(alpha - alphaOld) < this.tol && Math.abs(lambda - lambdaOld) < this.tol) {
        this.alpha_ = alpha;
        this.lambda_ = lambda;
        this.coef_ = coef;
        break;
      }
      this.alpha_ = alpha;
      this.lambda_ = lambda;
      this.coef_ = coef;
    }

    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < d; j++) intercept -= (this.coef_![j] ?? 0) * (xMean[j] ?? 0);
      this.intercept_ = intercept;
    }

    return this;
  }

  private xtYDot(X: Float64Array[], y: Float64Array, d: number, lambda: number): Float64Array {
    const xty = new Float64Array(d);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i] as Float64Array;
      for (let j = 0; j < d; j++) xty[j]! += lambda * (xi[j] ?? 0) * (y[i] ?? 0);
    }
    return xty;
  }

  private solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
    const n = A.length;
    const aug: Float64Array[] = A.map((row, i) => {
      const r = Float64Array.from(row);
      return Float64Array.from([...r, b[i] ?? 0]);
    });

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      let maxVal = Math.abs((aug[col] as Float64Array)[col] ?? 0);
      for (let row = col + 1; row < n; row++) {
        const v = Math.abs((aug[row] as Float64Array)[col] ?? 0);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      const tmp = aug[col]!;
      aug[col]! = aug[maxRow]!;
      aug[maxRow]! = tmp;

      const pivot = (aug[col] as Float64Array)[col] ?? 1;
      if (Math.abs(pivot) < 1e-12) continue;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = ((aug[row] as Float64Array)[col] ?? 0) / pivot;
        for (let j = col; j <= n; j++) {
          (aug[row] as Float64Array)[j]! -= factor * ((aug[col] as Float64Array)[j] ?? 0);
        }
      }
    }

    return Float64Array.from({ length: n }, (_, i) => {
      const row = aug[i] as Float64Array;
      const diag = row[i] ?? 1;
      return Math.abs(diag) < 1e-12 ? 0 : (row[n] ?? 0) / diag;
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("BayesianRidge");
    return Float64Array.from(X, (xi) => {
      let pred = this.intercept_;
      for (let j = 0; j < xi.length; j++) pred += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
      return pred;
    });
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}

// ─── ARDRegression ────────────────────────────────────────────────────────────

export interface ARDRegressionOptions {
  maxIter?: number;
  tol?: number;
  alpha1?: number;
  alpha2?: number;
  lambda1?: number;
  lambda2?: number;
  computeScore?: boolean;
  fitIntercept?: boolean;
  thresholdLambda?: number;
}

export class ARDRegression {
  maxIter: number;
  tol: number;
  alpha1: number;
  alpha2: number;
  lambda1: number;
  lambda2: number;
  fitIntercept: boolean;
  thresholdLambda: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  alpha_: number = 1;
  lambda_: Float64Array | null = null;

  constructor(opts: ARDRegressionOptions = {}) {
    this.maxIter = opts.maxIter ?? 300;
    this.tol = opts.tol ?? 1e-3;
    this.alpha1 = opts.alpha1 ?? 1e-6;
    this.alpha2 = opts.alpha2 ?? 1e-6;
    this.lambda1 = opts.lambda1 ?? 1e-6;
    this.lambda2 = opts.lambda2 ?? 1e-6;
    this.fitIntercept = opts.fitIntercept ?? true;
    this.thresholdLambda = opts.thresholdLambda ?? 1e4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;

    let Xfit = X;
    let yfit = y;
    const xMean = new Float64Array(d);
    let yMean = 0;

    if (this.fitIntercept) {
      for (const xi of X) for (let j = 0; j < d; j++) xMean[j]! += (xi[j] ?? 0) / n;
      for (let i = 0; i < n; i++) yMean += (y[i] ?? 0) / n;
      Xfit = X.map((xi) => Float64Array.from({ length: d }, (_, j) => (xi[j] ?? 0) - (xMean[j] ?? 0)));
      yfit = Float64Array.from(y, (v) => v - yMean);
    }

    let alpha = this.alpha_;
    const lambda = new Float64Array(d).fill(1);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Active features (lambda < threshold)
      const active = Array.from({ length: d }, (_, j) => (lambda[j] ?? 0) < this.thresholdLambda);

      const coef = new Float64Array(d);
      // Solve for active features only (simplified: use diagonal approximation)
      const xty = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const xi = Xfit[i] as Float64Array;
        for (let j = 0; j < d; j++) {
          if (active[j]) xty[j]! += (xi[j] ?? 0) * (yfit[i] ?? 0);
        }
      }

      // Diagonal approximation of (X^T X + diag(alpha/lambda))^{-1} X^T y
      const xtxDiag = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const xi = Xfit[i] as Float64Array;
        for (let j = 0; j < d; j++) xtxDiag[j]! += (xi[j] ?? 0) ** 2;
      }
      for (let j = 0; j < d; j++) {
        if (active[j]) {
          const denom = (xtxDiag[j] ?? 0) + alpha / (lambda[j] ?? 1);
          coef[j]! = denom > 0 ? (xty[j] ?? 0) / denom : 0;
        }
      }

      const alphaOld = alpha;
      const lambdaOld = Float64Array.from(lambda);

      const residuals = Float64Array.from({ length: n }, (_, i) => {
        let pred = 0;
        const xi = Xfit[i] as Float64Array;
        for (let j = 0; j < d; j++) pred += (coef[j] ?? 0) * (xi[j] ?? 0);
        return (yfit[i] ?? 0) - pred;
      });

      const ssRes = residuals.reduce((s, v) => s + v * v, 0);
      alpha = (this.alpha1 + n / 2) / (this.alpha2 + ssRes / 2);

      for (let j = 0; j < d; j++) {
        lambda[j]! = (this.lambda1 + 0.5) / (this.lambda2 + (coef[j] ?? 0) ** 2 / 2);
      }

      let converged = Math.abs(alpha - alphaOld) < this.tol;
      for (let j = 0; j < d; j++) {
        if (Math.abs((lambda[j] ?? 0) - (lambdaOld[j] ?? 0)) > this.tol) { converged = false; break; }
      }

      this.coef_ = coef;
      this.alpha_ = alpha;
      this.lambda_ = lambda;
      if (converged) break;
    }

    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < d; j++) intercept -= (this.coef_![j] ?? 0) * (xMean[j] ?? 0);
      this.intercept_ = intercept;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("ARDRegression");
    return Float64Array.from(X, (xi) => {
      let pred = this.intercept_;
      for (let j = 0; j < xi.length; j++) pred += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
      return pred;
    });
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
