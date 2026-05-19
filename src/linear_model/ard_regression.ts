/**
 * ARD Regression (Automatic Relevance Determination).
 * Mirrors sklearn.linear_model.ARDRegression.
 */

export interface ARDRegressionOptions {
  maxIter?: number;
  tol?: number;
  alphaInit?: number;
  lambdaInit?: number;
  computeScore?: boolean;
  thresholdLambda?: number;
  fitIntercept?: boolean;
  copyX?: boolean;
  verbose?: boolean;
}

/**
 * Bayesian ARD regression.
 * Uses automatic relevance determination to perform feature selection.
 */
export class ARDRegression {
  maxIter: number;
  tol: number;
  alphaInit: number;
  lambdaInit: number;
  computeScore: boolean;
  thresholdLambda: number;
  fitIntercept: boolean;
  verbose: boolean;

  coef_: Float64Array | null = null;
  alpha_: number = 1.0;
  lambda_: Float64Array | null = null;
  sigma_: Float64Array[] | null = null;
  scores_: number[] | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(options: ARDRegressionOptions = {}) {
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-3;
    this.alphaInit = options.alphaInit ?? 1e-6;
    this.lambdaInit = options.lambdaInit ?? 1e-6;
    this.computeScore = options.computeScore ?? false;
    this.thresholdLambda = options.thresholdLambda ?? 1e4;
    this.fitIntercept = options.fitIntercept ?? true;
    this.verbose = options.verbose ?? false;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;

    let xMean = new Float64Array(nFeatures);
    let yMean = 0;

    if (this.fitIntercept) {
      for (const row of X) {
        for (let j = 0; j < nFeatures; j++) xMean[j] = (xMean[j] ?? 0) + (row[j] ?? 0);
      }
      for (let j = 0; j < nFeatures; j++) xMean[j] = (xMean[j] ?? 0) / nSamples;
      for (const v of y) yMean += v;
      yMean /= nSamples;
    }

    const Xc = X.map(row => new Float64Array(row).map((v, j) => v - (xMean[j] ?? 0)));
    const yc = new Float64Array(y.map(v => v - yMean));

    // Initialize hyperparameters
    let alpha = this.alphaInit;
    const lambda = new Float64Array(nFeatures).fill(this.lambdaInit);

    // Compute X^T X (Gram matrix)
    const XtX: Float64Array[] = Array.from({ length: nFeatures }, () => new Float64Array(nFeatures));
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        for (let k = 0; k < nFeatures; k++) {
          XtX[j]![k] = (XtX[j]![k] ?? 0) + (Xc[i]?.[j] ?? 0) * (Xc[i]?.[k] ?? 0);
        }
      }
    }

    // Compute X^T y
    const Xty = new Float64Array(nFeatures);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) Xty[j] = (Xty[j] ?? 0) + (Xc[i]?.[j] ?? 0) * (yc[i] ?? 0);
    }

    let coef = new Float64Array(nFeatures);
    const scores: number[] = [];

    for (let iter = 0; iter < this.maxIter; iter++) {
      const prevCoef = new Float64Array(coef);

      // Update sigma (posterior covariance)
      // Sigma^{-1} = alpha * X^T X + diag(lambda)
      const sigmaInv: Float64Array[] = XtX.map((row, j) => {
        const r = new Float64Array(row);
        r[j] = (r[j] ?? 0) + (lambda[j] ?? 0) / alpha;
        return r;
      });

      // Solve for coef = alpha * sigma * X^T y
      const sigmaXty = this._solveSystem(sigmaInv, Xty);
      coef = sigmaXty;

      // Update alpha (noise precision)
      const residuals = new Float64Array(nSamples).map((_, i) => {
        let pred = 0;
        for (let j = 0; j < nFeatures; j++) pred += (Xc[i]?.[j] ?? 0) * (coef[j] ?? 0);
        return (yc[i] ?? 0) - pred;
      });
      const ssResid = residuals.reduce((s, v) => s + v ** 2, 0);
      alpha = nSamples / (ssResid + 1e-10);

      // Update lambda (weight precisions)
      for (let j = 0; j < nFeatures; j++) {
        lambda[j] = 1 / ((coef[j] ?? 0) ** 2 + 1e-10);
      }

      // Check convergence
      let maxChange = 0;
      for (let j = 0; j < nFeatures; j++) {
        maxChange = Math.max(maxChange, Math.abs((coef[j] ?? 0) - (prevCoef[j] ?? 0)));
      }

      this.nIter_ = iter + 1;
      if (this.computeScore) scores.push(-ssResid * alpha / 2);
      if (maxChange < this.tol) break;
    }

    // Prune irrelevant features (high lambda = low weight)
    const maxLambda = Math.max(...Array.from(lambda));
    for (let j = 0; j < nFeatures; j++) {
      if ((lambda[j] ?? 0) > this.thresholdLambda * maxLambda) coef[j] = 0;
    }

    this.coef_ = coef;
    this.alpha_ = alpha;
    this.lambda_ = lambda;
    if (this.computeScore) this.scores_ = scores;

    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < nFeatures; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
      this.intercept_ = intercept;
    }

    return this;
  }

  private _solveSystem(A: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    const mat = A.map((row, i) => { const r = new Float64Array(n + 1); r.set(row); r[n] = b[i] ?? 0; return r; });

    for (let col = 0; col < n; col++) {
      let maxVal = Math.abs(mat[col]?.[col] ?? 0);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(mat[row]?.[col] ?? 0) > maxVal) { maxVal = Math.abs(mat[row]?.[col] ?? 0); maxRow = row; }
      }
      if (maxRow !== col) { const tmp = mat[col]!; mat[col] = mat[maxRow]!; mat[maxRow] = tmp; }

      const pivot = mat[col]?.[col] ?? 1e-10;
      for (let row = col + 1; row < n; row++) {
        const f = (mat[row]?.[col] ?? 0) / (pivot || 1e-10);
        for (let j = col; j <= n; j++) mat[row]![j] = (mat[row]![j] ?? 0) - f * (mat[col]![j] ?? 0);
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = mat[i]?.[n] ?? 0;
      for (let j = i + 1; j < n; j++) x[i] -= (mat[i]?.[j] ?? 0) * (x[j] ?? 0);
      x[i] /= mat[i]?.[i] ?? 1e-10;
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("ARDRegression not fitted");
    return new Float64Array(X.map(row => {
      let pred = this.intercept_;
      for (let j = 0; j < this.coef_!.length; j++) pred += (row[j] ?? 0) * (this.coef_![j] ?? 0);
      return pred;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((s, v) => s + v, 0) / y.length;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ssTot < 1e-10 ? 1 : 1 - ssRes / ssTot;
  }
}
