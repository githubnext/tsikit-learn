/**
 * QuantileRegressor and TheilSenRegressor — sklearn linear_model ports.
 */

export class QuantileRegressor {
  quantile: number;
  alpha: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(quantile = 0.5, alpha = 1.0, fitIntercept = true, maxIter = 1000, tol = 1e-4) {
    this.quantile = quantile;
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const q = this.quantile;
    const cols = this.fitIntercept ? p + 1 : p;

    // Augment X with intercept column
    const Xaug = X.map((row) => {
      const r = new Float64Array(cols);
      for (let j = 0; j < p; j++) r[j] = row[j] ?? 0;
      if (this.fitIntercept) r[p] = 1;
      return r;
    });

    // Iteratively reweighted least squares for quantile regression
    let w = new Float64Array(cols).fill(0);
    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute residuals
      const resid = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let yhat = 0;
        for (let j = 0; j < cols; j++) yhat += (Xaug[i]?.[j] ?? 0) * (w[j] ?? 0);
        resid[i] = (y[i] ?? 0) - yhat;
      }

      // Compute pinball weights
      const weights = resid.map((r) => (r >= 0 ? q : q - 1));

      // Weighted normal equations with L1 regularization via coordinate descent step
      const XtWX: Float64Array[] = Array.from({ length: cols }, () => new Float64Array(cols));
      const XtWy = new Float64Array(cols);
      for (let i = 0; i < n; i++) {
        const wi = Math.abs(weights[i] ?? 0) + 1e-8;
        for (let j = 0; j < cols; j++) {
          const xij = Xaug[i]?.[j] ?? 0;
          XtWy[j]! += wi * xij * (y[i] ?? 0);
          for (let k = 0; k < cols; k++) {
            (XtWX[j]! as Float64Array)[k]! += wi * xij * (Xaug[i]?.[k] ?? 0);
          }
        }
      }

      // Add L1 regularization diagonal
      for (let j = 0; j < (this.fitIntercept ? p : cols); j++) {
        (XtWX[j]! as Float64Array)[j]! += this.alpha;
      }

      // Solve via Cholesky (simple Jacobi iteration here)
      const wNew = solveNormalEquations(XtWX, XtWy, cols);
      let maxDiff = 0;
      for (let j = 0; j < cols; j++) maxDiff = Math.max(maxDiff, Math.abs((wNew[j] ?? 0) - (w[j] ?? 0)));
      w = wNew as Float64Array<ArrayBuffer>;
      this.nIter_ = iter + 1;
      if (maxDiff < this.tol) break;
    }

    if (this.fitIntercept) {
      this.coef_ = w.slice(0, p);
      this.intercept_ = w[p] ?? 0;
    } else {
      this.coef_ = w;
      this.intercept_ = 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const coef = this.coef_ ?? new Float64Array(0);
    return new Float64Array(X.map((row) => {
      let v = this.intercept_;
      for (let j = 0; j < coef.length; j++) v += (row[j] ?? 0) * (coef[j] ?? 0);
      return v;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;
  }
}

function solveNormalEquations(A: Float64Array[], b: Float64Array, n: number): Float64Array {
  // Gaussian elimination
  const M: Float64Array[] = A.map((row) => new Float64Array(row));
  const rhs = new Float64Array(b);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs((M[row] as Float64Array)[col] ?? 0) > Math.abs((M[maxRow] as Float64Array)[col] ?? 0)) maxRow = row;
    }
    const tmp = M[col]; M[col] = M[maxRow] as Float64Array; M[maxRow] = tmp as Float64Array;
    const tmpR = rhs[col] ?? 0; rhs[col] = rhs[maxRow] ?? 0; rhs[maxRow] = tmpR;
    const pivot = (M[col] as Float64Array)[col] ?? 1e-12;
    for (let row = col + 1; row < n; row++) {
      const factor = ((M[row] as Float64Array)[col] ?? 0) / pivot;
      for (let k = col; k < n; k++) (M[row] as Float64Array)[k] = ((M[row] as Float64Array)[k] ?? 0) - factor * ((M[col] as Float64Array)[k] ?? 0);
      rhs[row] = (rhs[row] ?? 0) - factor * (rhs[col] ?? 0);
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = rhs[i] ?? 0;
    for (let j = i + 1; j < n; j++) s -= ((M[i] as Float64Array)[j] ?? 0) * (x[j] ?? 0);
    x[i] = s / ((M[i] as Float64Array)[i] ?? 1e-12);
  }
  return x;
}

export class TheilSenRegressor {
  fitIntercept: boolean;
  maxSubpopulation: number;
  nSubsamples: number | null;
  maxIter: number;
  tol: number;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  breakdown_: number = 0;
  nSubpopulation_: number = 0;

  constructor(fitIntercept = true, maxSubpopulation = 1e4, nSubsamples: number | null = null, maxIter = 300, tol = 1e-3) {
    this.fitIntercept = fitIntercept;
    this.maxSubpopulation = maxSubpopulation;
    this.nSubsamples = nSubsamples;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const nSub = Math.min(n, this.nSubsamples ?? n);
    this.nSubpopulation_ = Math.min(Math.max(nSub, p + 1), n);
    this.breakdown_ = 1 / (1 + Math.max(p + 1, 2));

    // Sample pairs and compute slopes (simplified: use median of pairwise slopes for each feature)
    const slopes: Float64Array[] = Array.from({ length: p }, () => new Float64Array(nSub));
    for (let s = 0; s < nSub; s++) {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      while (j === i) j = Math.floor(Math.random() * n);
      for (let f = 0; f < p; f++) {
        const dx = (X[i]?.[f] ?? 0) - (X[j]?.[f] ?? 0);
        const dy = (y[i] ?? 0) - (y[j] ?? 0);
        slopes[f]![s] = dx === 0 ? 0 : dy / dx;
      }
    }

    const coef = new Float64Array(p);
    for (let f = 0; f < p; f++) {
      const sorted = Array.from(slopes[f] ?? []).sort((a, b) => a - b);
      coef[f] = sorted[Math.floor(sorted.length / 2)] ?? 0;
    }
    this.coef_ = coef;

    if (this.fitIntercept) {
      const residuals = y.map((yi, i) => {
        let v = 0;
        for (let f = 0; f < p; f++) v += (X[i]?.[f] ?? 0) * (coef[f] ?? 0);
        return yi - v;
      });
      const sorted = Array.from(residuals).sort((a, b) => a - b);
      this.intercept_ = sorted[Math.floor(sorted.length / 2)] ?? 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const coef = this.coef_ ?? new Float64Array(0);
    return new Float64Array(X.map((row) => {
      let v = this.intercept_;
      for (let j = 0; j < coef.length; j++) v += (row[j] ?? 0) * (coef[j] ?? 0);
      return v;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    let ss_res = 0, ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;
  }
}
