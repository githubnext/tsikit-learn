/**
 * GammaRegressor, PoissonRegressor, TweedieRegressor (GLMs) — sklearn linear_model ports.
 */

type LinkFunction = "log" | "identity" | "logit";

export class GeneralizedLinearRegressor {
  power: number;
  alpha: number;
  fitIntercept: boolean;
  link: LinkFunction;
  maxIter: number;
  tol: number;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(
    power = 0,
    alpha = 1.0,
    fitIntercept = true,
    link: LinkFunction = "auto" as LinkFunction,
    maxIter = 100,
    tol = 1e-4,
  ) {
    this.power = power;
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    this.link = link === ("auto" as LinkFunction) ? (power === 0 ? "identity" : "log") : link;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  private linkFn(eta: number): number {
    if (this.link === "log") return Math.exp(Math.min(eta, 700));
    if (this.link === "logit") return 1 / (1 + Math.exp(-eta));
    return eta;
  }

  private linkDeriv(mu: number): number {
    if (this.link === "log") return mu;
    if (this.link === "logit") return mu * (1 - mu);
    return 1;
  }

  private varianceFn(mu: number): number {
    if (this.power === 0) return 1;
    if (this.power === 1) return mu;
    return mu ** this.power;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const cols = this.fitIntercept ? p + 1 : p;

    const Xa: Float64Array[] = X.map((row) => {
      const r = new Float64Array(cols);
      for (let j = 0; j < p; j++) r[j] = row[j] ?? 0;
      if (this.fitIntercept) r[p] = 1;
      return r;
    });

    let beta = new Float64Array(cols);
    // Initialize intercept from mean
    if (this.fitIntercept) {
      const mu0 = y.reduce((a, b) => a + b, 0) / n;
      if (this.link === "log") beta[p] = Math.log(Math.max(mu0, 1e-6));
      else if (this.link === "logit") beta[p] = Math.log(Math.max(mu0, 1e-6) / Math.max(1 - mu0, 1e-6));
      else beta[p] = mu0;
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute mu and working weights
      const eta = Xa.map((row) => {
        let s = 0;
        for (let j = 0; j < cols; j++) s += (row[j] ?? 0) * (beta[j] ?? 0);
        return s;
      });
      const mu = eta.map((e) => this.linkFn(e));
      const W = mu.map((m, i) => {
        const d = this.linkDeriv(m);
        const v = this.varianceFn(m);
        return d * d / Math.max(v, 1e-8);
      });
      const z = eta.map((e, i) => e + ((y[i] ?? 0) - (mu[i] ?? 0)) / Math.max(this.linkDeriv(mu[i] ?? 0), 1e-8));

      // IRLS: solve weighted normal equations
      const XtWX: Float64Array[] = Array.from({ length: cols }, () => new Float64Array(cols));
      const XtWz = new Float64Array(cols);
      for (let i = 0; i < n; i++) {
        const wi = W[i] ?? 0;
        for (let j = 0; j < cols; j++) {
          const xij = Xa[i]?.[j] ?? 0;
          XtWz[j] += wi * xij * (z[i] ?? 0);
          for (let k = 0; k < cols; k++) {
            (XtWX[j] as Float64Array)[k] += wi * xij * (Xa[i]?.[k] ?? 0);
          }
        }
      }
      for (let j = 0; j < (this.fitIntercept ? p : cols); j++) {
        (XtWX[j] as Float64Array)[j] += this.alpha;
      }

      const betaNew = solveGE(XtWX, XtWz, cols);
      let maxDiff = 0;
      for (let j = 0; j < cols; j++) maxDiff = Math.max(maxDiff, Math.abs((betaNew[j] ?? 0) - (beta[j] ?? 0)));
      beta = betaNew;
      this.nIter_ = iter + 1;
      if (maxDiff < this.tol) break;
    }

    if (this.fitIntercept) {
      this.coef_ = beta.slice(0, p);
      this.intercept_ = beta[p] ?? 0;
    } else {
      this.coef_ = beta;
      this.intercept_ = 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const coef = this.coef_ ?? new Float64Array(0);
    return new Float64Array(X.map((row) => {
      let eta = this.intercept_;
      for (let j = 0; j < coef.length; j++) eta += (row[j] ?? 0) * (coef[j] ?? 0);
      return this.linkFn(eta);
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    // D^2 score (deviance-based R^2)
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    let devFull = 0;
    let devNull = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = y[i] ?? 0;
      const pi = Math.max(yPred[i] ?? 1e-8, 1e-8);
      const mu0 = Math.max(yMean, 1e-8);
      if (this.link === "log") {
        devFull += 2 * (yi * Math.log(Math.max(yi, 1e-8) / pi) - (yi - pi));
        devNull += 2 * (yi * Math.log(Math.max(yi, 1e-8) / mu0) - (yi - mu0));
      } else {
        devFull += (yi - pi) ** 2;
        devNull += (yi - yMean) ** 2;
      }
    }
    return devNull === 0 ? 0 : 1 - devFull / devNull;
  }
}

function solveGE(A: Float64Array[], b: Float64Array, n: number): Float64Array {
  const M: Float64Array[] = A.map((row) => new Float64Array(row));
  const rhs = new Float64Array(b);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs((M[row] as Float64Array)[col] ?? 0) > Math.abs((M[maxRow] as Float64Array)[col] ?? 0)) maxRow = row;
    }
    const tmp = M[col]; M[col] = M[maxRow] as Float64Array; M[maxRow] = tmp as Float64Array;
    const t = rhs[col] ?? 0; rhs[col] = rhs[maxRow] ?? 0; rhs[maxRow] = t;
    const pivot = (M[col] as Float64Array)[col] ?? 1e-12;
    for (let row = col + 1; row < n; row++) {
      const f = ((M[row] as Float64Array)[col] ?? 0) / pivot;
      for (let k = col; k < n; k++) (M[row] as Float64Array)[k] = ((M[row] as Float64Array)[k] ?? 0) - f * ((M[col] as Float64Array)[k] ?? 0);
      rhs[row] = (rhs[row] ?? 0) - f * (rhs[col] ?? 0);
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

export class PoissonRegressor extends GeneralizedLinearRegressor {
  constructor(alpha = 1.0, fitIntercept = true, maxIter = 100, tol = 1e-4) {
    super(1, alpha, fitIntercept, "log", maxIter, tol);
  }
}

export class GammaRegressor extends GeneralizedLinearRegressor {
  constructor(alpha = 1.0, fitIntercept = true, maxIter = 100, tol = 1e-4) {
    super(2, alpha, fitIntercept, "log", maxIter, tol);
  }
}

export class TweedieRegressorExt extends GeneralizedLinearRegressor {
  constructor(power = 1.5, alpha = 1.0, fitIntercept = true, maxIter = 100, tol = 1e-4) {
    super(power, alpha, fitIntercept, "log", maxIter, tol);
  }
}
