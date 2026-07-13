/**
 * Generalized Linear Models: QuantileRegressor, TweedieRegressor, PoissonRegressor, GammaRegressor.
 * Mirrors sklearn.linear_model.QuantileRegressor, TweedieRegressor, etc.
 */

import { NotFittedError } from "../exceptions.js";

/** Soft-threshold for quantile regression subgradient. */
function quantileLoss(r: number, q: number): number {
  return r >= 0 ? q * r : (q - 1) * r;
}

/**
 * Linear regression via quantile loss (pinball loss) minimization.
 * Mirrors sklearn.linear_model.QuantileRegressor.
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

  constructor(
    options: {
      quantile?: number;
      alpha?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    this.quantile = options.quantile ?? 0.5;
    this.alpha = options.alpha ?? 1.0;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const q = this.quantile;

    // Subgradient descent for quantile regression
    const w = new Float64Array(p);
    let intercept = 0;
    const lr0 = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const lr = lr0 / (1 + 0.01 * iter);
      const gw = new Float64Array(p);
      let gi = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let pred = intercept;
        for (let j = 0; j < p; j++) pred += (w[j] ?? 0) * (xi[j] ?? 0);
        const r = (y[i] ?? 0) - pred;
        const sign = r >= 0 ? -q : 1 - q;
        for (let j = 0; j < p; j++) {
          gw[j] = (gw[j] ?? 0) + sign * (xi[j] ?? 0);
        }
        gi += sign;
      }

      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const grad = (gw[j] ?? 0) / n + this.alpha * (w[j] ?? 0);
        const delta = lr * grad;
        w[j] = (w[j] ?? 0) - delta;
        if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
      }
      if (this.fitIntercept) {
        const delta = lr * (gi / n);
        intercept -= delta;
        if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
      }

      this.nIter_ = iter + 1;
      if (maxDelta < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = this.fitIntercept ? intercept : 0;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("QuantileRegressor");
    const w = this.coef_;
    return new Float64Array(
      X.map((xi) => {
        let pred = this.intercept_;
        for (let j = 0; j < xi.length; j++) pred += (w[j] ?? 0) * (xi[j] ?? 0);
        return pred;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    let loss = 0;
    for (let i = 0; i < y.length; i++) {
      loss += quantileLoss((y[i] ?? 0) - (yPred[i] ?? 0), this.quantile);
    }
    return -loss / y.length;
  }
}

/** Link functions for GLMs */
function logLink(mu: number): number {
  return Math.log(Math.max(mu, 1e-8));
}
function expLink(eta: number): number {
  return Math.exp(eta);
}
function identityLink(mu: number): number {
  return mu;
}
function identityInvLink(eta: number): number {
  return eta;
}

/**
 * Generalized Linear Model with Tweedie distribution.
 * Covers Poisson (power=1), Gamma (power=2), and Tweedie family.
 * Mirrors sklearn.linear_model.TweedieRegressor.
 */
export class TweedieRegressor {
  power: number;
  alpha: number;
  link: "auto" | "identity" | "log";
  fitIntercept: boolean;
  maxIter: number;
  tol: number;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(
    options: {
      power?: number;
      alpha?: number;
      link?: "auto" | "identity" | "log";
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    this.power = options.power ?? 0;
    this.alpha = options.alpha ?? 1.0;
    this.link = options.link ?? "auto";
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
  }

  private _useLog(): boolean {
    if (this.link === "log") return true;
    if (this.link === "identity") return false;
    // auto: use log for power != 0
    return this.power !== 0;
  }

  private _mu(eta: number): number {
    return this._useLog() ? expLink(eta) : identityInvLink(eta);
  }

  private _eta(mu: number): number {
    return this._useLog() ? logLink(mu) : identityLink(mu);
  }

  /** Variance function V(mu) for Tweedie: mu^power */
  private _variance(mu: number): number {
    if (this.power === 0) return 1;
    return Math.max(mu, 1e-8) ** this.power;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    const w = new Float64Array(p);
    // Initialize intercept to log(mean(y)) or mean(y)
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
    let intercept = this._eta(Math.max(yMean, 1e-8));

    // IRLS (Iteratively Reweighted Least Squares)
    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute working weights and adjusted response
      const weights = new Float64Array(n);
      const z = new Float64Array(n);

      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let eta = intercept;
        for (let j = 0; j < p; j++) eta += (w[j] ?? 0) * (xi[j] ?? 0);
        const mu = this._mu(eta);
        const V = this._variance(mu);
        const dmu = this._useLog() ? mu : 1;
        weights[i] = (dmu * dmu) / Math.max(V, 1e-10);
        z[i] = eta + ((y[i] ?? 0) - mu) / Math.max(dmu, 1e-10);
      }

      // Weighted least squares update (gradient step)
      const gw = new Float64Array(p);
      let gi = 0;
      let wSum = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let eta = intercept;
        for (let j = 0; j < p; j++) eta += (w[j] ?? 0) * (xi[j] ?? 0);
        const r = (z[i] ?? 0) - eta;
        const wi = weights[i] ?? 0;
        wSum += wi;
        for (let j = 0; j < p; j++) {
          gw[j] = (gw[j] ?? 0) + wi * r * (xi[j] ?? 0);
        }
        gi += wi * r;
      }

      let maxDelta = 0;
      const lr = 0.1;
      for (let j = 0; j < p; j++) {
        const grad = (gw[j] ?? 0) / n - this.alpha * (w[j] ?? 0);
        const delta = lr * grad;
        w[j] = (w[j] ?? 0) + delta;
        if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
      }
      if (this.fitIntercept) {
        const delta = lr * (gi / n);
        intercept += delta;
        if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
      }

      this.nIter_ = iter + 1;
      if (maxDelta < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = this.fitIntercept ? intercept : 0;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("TweedieRegressor");
    const w = this.coef_;
    return new Float64Array(
      X.map((xi) => {
        let eta = this.intercept_;
        for (let j = 0; j < xi.length; j++) eta += (w[j] ?? 0) * (xi[j] ?? 0);
        return this._mu(eta);
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot > 0 ? 1 - ss_res / ss_tot : 0;
  }
}

/**
 * GLM with Poisson distribution (log link). Alias for TweedieRegressor(power=1).
 * Mirrors sklearn.linear_model.PoissonRegressor.
 */
export class PoissonRegressor extends TweedieRegressor {
  constructor(
    options: {
      alpha?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    super({ ...options, power: 1, link: "log" });
  }
}

/**
 * GLM with Gamma distribution (log link). Alias for TweedieRegressor(power=2).
 * Mirrors sklearn.linear_model.GammaRegressor.
 */
export class GammaRegressor extends TweedieRegressor {
  constructor(
    options: {
      alpha?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    super({ ...options, power: 2, link: "log" });
  }
}
