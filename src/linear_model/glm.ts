/**
 * Generalized Linear Models base infrastructure.
 * Mirrors sklearn.linear_model._glm.
 *
 * Provides link functions and GeneralizedLinearRegressor base class.
 */

import { BaseEstimator } from "../base.js";

// ── Link functions ─────────────────────────────────────────────────────────

/** Base class for GLM link functions. */
export abstract class BaseLink {
  abstract name: string;

  /** Link function: maps mean μ → linear predictor η */
  abstract link(mu: number): number;

  /** Inverse link: maps linear predictor η → mean μ */
  abstract inverseLink(eta: number): number;

  /** Derivative of the inverse link: d(μ)/d(η) */
  abstract inverseLinkDerivative(eta: number): number;

  /** Compute linear predictors from means */
  linkArray(mu: Float64Array): Float64Array {
    const eta = new Float64Array(mu.length);
    for (let i = 0; i < mu.length; i++) {
      eta[i] = this.link(mu[i]!);
    }
    return eta;
  }

  /** Compute means from linear predictors */
  inverseLinkArray(eta: Float64Array): Float64Array {
    const mu = new Float64Array(eta.length);
    for (let i = 0; i < eta.length; i++) {
      mu[i] = this.inverseLink(eta[i]!);
    }
    return mu;
  }
}

/**
 * Identity link: η = μ, μ = η
 * Used for Gaussian / normal distributions.
 */
export class IdentityLink extends BaseLink {
  name = "identity";
  link(mu: number): number { return mu; }
  inverseLink(eta: number): number { return eta; }
  inverseLinkDerivative(_eta: number): number { return 1; }
}

/**
 * Log link: η = log(μ), μ = exp(η)
 * Used for Poisson and gamma distributions.
 */
export class LogLink extends BaseLink {
  name = "log";
  link(mu: number): number { return Math.log(mu); }
  inverseLink(eta: number): number { return Math.exp(eta); }
  inverseLinkDerivative(eta: number): number { return Math.exp(eta); }
}

/**
 * Logit link: η = log(μ/(1-μ)), μ = 1/(1+exp(-η))
 * Used for Bernoulli/binomial distributions.
 */
export class LogitLink extends BaseLink {
  name = "logit";
  link(mu: number): number {
    const clipped = Math.max(1e-15, Math.min(1 - 1e-15, mu));
    return Math.log(clipped / (1 - clipped));
  }
  inverseLink(eta: number): number {
    return 1 / (1 + Math.exp(-eta));
  }
  inverseLinkDerivative(eta: number): number {
    const p = this.inverseLink(eta);
    return p * (1 - p);
  }
}

/**
 * Square-root link: η = sqrt(μ), μ = η²
 * Used for certain count data.
 */
export class SqrtLink extends BaseLink {
  name = "sqrt";
  link(mu: number): number { return Math.sqrt(mu); }
  inverseLink(eta: number): number { return eta * eta; }
  inverseLinkDerivative(eta: number): number { return 2 * eta; }
}

/**
 * Complementary log-log link: η = log(-log(1 - μ))
 * Used for extreme value models.
 */
export class CLogLogLink extends BaseLink {
  name = "cloglog";
  link(mu: number): number {
    const clipped = Math.max(1e-15, Math.min(1 - 1e-15, mu));
    return Math.log(-Math.log(1 - clipped));
  }
  inverseLink(eta: number): number {
    return 1 - Math.exp(-Math.exp(eta));
  }
  inverseLinkDerivative(eta: number): number {
    return Math.exp(eta - Math.exp(eta));
  }
}

// ── Distributions (variance functions) ────────────────────────────────────

/** Base class for exponential dispersion distributions. */
export abstract class BaseDistribution {
  abstract name: string;

  /** Unit deviance: 2 * (t(y, y) - t(y, mu)) */
  abstract unitDeviance(y: number, mu: number): number;

  /** Variance function: V(μ) */
  abstract variance(mu: number): number;

  /** Log-likelihood contribution for one observation */
  logLikelihood(y: number, mu: number, dispersion = 1): number {
    return -0.5 * this.unitDeviance(y, mu) / dispersion;
  }

  /** Total deviance */
  deviance(y: Float64Array, mu: Float64Array, weights?: Float64Array): number {
    let d = 0;
    for (let i = 0; i < y.length; i++) {
      const w = weights ? (weights[i] ?? 1) : 1;
      d += w * this.unitDeviance(y[i]!, mu[i]!);
    }
    return d;
  }
}

/** Normal / Gaussian distribution */
export class NormalDistribution extends BaseDistribution {
  name = "normal";
  unitDeviance(y: number, mu: number): number {
    return (y - mu) ** 2;
  }
  variance(_mu: number): number { return 1; }
}

/** Poisson distribution */
export class PoissonDistribution extends BaseDistribution {
  name = "poisson";
  unitDeviance(y: number, mu: number): number {
    if (y === 0) return 2 * mu;
    return 2 * (y * Math.log(y / mu) - (y - mu));
  }
  variance(mu: number): number { return mu; }
}

/** Gamma distribution */
export class GammaDistribution extends BaseDistribution {
  name = "gamma";
  unitDeviance(y: number, mu: number): number {
    return 2 * ((y - mu) / mu - Math.log(y / mu));
  }
  variance(mu: number): number { return mu * mu; }
}

/** Tweedie distribution with power parameter p */
export class TweedieDistribution extends BaseDistribution {
  name = "tweedie";
  constructor(public power = 0) { super(); }

  unitDeviance(y: number, mu: number): number {
    const p = this.power;
    if (p === 0) return (y - mu) ** 2;
    if (p === 1) return 2 * (y * Math.log(y / mu) - (y - mu));
    if (p === 2) return 2 * ((y - mu) / mu - Math.log(y / mu));
    const a = Math.max(0, y);
    const b = mu;
    return 2 * (
      (a ** (2 - p)) / ((1 - p) * (2 - p))
      - (a * b ** (1 - p)) / (1 - p)
      + (b ** (2 - p)) / (2 - p)
    );
  }
  variance(mu: number): number { return mu ** this.power; }
}

/** Bernoulli / Binomial distribution */
export class BinomialDistribution extends BaseDistribution {
  name = "binomial";
  unitDeviance(y: number, mu: number): number {
    const c1 = y > 0 ? y * Math.log(y / mu) : 0;
    const c2 = (1 - y) > 0 ? (1 - y) * Math.log((1 - y) / (1 - mu)) : 0;
    return 2 * (c1 + c2);
  }
  variance(mu: number): number { return mu * (1 - mu); }
}

// ── GeneralizedLinearRegressor ─────────────────────────────────────────────

export interface GLMOptions {
  /** Link function instance or name */
  link?: BaseLink | "identity" | "log" | "logit" | "sqrt" | "cloglog";
  /** Distribution instance */
  distribution?: BaseDistribution | "normal" | "poisson" | "gamma" | "binomial";
  /** L2 regularization strength */
  alpha?: number;
  /** Fit intercept */
  fitIntercept?: boolean;
  /** Max iterations for IRLS */
  maxIter?: number;
  /** Convergence tolerance */
  tol?: number;
  /** Verbose output */
  verbose?: boolean;
}

function resolveLink(link: GLMOptions["link"]): BaseLink {
  if (!link || link === "identity") return new IdentityLink();
  if (link === "log") return new LogLink();
  if (link === "logit") return new LogitLink();
  if (link === "sqrt") return new SqrtLink();
  if (link === "cloglog") return new CLogLogLink();
  return link;
}

function resolveDist(
  dist: GLMOptions["distribution"],
): BaseDistribution {
  if (!dist || dist === "normal") return new NormalDistribution();
  if (dist === "poisson") return new PoissonDistribution();
  if (dist === "gamma") return new GammaDistribution();
  if (dist === "binomial") return new BinomialDistribution();
  return dist;
}

/**
 * Generalized Linear Model fitted via Iteratively Reweighted Least Squares (IRLS).
 * Base class for PoissonRegressor, GammaRegressor, TweedieRegressor etc.
 *
 * @see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GeneralizedLinearRegressor.html
 */
export class GeneralizedLinearRegressor extends BaseEstimator {
  link: BaseLink;
  distribution: BaseDistribution;
  alpha: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  verbose: boolean;

  // Fitted attributes
  coef_?: Float64Array;
  intercept_?: number;
  n_iter_?: number;

  constructor(options: GLMOptions = {}) {
    super();
    this.link = resolveLink(options.link);
    this.distribution = resolveDist(options.distribution);
    this.alpha = options.alpha ?? 0;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
    this.verbose = options.verbose ?? false;
  }

  fit(X: Float64Array[], y: Float64Array, sampleWeight?: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const pFull = this.fitIntercept ? p + 1 : p;

    // Build design matrix with optional intercept column
    const Xfull: Float64Array[] = X.map(row => {
      if (!this.fitIntercept) return row;
      const r = new Float64Array(pFull);
      r[0] = 1;
      r.set(row, 1);
      return r;
    });

    // Initialize coefficients
    let beta: Float64Array = new Float64Array(pFull);
    // Initialize mu as mean(y) for all samples
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / n;
    let mu = new Float64Array(n).fill(Math.max(1e-4, yMean));

    let prevDev = Number.POSITIVE_INFINITY;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // IRLS step
      // Working response z_i = eta_i + (y_i - mu_i) / mu_prime_i
      // Weight w_i = w_i_sample * mu_prime_i^2 / V(mu_i)
      const eta = this.link.linkArray(mu);
      const z = new Float64Array(n);
      const W = new Float64Array(n);

      for (let i = 0; i < n; i++) {
        const muPrime = this.link.inverseLinkDerivative(eta[i]!);
        const V = this.distribution.variance(mu[i]!);
        const sw = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
        W[i] = sw * (muPrime * muPrime) / Math.max(1e-12, V);
        z[i] = eta[i]! + (y[i]! - mu[i]!) / Math.max(1e-12, muPrime);
      }

      // Weighted least squares: (X'WX + alpha*I) beta = X'Wz
      const XtW: Float64Array[] = Xfull.map((_, j) => {
        const col = new Float64Array(pFull);
        for (let i = 0; i < n; i++) {
          col[j] = (col[j] ?? 0) + (Xfull[i]![j] ?? 0) * W[i]!;
        }
        return col;
      });

      // Build XtWX (pFull x pFull)
      const XtWX = Array.from({ length: pFull }, () => new Float64Array(pFull));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < pFull; j++) {
          for (let k = 0; k < pFull; k++) {
            XtWX[j]![k]! += (Xfull[i]![j] ?? 0) * W[i]! * (Xfull[i]![k] ?? 0);
          }
        }
      }

      // Add L2 regularization (skip intercept if present)
      const start = this.fitIntercept ? 1 : 0;
      for (let j = start; j < pFull; j++) {
        XtWX[j]![j]! += this.alpha;
      }

      // Build XtWz (pFull)
      const XtWz = new Float64Array(pFull);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < pFull; j++) {
          XtWz[j]! += (Xfull[i]![j] ?? 0) * W[i]! * z[i]!;
        }
      }

      // Solve via Cholesky / Gaussian elimination
      beta = solveLinear(XtWX, XtWz);

      // Update mu
      for (let i = 0; i < n; i++) {
        let etaI = 0;
        for (let j = 0; j < pFull; j++) {
          etaI += (Xfull[i]![j] ?? 0) * (beta[j] ?? 0);
        }
        mu[i] = Math.max(1e-10, this.link.inverseLink(etaI));
      }

      // Check convergence
      const dev = this.distribution.deviance(y, mu, sampleWeight);
      if (Math.abs(prevDev - dev) / (Math.abs(prevDev) + 0.1) < this.tol) {
        this.n_iter_ = iter + 1;
        break;
      }
      prevDev = dev;
      this.n_iter_ = iter + 1;
    }

    if (this.fitIntercept) {
      this.intercept_ = beta[0] ?? 0;
      this.coef_ = beta.slice(1);
    } else {
      this.intercept_ = 0;
      this.coef_ = beta;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    this._check_is_fitted(["coef_"]);
    const n = X.length;
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let eta = this.intercept_ ?? 0;
      for (let j = 0; j < (this.coef_?.length ?? 0); j++) {
        eta += (X[i]![j] ?? 0) * (this.coef_![j] ?? 0);
      }
      result[i] = this.link.inverseLink(eta);
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    this._check_is_fitted(["coef_"]);
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    let ssTot = 0, ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += (y[i]! - yMean) ** 2;
      ssRes += (y[i]! - yPred[i]!) ** 2;
    }
    return 1 - ssRes / (ssTot + 1e-12);
  }
}

/** Simple Gaussian elimination for small dense systems */
function solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
  const n = b.length;
  // Augment
  const M: Float64Array[] = A.map((row, i) => {
    const r = new Float64Array(n + 1);
    r.set(row);
    r[n] = b[i] ?? 0;
    return r;
  });

  for (let col = 0; col < n; col++) {
    // Pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row]![col] ?? 0) > Math.abs(M[maxRow]![col] ?? 0)) maxRow = row;
    }
    const tmp = M[col]!; M[col] = M[maxRow]!; M[maxRow] = tmp;

    const pivot = M[col]![col] ?? 0;
    if (Math.abs(pivot) < 1e-14) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = (M[row]![col] ?? 0) / pivot;
      for (let k = col; k <= n; k++) {
        M[row]![k]! -= factor * (M[col]![k] ?? 0);
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i]![n] ?? 0;
    for (let j = i + 1; j < n; j++) {
      sum -= (M[i]![j] ?? 0) * (x[j] ?? 0);
    }
    x[i] = sum / (M[i]![i] ?? 1e-12);
  }
  return x;
}
