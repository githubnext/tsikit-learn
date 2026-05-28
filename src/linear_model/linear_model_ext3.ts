/**
 * Extended linear models: TweedieRegressor, PoissonRegressor, GammaRegressor
 * Port of sklearn.linear_model._glm
 */

import { NotFittedError } from "../exceptions.js";

export type TweedieLink = "auto" | "identity" | "log";
export type TweediePower = number;

function tweedieDeviance(
  y: Float64Array,
  yPred: Float64Array,
  power: number
): number {
  let dev = 0;
  for (let i = 0; i < y.length; i++) {
    const yi = y[i] ?? 0;
    const mui = yPred[i] ?? 0;
    if (power === 0) {
      dev += (yi - mui) ** 2;
    } else if (power === 1) {
      dev += 2 * (yi * Math.log((yi + 1e-15) / (mui + 1e-15)) - (yi - mui));
    } else if (power === 2) {
      dev += 2 * (Math.log((mui + 1e-15) / (yi + 1e-15)) + (yi - mui) / (mui + 1e-15));
    } else {
      const a = power - 1;
      const b = power - 2;
      dev += 2 * (
        (yi ** (2 - power)) / ((1 - power) * (2 - power)) -
        yi * (mui ** (1 - power)) / (1 - power) +
        (mui ** (2 - power)) / (2 - power)
      );
      void a; void b;
    }
  }
  return dev / y.length;
}

function applyLink(x: Float64Array, link: TweedieLink, inverse = false): Float64Array {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i] ?? 0;
    if (link === "log") {
      result[i] = inverse ? Math.exp(xi) : Math.log(Math.max(xi, 1e-15));
    } else {
      result[i] = xi;
    }
  }
  return result;
}

function irls(
  X: Float64Array[],
  y: Float64Array,
  link: TweedieLink,
  power: number,
  alpha: number,
  maxIter: number,
  tol: number
): { coef: Float64Array; intercept: number } {
  const n = X.length;
  const p = (X[0]?.length ?? 0) + 1;
  let coef = new Float64Array(p);

  for (let iter = 0; iter < maxIter; iter++) {
    const eta = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let val = coef[p - 1] ?? 0;
      const xi = X[i];
      if (xi) {
        for (let j = 0; j < xi.length; j++) val += (xi[j] ?? 0) * (coef[j] ?? 0);
      }
      eta[i] = val;
    }
    const mu = applyLink(eta, link, true);
    const z = new Float64Array(n);
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const mui = mu[i] ?? 0;
      const yi = y[i] ?? 0;
      const etai = eta[i] ?? 0;
      const dmu = link === "log" ? mui : 1.0;
      const variance = power === 0 ? 1 : power === 1 ? mui : power === 2 ? mui ** 2 : Math.pow(mui, power);
      w[i] = dmu ** 2 / (variance + 1e-15);
      z[i] = etai + (yi - mui) / (dmu + 1e-15);
    }
    const XtWX = Array.from({ length: p }, () => new Float64Array(p));
    const XtWz = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const wi = w[i] ?? 0;
      const zi = z[i] ?? 0;
      const xi = X[i];
      const row = new Float64Array(p);
      if (xi) for (let j = 0; j < xi.length; j++) row[j] = xi[j] ?? 0;
      row[p - 1] = 1.0;
      for (let j = 0; j < p; j++) {
        const rowj = row[j] ?? 0;
        XtWz[j] = (XtWz[j] ?? 0) + wi * rowj * zi;
        for (let k = 0; k < p; k++) {
          XtWX[j]![k] = (XtWX[j]![k] ?? 0) + wi * rowj * (row[k] ?? 0);
        }
      }
    }
    for (let j = 0; j < p - 1; j++) XtWX[j]![j] = (XtWX[j]![j] ?? 0) + alpha;
    const newCoef = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let s = XtWz[j] ?? 0;
      for (let k = 0; k < p; k++) {
        if (k !== j) s -= (XtWX[j]![k] ?? 0) * (newCoef[k] ?? coef[k] ?? 0);
      }
      newCoef[j] = s / ((XtWX[j]![j] ?? 1) + 1e-15);
    }
    let diff = 0;
    for (let j = 0; j < p; j++) diff += ((newCoef[j] ?? 0) - (coef[j] ?? 0)) ** 2;
    coef = newCoef;
    if (diff < tol) break;
    void iter;
  }
  const intercept = coef[p - 1] ?? 0;
  return { coef: coef.slice(0, p - 1), intercept };
}

export class TweedieRegressor {
  power: number;
  alpha: number;
  link: TweedieLink;
  maxIter: number;
  tol: number;

  private coef_: Float64Array | null = null;
  private intercept_ = 0;

  constructor(opts: {
    power?: number;
    alpha?: number;
    link?: TweedieLink;
    maxIter?: number;
    tol?: number;
  } = {}) {
    this.power = opts.power ?? 0;
    this.alpha = opts.alpha ?? 1.0;
    this.link = opts.link ?? "auto";
    this.maxIter = opts.maxIter ?? 100;
    this.tol = opts.tol ?? 1e-4;
  }

  private resolvedLink(): TweedieLink {
    if (this.link !== "auto") return this.link;
    return this.power === 0 ? "identity" : "log";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const link = this.resolvedLink();
    const { coef, intercept } = irls(X, y, link, this.power, this.alpha, this.maxIter, this.tol);
    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("TweedieRegressor is not fitted.");
    const link = this.resolvedLink();
    const result = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      let val = this.intercept_;
      if (xi) for (let j = 0; j < xi.length; j++) val += (xi[j] ?? 0) * (this.coef_[j] ?? 0);
      result[i] = link === "log" ? Math.exp(val) : val;
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    return 1 - tweedieDeviance(y, yPred, this.power) / (tweedieDeviance(y, new Float64Array(y.length).fill(y.reduce((a, b) => a + b, 0) / y.length), this.power) + 1e-15);
  }
}

export class PoissonRegressor extends TweedieRegressor {
  constructor(opts: { alpha?: number; maxIter?: number; tol?: number } = {}) {
    super({ ...opts, power: 1, link: "log" });
  }
}

export class GammaRegressor extends TweedieRegressor {
  constructor(opts: { alpha?: number; maxIter?: number; tol?: number } = {}) {
    super({ ...opts, power: 2, link: "log" });
  }
}
