/**
 * Extended linear model utilities: Tweedie/Poisson/Gamma GLM helpers,
 * coordinate-descent utilities, and link function implementations.
 */

/** Link function types matching sklearn's GLM link functions. */
export type LinkFunction = "identity" | "log" | "logit" | "probit" | "cloglog";

/** Apply the link function: eta = link(mu). */
export function applyLink(mu: Float64Array, link: LinkFunction): Float64Array {
  const eta = new Float64Array(mu.length);
  for (let i = 0; i < mu.length; i++) {
    const m = mu[i] ?? 0;
    switch (link) {
      case "identity":
        eta[i] = m;
        break;
      case "log":
        eta[i] = Math.log(Math.max(m, 1e-10));
        break;
      case "logit":
        eta[i] = Math.log(m / (1 - m + 1e-10));
        break;
      case "probit":
        eta[i] = probitInverse(m);
        break;
      case "cloglog":
        eta[i] = Math.log(-Math.log(1 - m + 1e-10));
        break;
    }
  }
  return eta;
}

/** Inverse link (mean function): mu = h(eta). */
export function inverseLink(eta: Float64Array, link: LinkFunction): Float64Array {
  const mu = new Float64Array(eta.length);
  for (let i = 0; i < eta.length; i++) {
    const e = eta[i] ?? 0;
    switch (link) {
      case "identity":
        mu[i] = e;
        break;
      case "log":
        mu[i] = Math.exp(e);
        break;
      case "logit":
        mu[i] = 1 / (1 + Math.exp(-e));
        break;
      case "probit":
        mu[i] = normalCDF(e);
        break;
      case "cloglog":
        mu[i] = 1 - Math.exp(-Math.exp(e));
        break;
    }
  }
  return mu;
}

/** Normal CDF approximation (Abramowitz & Stegun). */
function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const phi = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);
  const cdf = 1 - phi * poly;
  return x >= 0 ? cdf : 1 - cdf;
}

/** Inverse of normal CDF (probit). */
function probitInverse(p: number): number {
  const pClipped = Math.max(1e-10, Math.min(1 - 1e-10, p));
  // Rational approximation
  if (pClipped < 0.5) {
    const t = Math.sqrt(-2 * Math.log(pClipped));
    return -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
  }
  const t = Math.sqrt(-2 * Math.log(1 - pClipped));
  return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
}

/** Tweedie deviance residuals. */
export function tweediDeviance(y: Float64Array, yPred: Float64Array, power: number): number {
  let dev = 0;
  for (let i = 0; i < y.length; i++) {
    const yi = y[i] ?? 0;
    const yHat = Math.max(yPred[i] ?? 1e-10, 1e-10);
    if (power === 0) {
      dev += (yi - yHat) ** 2;
    } else if (power === 1) {
      dev += 2 * (yi * Math.log(yi / yHat + 1e-10) - (yi - yHat));
    } else if (power === 2) {
      dev += 2 * (Math.log(yHat / (yi + 1e-10)) + (yi / yHat) - 1);
    } else {
      dev += 2 * (
        (yi ** (2 - power)) / ((1 - power) * (2 - power)) -
        yi * yHat ** (1 - power) / (1 - power) +
        yHat ** (2 - power) / (2 - power)
      );
    }
  }
  return dev;
}

/** Poisson deviance. */
export function poissonDeviance(y: Float64Array, yPred: Float64Array): number {
  return tweediDeviance(y, yPred, 1);
}

/** Gamma deviance. */
export function gammaDeviance(y: Float64Array, yPred: Float64Array): number {
  return tweediDeviance(y, yPred, 2);
}

/** D² score (generalization of R² for GLMs). */
export function d2TweedieScore(y: Float64Array, yPred: Float64Array, power: number): number {
  const nullPred = new Float64Array(y.length).fill(
    y.reduce((a, b) => a + b, 0) / y.length
  );
  const devNull = tweediDeviance(y, nullPred, power);
  const devModel = tweediDeviance(y, yPred, power);
  return 1 - devModel / (devNull + 1e-10);
}

/** IRLS (Iteratively Reweighted Least Squares) update step. */
export function irlsStep(
  X: Float64Array[],
  y: Float64Array,
  beta: Float64Array,
  link: LinkFunction,
): Float64Array {
  const n = X.length;
  const p = beta.length;
  // eta = X @ beta
  const eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const xi = X[i];
    if (xi === undefined) continue;
    for (let j = 0; j < p; j++) sum += (xi[j] ?? 0) * (beta[j] ?? 0);
    eta[i] = sum;
  }
  const mu = inverseLink(eta, link);
  // Working response z = eta + (y - mu) * d_eta/d_mu
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = eta[i] ?? 0;
    const m = mu[i] ?? 1e-10;
    const yi = y[i] ?? 0;
    let dEtaDMu: number;
    switch (link) {
      case "identity": dEtaDMu = 1; break;
      case "log": dEtaDMu = 1 / m; break;
      case "logit": dEtaDMu = 1 / (m * (1 - m) + 1e-10); break;
      default: dEtaDMu = 1;
    }
    z[i] = e + (yi - m) * dEtaDMu;
  }
  // Simple gradient step: beta += X^T (z - eta) / n
  const grad = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let g = 0;
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      if (xi === undefined) continue;
      g += (xi[j] ?? 0) * ((z[i] ?? 0) - (eta[i] ?? 0));
    }
    grad[j] = g / n;
  }
  const newBeta = new Float64Array(p);
  for (let j = 0; j < p; j++) newBeta[j] = (beta[j] ?? 0) + 0.01 * (grad[j] ?? 0);
  return newBeta;
}
