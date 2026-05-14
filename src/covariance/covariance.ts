/**
 * Covariance estimators: EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS.
 * Mirrors sklearn.covariance.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute column means of X. */
function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const means = new Float64Array(p);
  const n = X.length;
  for (const xi of X) {
    for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) / n;
  return means;
}

/** Compute empirical covariance matrix (biased). */
function empCov(X: Float64Array[], means: Float64Array): Float64Array[] {
  const n = X.length;
  const p = means.length;
  const C = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of X) {
    for (let i = 0; i < p; i++) {
      const di = (xi[i] ?? 0) - (means[i] ?? 0);
      for (let j = i; j < p; j++) {
        const dj = (xi[j] ?? 0) - (means[j] ?? 0);
        C[i]![j] = (C[i]![j] ?? 0) + di * dj;
      }
    }
  }
  for (let i = 0; i < p; i++) {
    C[i]![i] = (C[i]![i] ?? 0) / n;
    for (let j = i + 1; j < p; j++) {
      C[i]![j] = (C[i]![j] ?? 0) / n;
      C[j]![i] = C[i]![j] ?? 0;
    }
  }
  return C;
}

/**
 * Maximum likelihood covariance estimator.
 * Mirrors sklearn.covariance.EmpiricalCovariance.
 */
export class EmpiricalCovariance {
  assumeCentered: boolean;

  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;

  constructor(options: { assumeCentered?: boolean } = {}) {
    this.assumeCentered = options.assumeCentered ?? false;
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    if (this.assumeCentered) {
      this.location_ = new Float64Array(p);
    } else {
      this.location_ = colMeans(X);
    }
    this.covariance_ = empCov(X, this.location_);
    return this;
  }

  score(X: Float64Array[]): number {
    if (this.covariance_ === null || this.location_ === null) throw new NotFittedError();
    // Negative log-likelihood
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    let logdet = 0;
    // Approximate log-det via trace of covariance
    for (let i = 0; i < p; i++) {
      logdet += Math.log(Math.abs(this.covariance_[i]![i] ?? 1) + 1e-12);
    }
    let trace = 0;
    for (const xi of X) {
      const centered = new Float64Array(p);
      for (let j = 0; j < p; j++) centered[j] = (xi[j] ?? 0) - (this.location_![j] ?? 0);
      for (let j = 0; j < p; j++) {
        const cjj = this.covariance_![j]![j] ?? 1e-12;
        trace += (centered[j] ?? 0) ** 2 / (cjj || 1e-12);
      }
    }
    return -(n * logdet + trace) / 2;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    if (this.covariance_ === null || this.location_ === null) throw new NotFittedError();
    const p = (X[0] ?? new Float64Array(0)).length;
    const dists = new Float64Array(X.length);
    for (let idx = 0; idx < X.length; idx++) {
      const xi = X[idx] ?? new Float64Array(p);
      let d = 0;
      for (let j = 0; j < p; j++) {
        const diff = (xi[j] ?? 0) - (this.location_![j] ?? 0);
        const cjj = this.covariance_![j]![j] ?? 1e-12;
        d += diff ** 2 / (cjj || 1e-12);
      }
      dists[idx] = Math.sqrt(d);
    }
    return dists;
  }
}

/**
 * Covariance estimator with shrinkage.
 * Mirrors sklearn.covariance.ShrunkCovariance.
 */
export class ShrunkCovariance extends EmpiricalCovariance {
  shrinkage: number;

  constructor(options: { assumeCentered?: boolean; shrinkage?: number } = {}) {
    super(options);
    this.shrinkage = options.shrinkage ?? 0.1;
  }

  override fit(X: Float64Array[]): this {
    super.fit(X);
    if (this.covariance_ !== null) {
      const p = this.covariance_.length;
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          if (i === j) continue;
          this.covariance_[i]![j] = (this.covariance_![i]![j] ?? 0) * (1 - this.shrinkage);
        }
      }
    }
    return this;
  }
}

/**
 * Ledoit-Wolf automatic covariance estimator.
 * Mirrors sklearn.covariance.LedoitWolf.
 */
export class LedoitWolf extends EmpiricalCovariance {
  blockSize: number;

  shrinkage_: number | null = null;

  constructor(options: { assumeCentered?: boolean; blockSize?: number } = {}) {
    super(options);
    this.blockSize = options.blockSize ?? 1000;
  }

  override fit(X: Float64Array[]): this {
    super.fit(X);
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    if (this.covariance_ !== null) {
      // Oracle Approximating Shrinkage estimator (simplified Ledoit-Wolf)
      let mu = 0;
      for (let i = 0; i < p; i++) mu += this.covariance_![i]![i] ?? 0;
      mu /= p;

      let delta = 0;
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          delta += (this.covariance_![i]![j] ?? 0) ** 2;
        }
      }

      const traceS2 = delta;
      const traceS = p * mu;
      const beta = (1 / (n * p)) * (traceS2 - traceS ** 2 / p);
      const alpha = Math.max(0, Math.min(1, beta / delta));
      this.shrinkage_ = alpha;

      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          this.covariance_![i]![j] =
            (1 - alpha) * (this.covariance_![i]![j] ?? 0) + (i === j ? alpha * mu : 0);
        }
      }
    }
    return this;
  }
}

/**
 * Oracle Approximating Shrinkage estimator.
 * Mirrors sklearn.covariance.OAS.
 */
export class OAS extends EmpiricalCovariance {
  shrinkage_: number | null = null;

  override fit(X: Float64Array[]): this {
    super.fit(X);
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    if (this.covariance_ !== null) {
      let trS = 0;
      let trS2 = 0;
      for (let i = 0; i < p; i++) {
        const sii = this.covariance_![i]![i] ?? 0;
        trS += sii;
        for (let j = 0; j < p; j++) {
          trS2 += (this.covariance_![i]![j] ?? 0) ** 2;
        }
      }
      const mu = trS / p;
      const rho = Math.max(
        0,
        Math.min(
          1,
          ((1 - 2 / p) * trS2 + trS ** 2) /
            ((n + 1 - 2 / p) * (trS2 - trS ** 2 / p)),
        ),
      );
      this.shrinkage_ = rho;
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          this.covariance_![i]![j] =
            (1 - rho) * (this.covariance_![i]![j] ?? 0) + (i === j ? rho * mu : 0);
        }
      }
    }
    return this;
  }
}
