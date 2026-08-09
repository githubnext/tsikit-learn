/**
 * Covariance estimators: LedoitWolf, OAS, and ShrunkCovariance.
 * Analogous to sklearn.covariance._shrunk_covariance and _ledoit_wolf.
 */

import { NotFittedError } from "../exceptions.js";

/** Result of a covariance estimate. */
export interface CovResult {
  /** Estimated covariance matrix (flat, nFeatures × nFeatures). */
  covariance: Float64Array;
  /** Estimated precision matrix (inverse of covariance). */
  precision: Float64Array;
  nFeatures: number;
}

/**
 * Computes the sample covariance matrix from a flat (nSamples × nFeatures) matrix X
 * that has already been mean-centered.
 */
function sampleCov(
  X: Float64Array,
  nSamples: number,
  nFeatures: number,
): Float64Array {
  const cov = new Float64Array(nFeatures * nFeatures);
  const scale = 1 / (nSamples - 1);
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++) {
      for (let k = j; k < nFeatures; k++) {
        const v = X[i * nFeatures + j]! * X[i * nFeatures + k]! * scale;
        cov[j * nFeatures + k]! += v;
        if (k !== j) cov[k * nFeatures + j]! += v;
      }
    }
  }
  return cov;
}

/** Centers X in-place and returns the column means. */
function centerMatrix(
  X: Float64Array,
  nSamples: number,
  nFeatures: number,
): Float64Array {
  const means = new Float64Array(nFeatures);
  for (let i = 0; i < nSamples; i++)
    for (let j = 0; j < nFeatures; j++) means[j]! += X[i * nFeatures + j] ?? 0;
  for (let j = 0; j < nFeatures; j++) means[j]! /= nSamples;
  for (let i = 0; i < nSamples; i++)
    for (let j = 0; j < nFeatures; j++) X[i * nFeatures + j]! -= means[j] ?? 0;
  return means;
}

/** Applies a shrinkage factor α: Σ_shrunk = (1-α)·S + α·(tr(S)/p)·I */
function shrinkCov(S: Float64Array, p: number, alpha: number): Float64Array {
  const mu = (() => {
    let t = 0;
    for (let j = 0; j < p; j++) t += S[j * p + j]!;
    return t / p;
  })();
  const out = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) out[i * p + j] = (1 - alpha) * S[i * p + j]!;
    out[i * p + i]! += alpha * mu;
  }
  return out;
}

/** Inverts a symmetric positive-definite p×p matrix via Gauss-Jordan. */
function invertPD(A: Float64Array, p: number): Float64Array {
  const aug = new Float64Array(p * 2 * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) aug[i * 2 * p + j] = A[i * p + j]!;
    aug[i * 2 * p + p + i] = 1;
  }
  for (let col = 0; col < p; col++) {
    let maxRow = col;
    for (let r = col + 1; r < p; r++)
      if (
        Math.abs(aug[r * 2 * p + col]!) > Math.abs(aug[maxRow * 2 * p + col]!)
      )
        maxRow = r;
    if (maxRow !== col) {
      for (let k = 0; k < 2 * p; k++) {
        const tmp = aug[col * 2 * p + k]!;
        aug[col * 2 * p + k] = aug[maxRow * 2 * p + k]!;
        aug[maxRow * 2 * p + k] = tmp;
      }
    }
    const pivot = aug[col * 2 * p + col]!;
    if (Math.abs(pivot) < 1e-14) continue;
    for (let k = 0; k < 2 * p; k++) aug[col * 2 * p + k]! /= pivot;
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const f = aug[r * 2 * p + col]!;
      for (let k = 0; k < 2 * p; k++)
        aug[r * 2 * p + k]! -= f * aug[col * 2 * p + k]!;
    }
  }
  const inv = new Float64Array(p * p);
  for (let i = 0; i < p; i++)
    for (let j = 0; j < p; j++) inv[i * p + j] = aug[i * 2 * p + p + j]!;
  return inv;
}

// ─── ShrunkCovariance ──────────────────────────────────────────────────────

export interface ShrunkCovarianceOptions {
  /** Shrinkage coefficient in [0, 1]. Default 0.1. */
  shrinkage?: number;
  /** Whether to store the precision matrix. Default true. */
  storePrecision?: boolean;
  /** Whether to assume the data is already centered. Default false. */
  assumeCentered?: boolean;
}

/** Covariance estimator with manually set shrinkage (Ledoit-Wolf is automatic). */
export class ShrunkCovariance {
  private opts: Required<ShrunkCovarianceOptions>;
  covariance_: Float64Array | undefined;
  precision_: Float64Array | undefined;
  location_: Float64Array | undefined;

  constructor(opts: ShrunkCovarianceOptions = {}) {
    this.opts = {
      shrinkage: opts.shrinkage ?? 0.1,
      storePrecision: opts.storePrecision ?? true,
      assumeCentered: opts.assumeCentered ?? false,
    };
  }

  fit(X: Float64Array, nSamples: number, nFeatures: number): this {
    const Xc = new Float64Array(X);
    let location: Float64Array;
    if (this.opts.assumeCentered) {
      location = new Float64Array(nFeatures);
    } else {
      location = centerMatrix(Xc, nSamples, nFeatures);
    }
    this.location_ = location;
    const S = sampleCov(Xc, nSamples, nFeatures);
    this.covariance_ = shrinkCov(S, nFeatures, this.opts.shrinkage);
    if (this.opts.storePrecision)
      this.precision_ = invertPD(this.covariance_, nFeatures);
    return this;
  }

  score(X: Float64Array, nSamples: number, nFeatures: number): number {
    if (!this.covariance_)
      throw new NotFittedError("ShrunkCovariance is not fitted");
    return logLikelihood(
      X,
      nSamples,
      nFeatures,
      this.covariance_,
      this.location_!,
    );
  }
}

// ─── OAS ───────────────────────────────────────────────────────────────────

export interface OASOptions {
  storePrecision?: boolean;
  assumeCentered?: boolean;
}

/**
 * Oracle Approximating Shrinkage (OAS) covariance estimator.
 * More accurate than Ledoit-Wolf for Gaussian data when n < p.
 */
export class OAS {
  private opts: Required<OASOptions>;
  covariance_: Float64Array | undefined;
  precision_: Float64Array | undefined;
  shrinkage_: number | undefined;
  location_: Float64Array | undefined;

  constructor(opts: OASOptions = {}) {
    this.opts = {
      storePrecision: opts.storePrecision ?? true,
      assumeCentered: opts.assumeCentered ?? false,
    };
  }

  fit(X: Float64Array, nSamples: number, nFeatures: number): this {
    const n = nSamples;
    const p = nFeatures;
    const Xc = new Float64Array(X);
    let location: Float64Array;
    if (this.opts.assumeCentered) {
      location = new Float64Array(p);
    } else {
      location = centerMatrix(Xc, n, p);
    }
    this.location_ = location;
    const S = sampleCov(Xc, n, p);

    // OAS shrinkage estimate
    const trS = (() => {
      let t = 0;
      for (let j = 0; j < p; j++) t += S[j * p + j]!;
      return t;
    })();
    const trS2 = (() => {
      let t = 0;
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++) t += S[i * p + j]! * S[j * p + i]!;
      return t;
    })();

    const mu = trS / p;
    const rho1 =
      ((1 - 2 / p) * trS2 + trS * trS) /
      ((n + 1 - 2 / p) * (trS2 - (trS * trS) / p));
    const alpha = Math.min(1, Math.max(0, rho1));
    this.shrinkage_ = alpha;
    this.covariance_ = shrinkCov(S, p, alpha);
    if (this.opts.storePrecision)
      this.precision_ = invertPD(this.covariance_, p);
    // suppress unused warning
    void mu;
    return this;
  }

  score(X: Float64Array, nSamples: number, nFeatures: number): number {
    if (!this.covariance_) throw new NotFittedError("OAS is not fitted");
    return logLikelihood(
      X,
      nSamples,
      nFeatures,
      this.covariance_,
      this.location_!,
    );
  }
}

// ─── Shared log-likelihood ─────────────────────────────────────────────────

/** Gaussian log-likelihood of X given a covariance estimate. */
function logLikelihood(
  X: Float64Array,
  nSamples: number,
  nFeatures: number,
  cov: Float64Array,
  loc: Float64Array,
): number {
  const p = nFeatures;
  const prec = invertPD(cov, p);
  let ll = 0;
  for (let i = 0; i < nSamples; i++) {
    let quad = 0;
    for (let j = 0; j < p; j++) {
      let row = 0;
      for (let k = 0; k < p; k++)
        row += prec[j * p + k]! * (X[i * p + k]! - loc[k]!);
      quad += (X[i * p + j]! - loc[j]!) * row;
    }
    ll -= 0.5 * quad;
  }
  // Subtract 0.5 * n * log|Σ|
  let logDet = 0;
  // Use the diagonal of a Cholesky factorisation for log-det
  const L = choleskyDiag(cov, p);
  for (let j = 0; j < p; j++) logDet += 2 * Math.log(Math.max(L[j]!, 1e-15));
  ll -= 0.5 * nSamples * logDet;
  ll -= 0.5 * nSamples * p * Math.log(2 * Math.PI);
  return ll / nSamples;
}

/** Returns only the diagonal of the lower Cholesky factor (for log-det). */
function choleskyDiag(A: Float64Array, p: number): Float64Array {
  const L = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * p + j]!;
      for (let k = 0; k < j; k++) s -= L[i * p + k]! * L[j * p + k]!;
      if (i === j) {
        L[i * p + j] = Math.sqrt(Math.max(s, 0));
      } else {
        L[i * p + j] = L[j * p + j]! > 0 ? s / L[j * p + j]! : 0;
      }
    }
  }
  return Float64Array.from({ length: p }, (_, j) => L[j * p + j]!);
}
