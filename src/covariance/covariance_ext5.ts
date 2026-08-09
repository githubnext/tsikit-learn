/**
 * Covariance extensions: OAS, OASCovariance, LedoitWolf extensions.
 * Mirrors sklearn.covariance advanced estimators.
 */

import { BaseEstimator } from "../base.js";

/** Oracle Approximating Shrinkage (OAS) covariance estimator. */
export class OASCovariance extends BaseEstimator {
  covariance_: Float64Array[] = [];
  precision_: Float64Array[] = [];
  shrinkage_: number = 0;
  location_: Float64Array = new Float64Array(0);

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.location_ = new Float64Array(p);
    for (const xi of X) for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) / n;
    // Sample covariance
    const S = Array.from({ length: p }, () => new Float64Array(p));
    for (const xi of X) {
      const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0));
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i]![j] = (S[i]![j] ?? 0) + (xc[i] ?? 0) * (xc[j] ?? 0);
    }
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i]![j] = (S[i]![j] ?? 0) / n;
    // OAS shrinkage coefficient
    let trS = 0, trS2 = 0, trS_sq = 0;
    for (let i = 0; i < p; i++) { trS += S[i]?.[i] ?? 0; for (let j = 0; j < p; j++) trS2 += ((S[i]?.[j] ?? 0) ** 2); }
    trS_sq = trS ** 2;
    const rho_num = (1 - 2 / p) * trS2 + trS_sq;
    const rho_denom = (n + 1 - 2 / p) * (trS2 - trS_sq / p);
    this.shrinkage_ = rho_denom !== 0 ? Math.min(1, rho_num / rho_denom) : 1;
    const mu = trS / p;
    this.covariance_ = S.map((row, i) =>
      new Float64Array(row.map((v, j) => (1 - this.shrinkage_) * v + (i === j ? this.shrinkage_ * mu : 0))),
    );
    this.precision_ = this._invert(this.covariance_, p);
    return this;
  }

  private _invert(A: Float64Array[], p: number): Float64Array[] {
    // Gauss-Jordan elimination
    const aug = A.map((row, i) => {
      const r = new Float64Array(2 * p);
      for (let j = 0; j < p; j++) r[j] = row[j] ?? 0;
      r[p + i] = 1;
      return r;
    });
    for (let i = 0; i < p; i++) {
      let maxRow = i;
      for (let k = i + 1; k < p; k++) if (Math.abs(aug[k]?.[i] ?? 0) > Math.abs(aug[maxRow]?.[i] ?? 0)) maxRow = k;
      [aug[i], aug[maxRow]] = [aug[maxRow]!, aug[i]!];
      const pivot = aug[i]?.[i] ?? 1e-10;
      if (Math.abs(pivot) < 1e-10) continue;
      for (let j = 0; j < 2 * p; j++) aug[i]![j] = (aug[i]![j] ?? 0) / pivot;
      for (let k = 0; k < p; k++) {
        if (k === i) continue;
        const factor = aug[k]?.[i] ?? 0;
        for (let j = 0; j < 2 * p; j++) aug[k]![j] = (aug[k]![j] ?? 0) - factor * (aug[i]![j] ?? 0);
      }
    }
    return Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => aug[i]?.[p + j] ?? 0));
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => {
      const xc = new Float64Array(xi.length).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0));
      let d = 0;
      for (let i = 0; i < xc.length; i++) for (let j = 0; j < xc.length; j++) d += (xc[i] ?? 0) * (this.precision_[i]?.[j] ?? 0) * (xc[j] ?? 0);
      return Math.max(d, 0);
    }));
  }
}

/** Ledoit-Wolf analytical covariance estimator. */
export class LedoitWolfExt extends BaseEstimator {
  covariance_: Float64Array[] = [];
  precision_: Float64Array[] = [];
  shrinkage_: number = 0;
  location_: Float64Array = new Float64Array(0);

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.location_ = new Float64Array(p);
    for (const xi of X) for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) / n;
    const Xc = X.map((xi) => new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0)));
    const S = Array.from({ length: p }, () => new Float64Array(p));
    for (const xc of Xc) for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i]![j] = (S[i]![j] ?? 0) + (xc[i] ?? 0) * (xc[j] ?? 0);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i]![j] = (S[i]![j] ?? 0) / n;
    // Ledoit-Wolf analytical formula
    let trS2 = 0, trS = 0;
    for (let i = 0; i < p; i++) { trS += S[i]?.[i] ?? 0; for (let j = 0; j < p; j++) trS2 += ((S[i]?.[j] ?? 0) ** 2); }
    let b2 = 0;
    for (const xc of Xc) {
      const xxt = Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => (xc[i] ?? 0) * (xc[j] ?? 0)));
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
        const diff = (xxt[i]?.[j] ?? 0) - (S[i]?.[j] ?? 0);
        b2 += diff ** 2;
      }
    }
    b2 /= (n ** 2);
    const delta = Math.max(0, Math.min(1, Math.min(b2, trS2) / ((trS2 - trS ** 2 / p) || 1)));
    this.shrinkage_ = delta;
    const mu = trS / p;
    this.covariance_ = S.map((row, i) =>
      new Float64Array(row.map((v, j) => (1 - delta) * v + (i === j ? delta * mu : 0))),
    );
    this.precision_ = this._invert(this.covariance_, p);
    return this;
  }

  private _invert(A: Float64Array[], p: number): Float64Array[] {
    const aug = A.map((row, i) => {
      const r = new Float64Array(2 * p);
      for (let j = 0; j < p; j++) r[j] = row[j] ?? 0;
      r[p + i] = 1;
      return r;
    });
    for (let i = 0; i < p; i++) {
      const pivot = aug[i]?.[i] ?? 1e-10;
      if (Math.abs(pivot) < 1e-10) continue;
      for (let j = 0; j < 2 * p; j++) aug[i]![j] = (aug[i]![j] ?? 0) / pivot;
      for (let k = 0; k < p; k++) {
        if (k === i) continue;
        const f = aug[k]?.[i] ?? 0;
        for (let j = 0; j < 2 * p; j++) aug[k]![j] = (aug[k]![j] ?? 0) - f * (aug[i]![j] ?? 0);
      }
    }
    return Array.from({ length: p }, (_, i) => new Float64Array(p).map((_, j) => aug[i]?.[p + j] ?? 0));
  }
}

/** MinCovDet: Minimum Covariance Determinant estimator. */
export class MinCovDetExt extends BaseEstimator {
  support_fraction_: number;
  location_: Float64Array = new Float64Array(0);
  covariance_: Float64Array[] = [];
  dist_: Float64Array = new Float64Array(0);

  constructor(supportFraction = 0.75) {
    super();
    this.support_fraction_ = supportFraction;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const h = Math.floor(n * this.support_fraction_);
    // Simple C-step: start from all points, iteratively refine
    let support = Array.from({ length: n }, (_, i) => i);
    for (let step = 0; step < 10; step++) {
      const Xs = support.map((i) => X[i]!);
      const loc = new Float64Array(p);
      for (const xi of Xs) for (let k = 0; k < p; k++) loc[k] = (loc[k] ?? 0) + (xi[k] ?? 0);
      for (let k = 0; k < p; k++) loc[k] = (loc[k] ?? 0) / Xs.length;
      const cov = Array.from({ length: p }, () => new Float64Array(p));
      for (const xi of Xs) {
        const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (loc[k] ?? 0));
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) cov[i]![j] = (cov[i]![j] ?? 0) + (xc[i] ?? 0) * (xc[j] ?? 0);
      }
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) cov[i]![j] = (cov[i]![j] ?? 0) / Xs.length;
      // Compute Mahalanobis distances
      const dist = X.map((xi) => {
        let d = 0;
        const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (loc[k] ?? 0));
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) d += (xc[i] ?? 0) * (cov[i]?.[j] ?? 0) * (xc[j] ?? 0);
        return d;
      });
      support = dist.map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d).slice(0, h).map((x) => x.i);
    }
    const Xs = support.map((i) => X[i]!);
    this.location_ = new Float64Array(p);
    for (const xi of Xs) for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < p; k++) this.location_[k] = (this.location_[k] ?? 0) / Xs.length;
    this.covariance_ = Array.from({ length: p }, () => new Float64Array(p));
    for (const xi of Xs) {
      const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0));
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) this.covariance_[i]![j] = (this.covariance_[i]![j] ?? 0) + (xc[i] ?? 0) * (xc[j] ?? 0);
    }
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) this.covariance_[i]![j] = (this.covariance_[i]![j] ?? 0) / Xs.length;
    this.dist_ = new Float64Array(n).map((_, i) => {
      const xi = X[i]!;
      let d = 0;
      const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0));
      for (let ii = 0; ii < p; ii++) for (let j = 0; j < p; j++) d += (xc[ii] ?? 0) * (this.covariance_[ii]?.[j] ?? 0) * (xc[j] ?? 0);
      return d;
    });
    return this;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    const p = this.location_.length;
    return new Float64Array(X.map((xi) => {
      const xc = new Float64Array(p).map((_, k) => (xi[k] ?? 0) - (this.location_[k] ?? 0));
      let d = 0;
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) d += (xc[i] ?? 0) * (this.covariance_[i]?.[j] ?? 0) * (xc[j] ?? 0);
      return Math.max(d, 0);
    }));
  }
}
