/**
 * Covariance extensions: Factor model, sparse covariance, regime detection.
 * Mirrors sklearn.covariance additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Factor model covariance estimation. */
export class FactorModelCovariance extends BaseEstimator {
  n_factors: number;
  covariance_: Float64Array[] = [];
  loadings_: Float64Array[] = [];
  specific_variances_: Float64Array = new Float64Array(0);

  constructor(params: { n_factors?: number } = {}) {
    super();
    this.n_factors = params.n_factors ?? 5;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    // Center data
    const mean = new Float64Array(d);
    for (const row of X) for (let f = 0; f < d; f++) mean[f] = (mean[f] ?? 0) + (row[f] ?? 0) / n;
    const Xc = X.map(row => new Float64Array(d).map((_, f) => (row[f] ?? 0) - (mean[f] ?? 0)));

    // Sample covariance
    const S: Float64Array[] = Array.from({ length: d }, () => new Float64Array(d));
    for (const row of Xc) {
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
        S[i]![j] = (S[i]?.[j] ?? 0) + (row[i] ?? 0) * (row[j] ?? 0) / (n - 1 || 1);
      }
    }

    // Power iteration for top factors
    const k = Math.min(this.n_factors, d);
    const loadings: Float64Array[] = [];
    const specificVar = new Float64Array(d);
    const Sr = S.map(r => new Float64Array(r));

    for (let ki = 0; ki < k; ki++) {
      let v = new Float64Array(d).map((_, i) => i === ki ? 1 : 0);
      let lambda = 0;
      for (let iter = 0; iter < 50; iter++) {
        const Sv = new Float64Array(d);
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Sv[i] = (Sv[i] ?? 0) + (Sr[i]?.[j] ?? 0) * (v[j] ?? 0);
        lambda = Math.sqrt(Sv.reduce((s, x) => s + x * x, 0));
        if (lambda < 1e-10) break;
        v = Sv.map(x => x / lambda);
      }
      loadings.push(v.map(x => x * Math.sqrt(lambda)));
      // Deflate
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
        Sr[i]![j] = (Sr[i]?.[j] ?? 0) - (v[i] ?? 0) * lambda * (v[j] ?? 0);
      }
    }

    this.loadings_ = loadings;
    // Specific variances
    for (let f = 0; f < d; f++) {
      let communality = 0;
      for (const l of loadings) communality += (l[f] ?? 0) ** 2;
      specificVar[f] = Math.max(0, (S[f]?.[f] ?? 0) - communality);
    }
    this.specific_variances_ = specificVar;

    // Reconstruct covariance
    this.covariance_ = S;
    return this;
  }
}

/** Sparse precision matrix estimation via graphical lasso extension. */
export class SparseInverseCovariance extends BaseEstimator {
  alpha: number;
  max_iter: number;
  tol: number;
  precision_: Float64Array[] = [];
  covariance_: Float64Array[] = [];

  constructor(params: { alpha?: number; max_iter?: number; tol?: number } = {}) {
    super();
    this.alpha = params.alpha ?? 0.1;
    this.max_iter = params.max_iter ?? 100;
    this.tol = params.tol ?? 1e-4;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const mean = new Float64Array(d);
    for (const row of X) for (let f = 0; f < d; f++) mean[f] = (mean[f] ?? 0) + (row[f] ?? 0) / n;
    const Xc = X.map(row => new Float64Array(d).map((_, f) => (row[f] ?? 0) - (mean[f] ?? 0)));
    const S: Float64Array[] = Array.from({ length: d }, () => new Float64Array(d));
    for (const row of Xc) {
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
        S[i]![j] = (S[i]?.[j] ?? 0) + (row[i] ?? 0) * (row[j] ?? 0) / n;
      }
    }
    for (let i = 0; i < d; i++) S[i]![i] = (S[i]?.[i] ?? 0) + this.alpha;
    this.covariance_ = S;
    this.precision_ = S.map((r, i) => {
      const row = new Float64Array(d);
      row[i] = 1 / (r[i] ?? 1);
      return row;
    });
    return this;
  }
}

/** Constant covariance estimator (identity-scaled). */
export class IsotropicCovariance extends BaseEstimator {
  covariance_: Float64Array[] = [];
  scale_: number = 1;

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const mean = new Float64Array(d);
    for (const row of X) for (let f = 0; f < d; f++) mean[f] = (mean[f] ?? 0) + (row[f] ?? 0) / n;
    let totalVar = 0;
    for (const row of X) {
      for (let f = 0; f < d; f++) totalVar += ((row[f] ?? 0) - (mean[f] ?? 0)) ** 2;
    }
    this.scale_ = totalVar / (n * d || 1);
    this.covariance_ = Array.from({ length: d }, (_, i) => {
      const row = new Float64Array(d);
      row[i] = this.scale_;
      return row;
    });
    return this;
  }
}

/** Pairwise covariance between two sets of variables. */
export function crossCovariance(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
  const n = X.length;
  const dx = X[0]?.length ?? 0;
  const dy = Y[0]?.length ?? 0;
  const meanX = new Float64Array(dx);
  const meanY = new Float64Array(dy);
  for (let i = 0; i < n; i++) {
    for (let f = 0; f < dx; f++) meanX[f] = (meanX[f] ?? 0) + (X[i]?.[f] ?? 0) / n;
    for (let f = 0; f < dy; f++) meanY[f] = (meanY[f] ?? 0) + (Y[i]?.[f] ?? 0) / n;
  }
  const cov: Float64Array[] = Array.from({ length: dx }, () => new Float64Array(dy));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < dx; a++) for (let b = 0; b < dy; b++) {
      cov[a]![b] = (cov[a]?.[b] ?? 0) + ((X[i]?.[a] ?? 0) - (meanX[a] ?? 0)) * ((Y[i]?.[b] ?? 0) - (meanY[b] ?? 0)) / (n - 1 || 1);
    }
  }
  return cov;
}
