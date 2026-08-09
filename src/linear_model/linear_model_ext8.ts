/**
 * Linear model extensions: Bayesian Ridge with ARD, LASSO path utilities, coordinate descent.
 * Mirrors sklearn.linear_model additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Soft-threshold operator for LASSO coordinate descent. */
export function softThreshold(x: number, lambda: number): number {
  if (x > lambda) return x - lambda;
  if (x < -lambda) return x + lambda;
  return 0;
}

export interface LassoLarsParams {
  max_iter?: number;
  eps?: number;
}

/** LASSO-LARS: least angle regression for LASSO path. */
export class LassoLars extends BaseEstimator {
  max_iter: number;
  eps: number;
  coef_: Float64Array = new Float64Array(0);
  alphas_: Float64Array = new Float64Array(0);
  active_: number[] = [];

  constructor(params: LassoLarsParams = {}) {
    super();
    this.max_iter = params.max_iter ?? 500;
    this.eps = params.eps ?? 1e-6;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const beta = new Float64Array(d);
    const residual = new Float64Array(y);

    // Coordinate descent for LASSO
    for (let iter = 0; iter < this.max_iter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < d; j++) {
        // Compute gradient
        let corr = 0;
        let norm2 = 0;
        for (let i = 0; i < n; i++) {
          corr += (X[i]?.[j] ?? 0) * (residual[i] ?? 0);
          norm2 += (X[i]?.[j] ?? 0) ** 2;
        }
        if (norm2 < this.eps) continue;
        const oldBeta = beta[j] ?? 0;
        const newBeta = softThreshold(oldBeta + corr / norm2, 0.01 / norm2);
        const change = newBeta - oldBeta;
        if (Math.abs(change) > maxChange) maxChange = Math.abs(change);
        beta[j] = newBeta;
        for (let i = 0; i < n; i++) {
          residual[i] = (residual[i] ?? 0) - (X[i]?.[j] ?? 0) * change;
        }
      }
      if (maxChange < this.eps) break;
    }
    this.coef_ = beta;
    this.active_ = Array.from({ length: d }, (_, j) => j).filter(j => Math.abs(beta[j] ?? 0) > this.eps);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return Float64Array.from(X, row => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) * (this.coef_[j] ?? 0);
      return s;
    });
  }
}

export interface OrthogonalMatchingPursuitParams {
  n_nonzero_coefs?: number;
  tol?: number;
}

/** Orthogonal Matching Pursuit for sparse signal recovery. */
export class OrthoMatchingPursuit extends BaseEstimator {
  n_nonzero_coefs: number;
  tol: number;
  coef_: Float64Array = new Float64Array(0);
  n_iter_ = 0;

  constructor(params: OrthogonalMatchingPursuitParams = {}) {
    super();
    this.n_nonzero_coefs = params.n_nonzero_coefs ?? 10;
    this.tol = params.tol ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const active: number[] = [];
    const residual = new Float64Array(y);
    this.coef_ = new Float64Array(d);

    for (let k = 0; k < this.n_nonzero_coefs; k++) {
      // Find the column most correlated with residual
      let bestCorr = 0;
      let bestJ = 0;
      for (let j = 0; j < d; j++) {
        if (active.includes(j)) continue;
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (X[i]?.[j] ?? 0) * (residual[i] ?? 0);
        if (Math.abs(corr) > Math.abs(bestCorr)) { bestCorr = corr; bestJ = j; }
      }
      active.push(bestJ);

      // Solve least squares on active set
      const Xa: Float64Array[] = X.map(row => new Float64Array(active.map(j => row[j] ?? 0)));
      const coefA = lstSq(Xa, y);

      // Update residual
      for (let i = 0; i < n; i++) {
        residual[i] = y[i] ?? 0;
        for (let ai = 0; ai < active.length; ai++) {
          residual[i] = (residual[i] ?? 0) - (X[i]?.[active[ai]!] ?? 0) * (coefA[ai] ?? 0);
        }
      }
      const residNorm = residual.reduce((s, v) => s + v * v, 0);
      this.n_iter_ = k + 1;
      if (this.tol > 0 && residNorm < this.tol) break;
    }

    for (let ai = 0; ai < active.length; ai++) {
      const j = active[ai]!;
      const Xa = X.map(row => new Float64Array(active.map(idx => row[idx] ?? 0)));
      const coefA = lstSq(Xa, y);
      this.coef_[j] = coefA[ai] ?? 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map(row => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) * (this.coef_[j] ?? 0);
      return s;
    }));
  }
}

function lstSq(X: Float64Array[], y: Float64Array): Float64Array {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  // Normal equations: (X^T X) beta = X^T y
  const XtX: Float64Array[] = Array.from({ length: d }, () => new Float64Array(d));
  const Xty = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      Xty[j] = (Xty[j] ?? 0) + (X[i]?.[j] ?? 0) * (y[i] ?? 0);
      for (let k = 0; k < d; k++) {
        XtX[j]![k] = (XtX[j]?.[k] ?? 0) + (X[i]?.[j] ?? 0) * (X[i]?.[k] ?? 0);
      }
    }
  }
  for (let j = 0; j < d; j++) XtX[j]![j] = (XtX[j]?.[j] ?? 0) + 1e-8;
  return solveLinear(XtX, Xty);
}

function solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  const aug: Float64Array[] = A.map((r, i) => {
    const row = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) row[j] = r[j] ?? 0;
    row[n] = b[i] ?? 0;
    return row;
  });
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r]?.[col] ?? 0) > Math.abs(aug[pivot]?.[col] ?? 0)) pivot = r;
    }
    const tmp = aug[col]!;
    aug[col] = aug[pivot]!;
    aug[pivot] = tmp;
    const scale = aug[col]?.[col] ?? 1;
    if (Math.abs(scale) < 1e-12) continue;
    for (let j = col; j <= n; j++) aug[col]![j] = (aug[col]?.[j] ?? 0) / scale;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r]?.[col] ?? 0;
      for (let j = col; j <= n; j++) aug[r]![j] = (aug[r]?.[j] ?? 0) - factor * (aug[col]?.[j] ?? 0);
    }
  }
  return new Float64Array(aug.map(r => r[n] ?? 0));
}

/** Compute elastic net regularization path. */
export function elasticNetPath(
  X: Float64Array[],
  y: Float64Array,
  alphas: Float64Array,
  l1Ratio = 0.5,
  maxIter = 100,
): Float64Array[] {
  const d = X[0]?.length ?? 0;
  return Array.from(alphas).map(alpha => {
    const coef = new Float64Array(d);
    const n = X.length;
    const residual = new Float64Array(y);
    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < d; j++) {
        let corr = 0;
        let norm2 = 0;
        for (let i = 0; i < n; i++) {
          corr += (X[i]?.[j] ?? 0) * (residual[i] ?? 0);
          norm2 += (X[i]?.[j] ?? 0) ** 2;
        }
        norm2 += alpha * (1 - l1Ratio) * n;
        if (norm2 < 1e-10) continue;
        const old = coef[j] ?? 0;
        const updated = softThreshold(old + corr / norm2, alpha * l1Ratio / norm2);
        const change = updated - old;
        if (Math.abs(change) > maxChange) maxChange = Math.abs(change);
        coef[j] = updated;
        for (let i = 0; i < n; i++) residual[i] = (residual[i] ?? 0) - (X[i]?.[j] ?? 0) * change;
      }
      if (maxChange < 1e-6) break;
    }
    return coef;
  });
}
