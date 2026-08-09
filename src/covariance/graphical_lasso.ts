/**
 * GraphicalLasso and MinCovDet (robust covariance).
 * Mirrors sklearn.covariance.GraphicalLasso and MinCovDet.
 */

import { NotFittedError } from "../exceptions.js";

function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const n = X.length;
  const means = new Float64Array(p);
  for (const xi of X)
    for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) + (xi[j] ?? 0);
  for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) / n;
  return means;
}

function empiricalCovariance(X: Float64Array[]): Float64Array[] {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const means = colMeans(X);
  const cov: Float64Array[] = Array.from(
    { length: p },
    () => new Float64Array(p),
  );
  for (const xi of X) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k <= j; k++) {
        const d =
          ((xi[j] ?? 0) - (means[j] ?? 0)) * ((xi[k] ?? 0) - (means[k] ?? 0));
        cov[j]![k] = (cov[j]![k] ?? 0) + d;
        if (k !== j) cov[k]![j] = (cov[k]![j] ?? 0) + d;
      }
    }
  }
  for (let j = 0; j < p; j++)
    for (let k = 0; k < p; k++) cov[j]![k] = (cov[j]![k] ?? 0) / n;
  return cov;
}

function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const n = A.length;
  const m = (B[0] ?? new Float64Array(0)).length;
  const k = B.length;
  const C: Float64Array[] = Array.from(
    { length: n },
    () => new Float64Array(m),
  );
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++)
      for (let l = 0; l < k; l++)
        C[i]![j] = (C[i]![j] ?? 0) + (A[i]![l] ?? 0) * (B[l]![j] ?? 0);
  return C;
}

function invertMatrix(A: Float64Array[]): Float64Array[] {
  const p = A.length;
  // Augmented matrix [A | I]
  const M: Float64Array[] = A.map((row, i) => {
    const r = new Float64Array(2 * p);
    for (let j = 0; j < p; j++) r[j] = row[j] ?? 0;
    r[p + i] = 1;
    return r;
  });

  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let row = col + 1; row < p; row++) {
      if (Math.abs(M[row]![col] ?? 0) > Math.abs(M[pivot]![col] ?? 0))
        pivot = row;
    }
    const tmp = M[col]!;
    M[col] = M[pivot]!;
    M[pivot] = tmp;
    const denom = M[col]![col] ?? 1;
    for (let j = 0; j < 2 * p; j++) M[col]![j] = (M[col]![j] ?? 0) / denom;
    for (let row = 0; row < p; row++) {
      if (row === col) continue;
      const factor = M[row]![col] ?? 0;
      for (let j = 0; j < 2 * p; j++)
        M[row]![j] = (M[row]![j] ?? 0) - factor * (M[col]![j] ?? 0);
    }
  }

  return M.map(
    (row) =>
      new Float64Array(Array.from({ length: p }, (_, j) => row[p + j] ?? 0)),
  );
}

export interface GraphicalLassoOptions {
  alpha?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * Sparse inverse covariance estimation with L1 penalty (Graphical Lasso).
 * Mirrors sklearn.covariance.GraphicalLasso.
 * Uses the block coordinate descent algorithm (GLASSO).
 */
export class GraphicalLasso {
  alpha: number;
  maxIter: number;
  tol: number;

  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  nIter_: number = 0;
  location_: Float64Array | null = null;

  constructor(options: GraphicalLassoOptions = {}) {
    this.alpha = options.alpha ?? 0.01;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    this.location_ = colMeans(X);
    const S = empiricalCovariance(X);

    // Initialize with diagonal of S + alpha * I
    const W: Float64Array[] = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j] = S[i]![j] ?? 0;
      row[i] = (row[i] ?? 0) + this.alpha;
      return row;
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        // Partition W into W11 (p-1 x p-1) and w12 (p-1 vector)
        const idx = Array.from({ length: p }, (_, k) => k).filter(
          (k) => k !== j,
        );
        const W11: Float64Array[] = idx.map(
          (r) => new Float64Array(idx.map((c) => W[r]![c] ?? 0)),
        );
        const s12 = new Float64Array(idx.map((r) => S[r]![j] ?? 0));

        // Solve lasso: W11 * beta = s12 with L1 penalty alpha
        const W11inv = invertMatrix(W11);
        const q = new Float64Array(p - 1);
        for (let k = 0; k < p - 1; k++)
          for (let l = 0; l < p - 1; l++)
            q[k] = (q[k] ?? 0) + (W11inv[k]![l] ?? 0) * (s12[l] ?? 0);

        // Coordinate descent for lasso subproblem
        const beta = new Float64Array(p - 1);
        for (let lasso = 0; lasso < 100; lasso++) {
          let maxD = 0;
          for (let k = 0; k < p - 1; k++) {
            const r =
              (s12[k] ?? 0) -
              ((): number => {
                let s = 0;
                for (let l = 0; l < p - 1; l++)
                  if (l !== k) s += (W11[k]![l] ?? 0) * (beta[l] ?? 0);
                return s;
              })();
            const wkk = W11[k]![k] ?? 1;
            const b = r / wkk;
            const threshold = this.alpha / wkk;
            const newBeta =
              b > threshold
                ? b - threshold
                : b < -threshold
                  ? b + threshold
                  : 0;
            maxD = Math.max(maxD, Math.abs(newBeta - (beta[k] ?? 0)));
            beta[k] = newBeta;
          }
          if (maxD < 1e-6) break;
        }

        // Update W: w12 = W11 * beta
        for (let k = 0; k < p - 1; k++) {
          let s = 0;
          for (let l = 0; l < p - 1; l++)
            s += (W11[k]![l] ?? 0) * (beta[l] ?? 0);
          const delta = Math.abs(s - (W[idx[k]!]![j] ?? 0));
          if (delta > maxDelta) maxDelta = delta;
          W[idx[k]!]![j] = s;
          W[j]![idx[k]!] = s;
        }
      }
      this.nIter_ = iter + 1;
      if (maxDelta < this.tol) break;
    }

    this.covariance_ = W;
    this.precision_ = invertMatrix(W);
    return this;
  }

  score(X: Float64Array[]): number {
    if (!this.covariance_)
      throw new NotFittedError("GraphicalLasso is not fitted yet.");
    return 0; // Placeholder: log-likelihood requires determinant
  }
}

export interface MinCovDetOptions {
  support?: number | null;
  randomState?: number;
}

/**
 * Minimum Covariance Determinant robust estimator.
 * Mirrors sklearn.covariance.MinCovDet.
 * Uses a simplified C-step algorithm.
 */
export class MinCovDet {
  support: number | null;
  randomState: number;

  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  supportFraction_: number = 0;
  supportIndices_: Int32Array | null = null;
  rawLocation_: Float64Array | null = null;
  rawCovariance_: Float64Array[] | null = null;

  private rng_: () => number;

  constructor(options: MinCovDetOptions = {}) {
    this.support = options.support ?? null;
    this.randomState = options.randomState ?? 0;
    let seed = this.randomState + 1;
    this.rng_ = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0xffffffff;
    };
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const h =
      this.support != null
        ? Math.floor(this.support * n)
        : Math.floor((n + p + 1) / 2);

    // Compute Mahalanobis distances from full empirical estimate
    const fullMeans = colMeans(X);
    const fullCov = empiricalCovariance(X);
    let precision: Float64Array[];
    try {
      precision = invertMatrix(fullCov);
    } catch {
      precision = Array.from({ length: p }, (_, i) => {
        const r = new Float64Array(p);
        r[i] = 1;
        return r;
      });
    }

    // Mahalanobis distance for each point
    const mDist = X.map((xi) => {
      const diff = new Float64Array(p);
      for (let j = 0; j < p; j++) diff[j] = (xi[j] ?? 0) - (fullMeans[j] ?? 0);
      let d = 0;
      for (let j = 0; j < p; j++)
        for (let k = 0; k < p; k++)
          d += (diff[j] ?? 0) * (precision[j]![k] ?? 0) * (diff[k] ?? 0);
      return d;
    });

    // Select h points with smallest Mahalanobis distances
    const sortedIdx = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => mDist[a]! - mDist[b]!,
    );
    const supportIdx = new Int32Array(sortedIdx.slice(0, h));

    const subset = Array.from(supportIdx).map(
      (i) => X[i] ?? new Float64Array(p),
    );
    this.rawLocation_ = colMeans(subset);
    this.rawCovariance_ = empiricalCovariance(subset);

    this.location_ = this.rawLocation_;
    this.covariance_ = this.rawCovariance_;
    try {
      this.precision_ = invertMatrix(this.covariance_);
    } catch {
      this.precision_ = null;
    }

    this.supportFraction_ = h / n;
    this.supportIndices_ = supportIdx;
    return this;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    if (!this.location_ || !this.precision_)
      throw new NotFittedError("MinCovDet is not fitted yet.");
    const p = this.location_.length;
    return new Float64Array(
      X.map((xi) => {
        const diff = new Float64Array(p);
        for (let j = 0; j < p; j++)
          diff[j] = (xi[j] ?? 0) - (this.location_![j] ?? 0);
        let d = 0;
        for (let j = 0; j < p; j++)
          for (let k = 0; k < p; k++)
            d +=
              (diff[j] ?? 0) * (this.precision_![j]![k] ?? 0) * (diff[k] ?? 0);
        return d;
      }),
    );
  }
}
