/**
 * EllipticEnvelope: outlier detection via robust covariance estimation.
 * Mirrors sklearn.covariance.EllipticEnvelope.
 */

import { NotFittedError } from "../exceptions.js";

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

/** Compute log-determinant of a positive-definite matrix via Cholesky. */
function logDet(M: Float64Array[]): number {
  const p = M.length;
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i]![j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      if (i === j) {
        L[i]![j] = Math.sqrt(Math.max(s, 1e-12));
      } else {
        L[i]![j] = s / Math.max(L[j]![j] ?? 1e-12, 1e-12);
      }
    }
  }
  let logd = 0;
  for (let i = 0; i < p; i++)
    logd += Math.log(Math.max(L[i]![i] ?? 1e-12, 1e-12));
  return 2 * logd;
}

/** Invert a matrix via Gauss-Jordan. Returns null if singular. */
function invertMatrix(M: Float64Array[]): Float64Array[] | null {
  const p = M.length;
  const A = M.map((row) => new Float64Array(row));
  const I = Array.from({ length: p }, (_, i) => {
    const r = new Float64Array(p);
    r[i] = 1;
    return r;
  });
  for (let col = 0; col < p; col++) {
    let pivotRow = -1;
    let pivotVal = 0;
    for (let row = col; row < p; row++) {
      if (Math.abs(A[row]![col] ?? 0) > Math.abs(pivotVal)) {
        pivotVal = A[row]![col] ?? 0;
        pivotRow = row;
      }
    }
    if (pivotRow === -1 || Math.abs(pivotVal) < 1e-12) return null;
    const tmpA = A[col]!;
    A[col] = A[pivotRow]!;
    A[pivotRow] = tmpA;
    const tmpI = I[col]!;
    I[col] = I[pivotRow]!;
    I[pivotRow] = tmpI;
    const scale = A[col]![col] ?? 1;
    for (let j = 0; j < p; j++) {
      A[col]![j] = (A[col]![j] ?? 0) / scale;
      I[col]![j] = (I[col]![j] ?? 0) / scale;
    }
    for (let row = 0; row < p; row++) {
      if (row === col) continue;
      const factor = A[row]![col] ?? 0;
      for (let j = 0; j < p; j++) {
        A[row]![j] = (A[row]![j] ?? 0) - factor * (A[col]![j] ?? 0);
        I[row]![j] = (I[row]![j] ?? 0) - factor * (I[col]![j] ?? 0);
      }
    }
  }
  return I;
}

/** Mahalanobis distance squared for each row. */
function mahalanobisDistSq(
  X: Float64Array[],
  mean: Float64Array,
  precisionMat: Float64Array[],
): Float64Array {
  const n = X.length;
  const p = mean.length;
  const dists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(p);
    let d = 0;
    for (let j = 0; j < p; j++) {
      let row = 0;
      for (let k = 0; k < p; k++) {
        row += (precisionMat[j]![k] ?? 0) * ((xi[k] ?? 0) - (mean[k] ?? 0));
      }
      d += ((xi[j] ?? 0) - (mean[j] ?? 0)) * row;
    }
    dists[i] = d;
  }
  return dists;
}

/**
 * EllipticEnvelope: fits a robust covariance estimate to detect outliers.
 * Uses minimum covariance determinant (fast approximation).
 * Mirrors sklearn.covariance.EllipticEnvelope.
 */
export class EllipticEnvelope {
  contamination: number;
  supportFraction: number | null;
  randomState: number;

  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  threshold_: number = 0;
  offset_: number = 0;

  constructor(
    options: {
      contamination?: number;
      supportFraction?: number | null;
      randomState?: number;
    } = {},
  ) {
    this.contamination = options.contamination ?? 0.1;
    this.supportFraction = options.supportFraction ?? null;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const h =
      this.supportFraction !== null
        ? Math.floor(this.supportFraction * n)
        : Math.floor((n + p + 1) / 2);

    // Fast MCD approximation: random subsample + C-step iterations
    let bestDet = Number.POSITIVE_INFINITY;
    let bestMean = new Float64Array(p);
    let bestCov: Float64Array[] = Array.from(
      { length: p },
      () => new Float64Array(p),
    );

    const rng = this.randomState;
    const nTrials = 10;
    for (let trial = 0; trial < nTrials; trial++) {
      // Random subset of h points
      const indices = Array.from({ length: n }, (_, i) => i);
      // Pseudo-random shuffle using simple LCG
      for (let i = n - 1; i > 0; i--) {
        const j = Math.abs(
          (rng * 1664525 + 1013904223 + i * trial * 31337) % (i + 1),
        );
        const tmp = indices[i]!;
        indices[i] = indices[j]!;
        indices[j] = tmp;
      }
      const subset = indices
        .slice(0, h)
        .map((i) => X[i] ?? new Float64Array(p));

      // C-step iterations
      let curSubset = subset;
      for (let cstep = 0; cstep < 30; cstep++) {
        const mean = colMeans(curSubset);
        const cov = empCov(curSubset, mean);
        const inv = invertMatrix(cov);
        if (!inv) break;
        const dists = mahalanobisDistSq(X, mean, inv);
        const sortedIdx = Array.from({ length: n }, (_, i) => i).sort(
          (a, b) => (dists[a] ?? 0) - (dists[b] ?? 0),
        );
        curSubset = sortedIdx
          .slice(0, h)
          .map((i) => X[i] ?? new Float64Array(p));
      }

      const mean = colMeans(curSubset);
      const cov = empCov(curSubset, mean);
      const det = logDet(cov);
      if (det < bestDet) {
        bestDet = det;
        bestMean = mean;
        bestCov = cov;
      }
    }

    const inv = invertMatrix(bestCov) ?? bestCov;
    this.location_ = bestMean;
    this.covariance_ = bestCov;
    this.precision_ = inv;

    // Compute threshold based on contamination
    const dists = mahalanobisDistSq(X, bestMean, inv);
    const sorted = Array.from(dists).sort((a, b) => a - b);
    const threshIdx = Math.floor((1 - this.contamination) * n);
    this.threshold_ = sorted[Math.min(threshIdx, n - 1)] ?? 0;
    this.offset_ = -this.threshold_;
    return this;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    if (this.location_ === null || this.precision_ === null) {
      throw new NotFittedError("EllipticEnvelope");
    }
    return mahalanobisDistSq(X, this.location_, this.precision_);
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    const dists = this.mahalanobis(X);
    return new Float64Array(dists.map((d) => -d - this.offset_));
  }

  predict(X: Float64Array[]): Int32Array {
    const scores = this.decisionFunction(X);
    return new Int32Array(scores.map((s) => (s >= 0 ? 1 : -1)));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if ((yPred[i] ?? 0) === (y[i] ?? 0)) correct++;
    }
    return correct / y.length;
  }
}
