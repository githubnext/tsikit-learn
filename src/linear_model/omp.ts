/**
 * Orthogonal Matching Pursuit (OMP) and OMP-CV.
 * Mirrors sklearn.linear_model.OrthogonalMatchingPursuit.
 */

import { NotFittedError } from "../exceptions.js";

/** Dot product of two Float64Arrays. */
function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

/** L2 norm of a Float64Array. */
function norm2(a: Float64Array): number {
  return Math.sqrt(dot(a, a));
}

/** Solve a small dense least-squares system A*x = b using Gram-Schmidt. */
function leastSquares(A: Float64Array[], b: Float64Array): Float64Array {
  const n = b.length;
  const k = A.length;
  // Use normal equations via Gram-Schmidt
  const Q: Float64Array[] = [];
  const R: Float64Array[] = [];

  for (let j = 0; j < k; j++) {
    const v = new Float64Array(A[j] ?? new Float64Array(n));
    const rj = new Float64Array(j + 1);
    for (let i = 0; i < j; i++) {
      const qi = Q[i] ?? new Float64Array(n);
      const d = dot(qi, v);
      rj[i] = d;
      for (let l = 0; l < n; l++) v[l] = (v[l] ?? 0) - d * (qi[l] ?? 0);
    }
    const nrm = norm2(v);
    rj[j] = nrm;
    R.push(rj);
    if (nrm > 1e-14) {
      const q = new Float64Array(n);
      for (let l = 0; l < n; l++) q[l] = (v[l] ?? 0) / nrm;
      Q.push(q);
    } else {
      Q.push(new Float64Array(n));
    }
  }

  // Back-substitution: x = R^{-1} Q^T b
  const Qtb = new Float64Array(k);
  for (let i = 0; i < k; i++) Qtb[i] = dot(Q[i] ?? new Float64Array(n), b);

  const x = new Float64Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let s = Qtb[i] ?? 0;
    const ri = R[i] ?? new Float64Array(0);
    for (let j = i + 1; j < k; j++) s -= (ri[j] ?? 0) * (x[j] ?? 0);
    const rii = ri[i] ?? 0;
    x[i] = rii !== 0 ? s / rii : 0;
  }
  return x;
}

export interface OMPOptions {
  nNonzeroCoefs?: number | null;
  tol?: number | null;
  fitIntercept?: boolean;
}

/**
 * Orthogonal Matching Pursuit regressor.
 * Greedily selects features that maximally reduce residual.
 * Mirrors sklearn.linear_model.OrthogonalMatchingPursuit.
 */
export class OrthogonalMatchingPursuit {
  nNonzeroCoefs: number | null;
  tol: number | null;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;
  nNonzeroCoefs_: number = 0;

  constructor(options: OMPOptions = {}) {
    this.nNonzeroCoefs = options.nNonzeroCoefs ?? null;
    this.tol = options.tol ?? null;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    let Xc = X;
    let yc = y;
    const xMeans = new Float64Array(p);
    let yMean = 0;

    if (this.fitIntercept) {
      for (const xi of X)
        for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0);
      for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) / n;
      for (let i = 0; i < n; i++) yMean += y[i] ?? 0;
      yMean /= n;
      Xc = X.map((xi) => {
        const r = new Float64Array(p);
        for (let j = 0; j < p; j++) r[j] = (xi[j] ?? 0) - (xMeans[j] ?? 0);
        return r;
      });
      yc = new Float64Array(n);
      for (let i = 0; i < n; i++) yc[i] = (y[i] ?? 0) - yMean;
    }

    const maxK = this.nNonzeroCoefs ?? Math.min(p, n);
    const tolSq = this.tol != null ? this.tol ** 2 : null;

    const residual = new Float64Array(yc);
    const supportSet: number[] = [];
    const coefFull = new Float64Array(p);

    for (let iter = 0; iter < maxK; iter++) {
      // Find feature with max |correlation|
      let bestJ = -1;
      let bestCorr = -1;
      for (let j = 0; j < p; j++) {
        if (supportSet.includes(j)) continue;
        const col = Xc.map((xi) => xi[j] ?? 0);
        const colF = new Float64Array(col);
        const nrm = norm2(colF);
        if (nrm < 1e-14) continue;
        const c = Math.abs(dot(colF, residual)) / nrm;
        if (c > bestCorr) {
          bestCorr = c;
          bestJ = j;
        }
      }
      if (bestJ === -1) break;
      supportSet.push(bestJ);

      // OLS on support set
      const subA = supportSet.map(
        (j) => new Float64Array(Xc.map((xi) => xi[j] ?? 0)),
      );
      // Transpose: subA[j][i] → need column matrix
      const subACols: Float64Array[] = [];
      for (const j of supportSet) {
        const col = new Float64Array(n);
        for (let i = 0; i < n; i++)
          col[i] = (Xc[i] ?? new Float64Array(0))[j] ?? 0;
        subACols.push(col);
      }
      const subCoef = leastSquares(subACols, yc);

      // Update residual
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let ki = 0; ki < supportSet.length; ki++) {
          pred +=
            ((Xc[i] ?? new Float64Array(0))[supportSet[ki] ?? 0] ?? 0) *
            (subCoef[ki] ?? 0);
        }
        residual[i] = (yc[i] ?? 0) - pred;
      }

      this.nIter_ = iter + 1;

      if (tolSq !== null) {
        const resSq = dot(residual, residual);
        if (resSq <= tolSq) break;
      }

      // Store latest coef
      for (let ki = 0; ki < supportSet.length; ki++) {
        coefFull[supportSet[ki] ?? 0] = subCoef[ki] ?? 0;
      }
    }

    this.coef_ = coefFull;
    this.nNonzeroCoefs_ = supportSet.length;

    if (this.fitIntercept) {
      this.intercept_ = yMean;
      for (let j = 0; j < p; j++)
        this.intercept_ -= (coefFull[j] ?? 0) * (xMeans[j] ?? 0);
    } else {
      this.intercept_ = 0;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_)
      throw new NotFittedError("OrthogonalMatchingPursuit is not fitted yet.");
    return new Float64Array(
      X.map((xi) => {
        let s = this.intercept_;
        for (let j = 0; j < xi.length; j++)
          s += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
        return s;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (let i = 0; i < y.length; i++) yMean += y[i] ?? 0;
    yMean /= y.length;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
