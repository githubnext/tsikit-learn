/**
 * Cross-validated linear model selectors: RidgeCV, LassoCV, ElasticNetCV.
 * Mirrors sklearn.linear_model.RidgeCV, LassoCV, ElasticNetCV.
 */

import { NotFittedError } from "../exceptions.js";
import { KFold } from "../model_selection/split.js";

/** Mean of an array. */
function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/** R² score. */
function r2Score(y: Float64Array, yPred: Float64Array): number {
  const yMean = mean(Array.from(y));
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    ssTot += ((y[i] ?? 0) - yMean) ** 2;
  }
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

/** MSE. */
function mse(y: Float64Array, yPred: Float64Array): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
  return s / y.length;
}

/** Solve Ridge regression (OLS + L2): (X^T X + alpha I) w = X^T y. */
function solveRidge(X: Float64Array[], y: Float64Array, alpha: number, fitIntercept: boolean): { w: Float64Array; intercept: number } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;

  let Xuse = X;
  let yMean = 0;
  const xMeans = new Float64Array(p);

  if (fitIntercept) {
    yMean = mean(Array.from(y));
    for (const xi of X) for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0) / n;
    Xuse = X.map((xi) => new Float64Array(xi.map((v, j) => v - (xMeans[j] ?? 0))));
  }

  const yc = new Float64Array(y.map((v) => v - yMean));

  // Build X^T X + alpha I (p x p)
  const A = Array.from({ length: p }, (_, i) => {
    const row = new Float64Array(p);
    row[i] = alpha;
    return row;
  });
  const b = new Float64Array(p);

  for (let i = 0; i < n; i++) {
    const xi = Xuse[i] ?? new Float64Array(p);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) A[j]![k] = (A[j]![k] ?? 0) + (xi[j] ?? 0) * (xi[k] ?? 0);
      b[j] = (b[j] ?? 0) + (xi[j] ?? 0) * (yc[i] ?? 0);
    }
  }

  // Gauss-Jordan solve
  const Ab = A.map((row, i) => { const r = new Float64Array(p + 1); for (let j = 0; j < p; j++) r[j] = row[j] ?? 0; r[p] = b[i] ?? 0; return r; });
  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let row = col + 1; row < p; row++) if (Math.abs(Ab[row]![col] ?? 0) > Math.abs(Ab[pivot]![col] ?? 0)) pivot = row;
    const tmp = Ab[col]!; Ab[col] = Ab[pivot]!; Ab[pivot] = tmp;
    const scale = Ab[col]![col] ?? 1;
    if (Math.abs(scale) < 1e-14) continue;
    for (let j = col; j <= p; j++) Ab[col]![j] = (Ab[col]![j] ?? 0) / scale;
    for (let row = 0; row < p; row++) {
      if (row === col) continue;
      const f = Ab[row]![col] ?? 0;
      for (let j = col; j <= p; j++) Ab[row]![j] = (Ab[row]![j] ?? 0) - f * (Ab[col]![j] ?? 0);
    }
  }
  const w = new Float64Array(p);
  for (let j = 0; j < p; j++) w[j] = Ab[j]![p] ?? 0;

  let intercept = yMean;
  if (fitIntercept) {
    for (let j = 0; j < p; j++) intercept -= (w[j] ?? 0) * (xMeans[j] ?? 0);
  }

  return { w, intercept };
}

function predictLinear(X: Float64Array[], w: Float64Array, intercept: number): Float64Array {
  return new Float64Array(X.map((xi) => {
    let pred = intercept;
    for (let j = 0; j < xi.length; j++) pred += (w[j] ?? 0) * (xi[j] ?? 0);
    return pred;
  }));
}

/**
 * Ridge regression with built-in cross-validation.
 * Mirrors sklearn.linear_model.RidgeCV.
 */
export class RidgeCV {
  alphas: number[];
  fitIntercept: boolean;
  cv: number;

  alpha_: number = 1.0;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  bestScore_: number = -Number.POSITIVE_INFINITY;

  constructor(
    options: {
      alphas?: number[];
      fitIntercept?: boolean;
      cv?: number;
    } = {},
  ) {
    this.alphas = options.alphas ?? [0.1, 1.0, 10.0];
    this.fitIntercept = options.fitIntercept ?? true;
    this.cv = options.cv ?? 5;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const kf = new KFold({ nSplits: Math.min(this.cv, X.length) });
    let bestAlpha = this.alphas[0] ?? 1.0;
    let bestScore = -Number.POSITIVE_INFINITY;

    for (const alpha of this.alphas) {
      const scores: number[] = [];
      for (const fold of kf.split(X)) {
        const Xtrain = Array.from(fold.trainIndex).map((i) => X[i] ?? new Float64Array(0));
        const ytrain = new Float64Array(Array.from(fold.trainIndex).map((i) => y[i] ?? 0));
        const Xval = Array.from(fold.testIndex).map((i) => X[i] ?? new Float64Array(0));
        const yval = new Float64Array(Array.from(fold.testIndex).map((i) => y[i] ?? 0));
        const { w, intercept } = solveRidge(Xtrain, ytrain, alpha, this.fitIntercept);
        const yPred = predictLinear(Xval, w, intercept);
        scores.push(r2Score(yval, yPred));
      }
      const s = mean(scores);
      if (s > bestScore) { bestScore = s; bestAlpha = alpha; }
    }

    this.alpha_ = bestAlpha;
    this.bestScore_ = bestScore;
    const { w, intercept } = solveRidge(X, y, bestAlpha, this.fitIntercept);
    this.coef_ = w;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("RidgeCV");
    return predictLinear(X, this.coef_, this.intercept_);
  }

  score(X: Float64Array[], y: Float64Array): number {
    return r2Score(y, this.predict(X));
  }
}

/** Coordinate-descent Lasso for a single alpha. Returns coef. */
function lassoCD(X: Float64Array[], y: Float64Array, alpha: number, maxIter: number, tol: number): Float64Array {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const w = new Float64Array(p);
  for (let iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      let rho = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(p);
        let pred = 0;
        for (let k = 0; k < p; k++) if (k !== j) pred += (w[k] ?? 0) * (xi[k] ?? 0);
        rho += (xi[j] ?? 0) * ((y[i] ?? 0) - pred);
      }
      rho /= n;
      const normSq = Array.from(X).reduce((s, xi) => s + (xi[j] ?? 0) ** 2, 0) / n;
      const wOld = w[j] ?? 0;
      const r = rho;
      w[j] = normSq > 0 ? (r > alpha ? (r - alpha) / normSq : r < -alpha ? (r + alpha) / normSq : 0) : 0;
      maxDelta = Math.max(maxDelta, Math.abs((w[j] ?? 0) - wOld));
    }
    if (maxDelta < tol) break;
  }
  return w;
}

/**
 * Lasso with built-in cross-validation to find optimal alpha.
 * Mirrors sklearn.linear_model.LassoCV.
 */
export class LassoCV {
  eps: number;
  nAlphas: number;
  alphas: number[] | null;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  cv: number;

  alpha_: number = 1.0;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  msePathMin_: number = Number.POSITIVE_INFINITY;

  constructor(
    options: {
      eps?: number;
      nAlphas?: number;
      alphas?: number[] | null;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
      cv?: number;
    } = {},
  ) {
    this.eps = options.eps ?? 1e-3;
    this.nAlphas = options.nAlphas ?? 100;
    this.alphas = options.alphas ?? null;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
    this.cv = options.cv ?? 5;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    // Center data
    const yMean = this.fitIntercept ? mean(Array.from(y)) : 0;
    const xMeans = new Float64Array(p);
    if (this.fitIntercept) {
      for (const xi of X) for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0) / n;
    }
    const Xc = X.map((xi) => new Float64Array(xi.map((v, j) => v - (xMeans[j] ?? 0))));
    const yc = new Float64Array(y.map((v) => v - yMean));

    // Compute alpha_max
    let alphaMax = 0;
    for (let j = 0; j < p; j++) {
      let corr = 0;
      for (let i = 0; i < n; i++) corr += ((Xc[i] ?? new Float64Array(p))[j] ?? 0) * (yc[i] ?? 0);
      alphaMax = Math.max(alphaMax, Math.abs(corr / n));
    }

    const alphas = this.alphas ?? Array.from({ length: this.nAlphas }, (_, i) => {
      const t = i / (this.nAlphas - 1);
      return alphaMax * Math.pow(this.eps, t);
    });

    const kf = new KFold({ nSplits: Math.min(this.cv, n) });
    let bestAlpha = alphas[0] ?? 1.0;
    let bestMse = Number.POSITIVE_INFINITY;

    for (const alpha of alphas) {
      const mses: number[] = [];
      for (const fold of kf.split(Xc)) {
        const Xtrain = Array.from(fold.trainIndex).map((i) => Xc[i] ?? new Float64Array(p));
        const ytrain = new Float64Array(Array.from(fold.trainIndex).map((i) => yc[i] ?? 0));
        const Xval = Array.from(fold.testIndex).map((i) => Xc[i] ?? new Float64Array(p));
        const yval = new Float64Array(Array.from(fold.testIndex).map((i) => yc[i] ?? 0));
        const w = lassoCD(Xtrain, ytrain, alpha, this.maxIter, this.tol);
        const yPred = predictLinear(Xval, w, 0);
        mses.push(mse(yval, yPred));
      }
      const avgMse = mean(mses);
      if (avgMse < bestMse) { bestMse = avgMse; bestAlpha = alpha; }
    }

    this.alpha_ = bestAlpha;
    this.msePathMin_ = bestMse;
    const w = lassoCD(Xc, yc, bestAlpha, this.maxIter, this.tol);
    this.coef_ = w;
    let intercept = yMean;
    if (this.fitIntercept) for (let j = 0; j < p; j++) intercept -= (w[j] ?? 0) * (xMeans[j] ?? 0);
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("LassoCV");
    return predictLinear(X, this.coef_, this.intercept_);
  }

  score(X: Float64Array[], y: Float64Array): number {
    return r2Score(y, this.predict(X));
  }
}

/**
 * ElasticNet with built-in cross-validation.
 * Mirrors sklearn.linear_model.ElasticNetCV.
 */
export class ElasticNetCV {
  l1Ratio: number | number[];
  eps: number;
  nAlphas: number;
  alphas: number[] | null;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  cv: number;

  alpha_: number = 1.0;
  l1Ratio_: number = 0.5;
  coef_: Float64Array | null = null;
  intercept_: number = 0;

  constructor(
    options: {
      l1Ratio?: number | number[];
      eps?: number;
      nAlphas?: number;
      alphas?: number[] | null;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
      cv?: number;
    } = {},
  ) {
    this.l1Ratio = options.l1Ratio ?? 0.5;
    this.eps = options.eps ?? 1e-3;
    this.nAlphas = options.nAlphas ?? 100;
    this.alphas = options.alphas ?? null;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
    this.cv = options.cv ?? 5;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const ratios = Array.isArray(this.l1Ratio) ? this.l1Ratio : [this.l1Ratio];

    const yMean = this.fitIntercept ? mean(Array.from(y)) : 0;
    const xMeans = new Float64Array(p);
    if (this.fitIntercept) for (const xi of X) for (let j = 0; j < p; j++) xMeans[j] = (xMeans[j] ?? 0) + (xi[j] ?? 0) / n;
    const Xc = X.map((xi) => new Float64Array(xi.map((v, j) => v - (xMeans[j] ?? 0))));
    const yc = new Float64Array(y.map((v) => v - yMean));

    let alphaMax = 0;
    for (let j = 0; j < p; j++) {
      let corr = 0;
      for (let i = 0; i < n; i++) corr += ((Xc[i] ?? new Float64Array(p))[j] ?? 0) * (yc[i] ?? 0);
      alphaMax = Math.max(alphaMax, Math.abs(corr / n));
    }

    const alphas = this.alphas ?? Array.from({ length: this.nAlphas }, (_, i) => {
      const t = i / (this.nAlphas - 1);
      return alphaMax * Math.pow(this.eps, t);
    });

    const kf = new KFold({ nSplits: Math.min(this.cv, n) });
    let bestAlpha = alphas[0] ?? 1.0;
    let bestRatio = ratios[0] ?? 0.5;
    let bestMse = Number.POSITIVE_INFINITY;

    for (const ratio of ratios) {
      for (const alpha of alphas) {
        const l1 = alpha * ratio;
        const l2 = alpha * (1 - ratio);
        const mses: number[] = [];
        for (const fold of kf.split(Xc)) {
          const Xtrain = Array.from(fold.trainIndex).map((i) => Xc[i] ?? new Float64Array(p));
          const ytrain = new Float64Array(Array.from(fold.trainIndex).map((i) => yc[i] ?? 0));
          const Xval = Array.from(fold.testIndex).map((i) => Xc[i] ?? new Float64Array(p));
          const yval = new Float64Array(Array.from(fold.testIndex).map((i) => yc[i] ?? 0));
          // Elastic net CD
          const w = new Float64Array(p);
          for (let iter = 0; iter < this.maxIter; iter++) {
            let maxDelta = 0;
            for (let j = 0; j < p; j++) {
              let rho = 0;
              for (let ii = 0; ii < Xtrain.length; ii++) {
                const xi = Xtrain[ii] ?? new Float64Array(p);
                let pred = 0;
                for (let k = 0; k < p; k++) if (k !== j) pred += (w[k] ?? 0) * (xi[k] ?? 0);
                rho += (xi[j] ?? 0) * ((ytrain[ii] ?? 0) - pred);
              }
              rho /= Xtrain.length;
              const normSq = Xtrain.reduce((s, xi) => s + (xi[j] ?? 0) ** 2, 0) / Xtrain.length + l2;
              const wOld = w[j] ?? 0;
              w[j] = normSq > 0 ? (rho > l1 ? (rho - l1) / normSq : rho < -l1 ? (rho + l1) / normSq : 0) : 0;
              maxDelta = Math.max(maxDelta, Math.abs((w[j] ?? 0) - wOld));
            }
            if (maxDelta < this.tol) break;
          }
          const yPred = predictLinear(Xval, w, 0);
          mses.push(mse(yval, yPred));
        }
        const avgMse = mean(mses);
        if (avgMse < bestMse) { bestMse = avgMse; bestAlpha = alpha; bestRatio = ratio; }
      }
    }

    this.alpha_ = bestAlpha;
    this.l1Ratio_ = bestRatio;
    const l1 = bestAlpha * bestRatio;
    const l2 = bestAlpha * (1 - bestRatio);
    const w = new Float64Array(p);
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          const xi = Xc[i] ?? new Float64Array(p);
          let pred = 0;
          for (let k = 0; k < p; k++) if (k !== j) pred += (w[k] ?? 0) * (xi[k] ?? 0);
          rho += (xi[j] ?? 0) * ((yc[i] ?? 0) - pred);
        }
        rho /= n;
        const normSq = Xc.reduce((s, xi) => s + (xi[j] ?? 0) ** 2, 0) / n + l2;
        const wOld = w[j] ?? 0;
        w[j] = normSq > 0 ? (rho > l1 ? (rho - l1) / normSq : rho < -l1 ? (rho + l1) / normSq : 0) : 0;
        maxDelta = Math.max(maxDelta, Math.abs((w[j] ?? 0) - wOld));
      }
      if (maxDelta < this.tol) break;
    }
    this.coef_ = w;
    let intercept = yMean;
    if (this.fitIntercept) for (let j = 0; j < p; j++) intercept -= (w[j] ?? 0) * (xMeans[j] ?? 0);
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coef_ === null) throw new NotFittedError("ElasticNetCV");
    return predictLinear(X, this.coef_, this.intercept_);
  }

  score(X: Float64Array[], y: Float64Array): number {
    return r2Score(y, this.predict(X));
  }
}
