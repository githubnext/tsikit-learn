/**
 * MultiTaskLassoCV and MultiTaskElasticNetCV: cross-validated multi-task regularization.
 * Mirrors sklearn.linear_model.MultiTaskLassoCV and MultiTaskElasticNetCV.
 */

import { BaseEstimator } from "../base.js";
import { NotFittedError } from "../exceptions.js";

export interface MultiTaskLassoCVOptions {
  eps?: number;
  nAlphas?: number;
  alphas?: Float64Array;
  fitIntercept?: boolean;
  maxIter?: number;
  tol?: number;
  cv?: number;
}

export interface MultiTaskElasticNetCVOptions extends MultiTaskLassoCVOptions {
  l1Ratio?: number | number[];
}

function softThresholdVec(v: Float64Array, threshold: number): Float64Array {
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const vi = v[i] ?? 0;
    const norm = Math.abs(vi);
    out[i] = norm <= threshold ? 0 : vi * (1 - threshold / norm);
  }
  return out;
}

function blockCoordinateDescent(
  X: Float64Array[],
  Y: Float64Array[],
  alpha: number,
  l1Ratio: number,
  maxIter: number,
  tol: number,
): Float64Array[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const q = Y[0]?.length ?? 0;

  // W: p x q coefficient matrix (stored as rows = features)
  const W = Array.from({ length: p }, () => new Float64Array(q));
  const residuals = Y.map((y) => new Float64Array(y));

  for (let iter = 0; iter < maxIter; iter++) {
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      // Partial residual for feature j
      const rj = new Float64Array(q);
      for (let t = 0; t < n; t++) {
        const xjt = X[t]![j] ?? 0;
        for (let k = 0; k < q; k++) {
          rj[k] = (rj[k] ?? 0) + xjt * (residuals[t]![k] ?? 0);
        }
      }
      // Add back current contribution
      const wj = W[j]!;
      let normXj = 0;
      for (let t = 0; t < n; t++) normXj += (X[t]![j] ?? 0) ** 2;
      if (normXj === 0) continue;

      const candidate = new Float64Array(q);
      for (let k = 0; k < q; k++) {
        candidate[k] = (rj[k] ?? 0) / normXj + (wj[k] ?? 0);
      }

      // L1/L2 regularization
      const l1 = alpha * l1Ratio / normXj * n;
      const l2 = alpha * (1 - l1Ratio) / normXj * n;
      const newWj = softThresholdVec(candidate, l1);
      const norm2 = Math.sqrt(newWj.reduce((s, v) => s + v ** 2, 0));
      const scale = norm2 > 0 ? Math.max(0, 1 - l2 / norm2) : 0;
      for (let k = 0; k < q; k++) newWj[k] = (newWj[k] ?? 0) * scale;

      // Update residuals
      const delta = new Float64Array(q);
      for (let k = 0; k < q; k++) delta[k] = (newWj[k] ?? 0) - (wj[k] ?? 0);
      for (let t = 0; t < n; t++) {
        const xjt = X[t]![j] ?? 0;
        for (let k = 0; k < q; k++) {
          residuals[t]![k] = (residuals[t]![k] ?? 0) - xjt * (delta[k] ?? 0);
        }
      }

      let change = 0;
      for (let k = 0; k < q; k++) change += (delta[k] ?? 0) ** 2;
      maxChange = Math.max(maxChange, Math.sqrt(change));
      W[j]! = newWj as Float64Array<ArrayBuffer>;
    }
    if (maxChange < tol) break;
  }
  return W;
}

function cvScore(
  X: Float64Array[],
  Y: Float64Array[],
  alpha: number,
  l1Ratio: number,
  cv: number,
  maxIter: number,
  tol: number,
): number {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  let total = 0;
  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize;
    const end = fold === cv - 1 ? n : start + foldSize;
    const trainX = X.filter((_, i) => i < start || i >= end);
    const trainY = Y.filter((_, i) => i < start || i >= end);
    const testX = X.slice(start, end);
    const testY = Y.slice(start, end);
    const W = blockCoordinateDescent(trainX, trainY, alpha, l1Ratio, maxIter, tol);
    const q = Y[0]?.length ?? 0;
    let ss_res = 0;
    for (let i = 0; i < testX.length; i++) {
      for (let k = 0; k < q; k++) {
        let pred = 0;
        for (let j = 0; j < (testX[0]?.length ?? 0); j++) {
          pred += (testX[i]![j] ?? 0) * (W[j]![k] ?? 0);
        }
        ss_res += ((testY[i]![k] ?? 0) - pred) ** 2;
      }
    }
    total += ss_res;
  }
  return -total; // higher is better
}

export class MultiTaskLassoCV extends BaseEstimator {
  eps: number;
  nAlphas: number;
  alphas: Float64Array | null;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  cv: number;

  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  alpha_: number | null = null;
  alphasPath_: Float64Array | null = null;
  msePathCV_: Float64Array | null = null;

  constructor(opts: MultiTaskLassoCVOptions = {}) {
    super();
    this.eps = opts.eps ?? 1e-3;
    this.nAlphas = opts.nAlphas ?? 100;
    this.alphas = opts.alphas ?? null;
    this.fitIntercept = opts.fitIntercept ?? true;
    this.maxIter = opts.maxIter ?? 1000;
    this.tol = opts.tol ?? 1e-4;
    this.cv = opts.cv ?? 5;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    let Xfit = X;
    let interceptMeans: Float64Array | null = null;

    if (this.fitIntercept) {
      const p = Y[0]?.length ?? 0;
      interceptMeans = new Float64Array(p);
      for (const y of Y) for (let k = 0; k < p; k++) interceptMeans[k] = (interceptMeans[k] ?? 0) + (y[k] ?? 0);
      for (let k = 0; k < (interceptMeans.length); k++) interceptMeans[k] = (interceptMeans[k] ?? 0) / n;
      const Yc = Y.map((y) => {
        const out = new Float64Array(y);
        for (let k = 0; k < out.length; k++) out[k] = (out[k] ?? 0) - (interceptMeans![k] ?? 0);
        return out;
      });
      Y = Yc;
    }

    // Generate alpha path
    const alphas = this.alphas ?? this._alphaGrid(Xfit, Y);
    this.alphasPath_ = alphas;

    // CV over alphas
    let bestScore = -Number.POSITIVE_INFINITY;
    let bestAlpha = alphas[0] ?? 1;
    const scores = new Float64Array(alphas.length);
    for (let ai = 0; ai < alphas.length; ai++) {
      const score = cvScore(Xfit, Y, alphas[ai] ?? 1, 1, this.cv, this.maxIter, this.tol);
      scores[ai] = score;
      if (score > bestScore) { bestScore = score; bestAlpha = alphas[ai] ?? 1; }
    }
    this.msePathCV_ = scores;
    this.alpha_ = bestAlpha;

    // Refit on full data
    this.coef_ = blockCoordinateDescent(Xfit, Y, bestAlpha, 1, this.maxIter, this.tol);
    this.intercept_ = interceptMeans ?? new Float64Array(Y[0]?.length ?? 0);
    return this;
  }

  protected _alphaGrid(X: Float64Array[], Y: Float64Array[]): Float64Array {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const q = Y[0]?.length ?? 0;
    let maxCorr = 0;
    for (let j = 0; j < p; j++) {
      let corrNorm = 0;
      for (let k = 0; k < q; k++) {
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (X[i]![j] ?? 0) * (Y[i]![k] ?? 0);
        corrNorm += corr ** 2;
      }
      maxCorr = Math.max(maxCorr, Math.sqrt(corrNorm));
    }
    const alphaMax = maxCorr / n;
    const alphaMin = alphaMax * this.eps;
    const alphas = new Float64Array(this.nAlphas);
    for (let i = 0; i < this.nAlphas; i++) {
      alphas[i] = alphaMax * Math.exp((Math.log(alphaMin / alphaMax) * i) / (this.nAlphas - 1));
    }
    return alphas;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.coef_) throw new NotFittedError("MultiTaskLassoCV");
    const W = this.coef_;
    const q = this.intercept_?.length ?? 0;
    return X.map((row) => {
      const pred = new Float64Array(q);
      for (let k = 0; k < q; k++) pred[k] = this.intercept_![k] ?? 0;
      for (let j = 0; j < W.length; j++) {
        for (let k = 0; k < q; k++) pred[k] = (pred[k] ?? 0) + (row[j] ?? 0) * (W[j]![k] ?? 0);
      }
      return pred;
    });
  }
}

export class MultiTaskElasticNetCV extends MultiTaskLassoCV {
  l1Ratio: number | number[];

  constructor(opts: MultiTaskElasticNetCVOptions = {}) {
    super(opts);
    this.l1Ratio = opts.l1Ratio ?? 0.5;
  }

  override fit(X: Float64Array[], Y: Float64Array[]): this {
    const l1Ratios = Array.isArray(this.l1Ratio) ? this.l1Ratio : [this.l1Ratio];
    const n = X.length;
    const alphas = this.alphas ?? this._alphaGridPublic(X, Y);
    this.alphasPath_ = alphas;

    let bestScore = -Number.POSITIVE_INFINITY;
    let bestAlpha = alphas[0] ?? 1;
    let bestL1 = l1Ratios[0] ?? 0.5;

    for (const l1 of l1Ratios) {
      for (let ai = 0; ai < alphas.length; ai++) {
        const score = cvScore(X, Y, alphas[ai] ?? 1, l1, this.cv, this.maxIter, this.tol);
        if (score > bestScore) { bestScore = score; bestAlpha = alphas[ai] ?? 1; bestL1 = l1; }
      }
    }

    this.alpha_ = bestAlpha;
    this.coef_ = blockCoordinateDescent(X, Y, bestAlpha, bestL1, this.maxIter, this.tol);
    this.intercept_ = new Float64Array(Y[0]?.length ?? 0);
    return this;
  }

  private _alphaGridPublic(X: Float64Array[], Y: Float64Array[]): Float64Array {
    return this._alphaGrid(X, Y);
  }
}
