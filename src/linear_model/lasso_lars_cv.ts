/**
 * LassoLarsCV, LassoLarsIC — model selection via cross-validation and IC for Lasso-LARS path.
 * Mirrors sklearn.linear_model.LassoLarsCV and LassoLarsIC.
 */

import { NotFittedError } from "../exceptions.js";

export interface LassoLarsCVOptions {
  fitIntercept?: boolean;
  maxIter?: number;
  cv?: number;
  maxNAlphas?: number;
  eps?: number;
}

/**
 * Cross-validated LassoLars: selects the regularization parameter via k-fold CV.
 */
export class LassoLarsCV {
  private fitIntercept: boolean;
  private maxIter: number;
  private cv: number;
  private eps: number;

  coef_?: Float64Array;
  intercept_?: number;
  alpha_?: number;
  alphas_?: Float64Array;
  cvAlphas_?: Float64Array;
  msePathPerAlpha_?: Float64Array[];

  constructor(options: LassoLarsCVOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 500;
    this.cv = options.cv ?? 5;
    this.eps = options.eps ?? 2.220446049250313e-16;
  }

  private lassoPath(
    X: Float64Array[],
    y: Float64Array,
    alpha: number,
  ): Float64Array {
    // Coordinate descent for LASSO at a single alpha
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const coef = new Float64Array(p);
    let intercept = 0;

    const yMean = this.fitIntercept
      ? Array.from(y).reduce((a, b) => a + b, 0) / n
      : 0;
    const xMean = this.fitIntercept
      ? new Float64Array(p).map(
          (_, j) => Array.from(X).reduce((a, row) => a + (row[j] ?? 0), 0) / n,
        )
      : new Float64Array(p);

    const yC = new Float64Array(y.map((v, i) => v - yMean));
    const XC = X.map(
      (row) => new Float64Array(row.map((v, j) => v - (xMean[j] ?? 0))),
    );

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          let pred = 0;
          for (let k = 0; k < p; k++) {
            if (k !== j) pred += (coef[k] ?? 0) * (XC[i]![k] ?? 0);
          }
          rho += (XC[i]![j] ?? 0) * ((yC[i] ?? 0) - pred);
        }
        const xjNorm2 = XC.reduce((s, row) => s + (row[j] ?? 0) ** 2, 0);
        const prev = coef[j] ?? 0;
        if (xjNorm2 < this.eps) {
          coef[j] = 0;
        } else {
          const z = rho / xjNorm2;
          const thresh = (alpha * n) / xjNorm2;
          if (z > thresh) coef[j] = z - thresh;
          else if (z < -thresh) coef[j] = z + thresh;
          else coef[j] = 0;
        }
        maxDelta = Math.max(maxDelta, Math.abs((coef[j] ?? 0) - prev));
      }
      if (maxDelta < this.eps) break;
    }

    if (this.fitIntercept) {
      intercept = yMean;
      for (let j = 0; j < p; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
    }
    void intercept;
    return coef;
  }

  private mse(
    coef: Float64Array,
    intercept: number,
    X: Float64Array[],
    y: Float64Array,
  ): number {
    let err = 0;
    for (let i = 0; i < X.length; i++) {
      let pred = intercept;
      const xi = X[i]!;
      for (let j = 0; j < coef.length; j++)
        pred += (coef[j] ?? 0) * (xi[j] ?? 0);
      err += ((y[i] ?? 0) - pred) ** 2;
    }
    return err / X.length;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const alphas = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0];
    const foldSize = Math.floor(n / this.cv);
    const msePerAlpha = alphas.map(() => new Float64Array(this.cv));

    for (let fold = 0; fold < this.cv; fold++) {
      const start = fold * foldSize;
      const end = fold === this.cv - 1 ? n : (fold + 1) * foldSize;
      const valIdx = Array.from({ length: end - start }, (_, i) => start + i);
      const trainIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i < start || i >= end) trainIdx.push(i);
      }
      const xTrain = trainIdx.map((i) => X[i]!);
      const xVal = valIdx.map((i) => X[i]!);
      const yTrain = new Float64Array(trainIdx.map((i) => y[i] ?? 0));
      const yVal = new Float64Array(valIdx.map((i) => y[i] ?? 0));

      for (let ai = 0; ai < alphas.length; ai++) {
        const coef = this.lassoPath(xTrain, yTrain, alphas[ai]!);
        let intercept = 0;
        if (this.fitIntercept) {
          const yMean =
            Array.from(yTrain).reduce((a, b) => a + b, 0) / yTrain.length;
          const xMean = new Float64Array(coef.length).map(
            (_, j) =>
              Array.from(xTrain).reduce((a, row) => a + (row[j] ?? 0), 0) /
              xTrain.length,
          );
          intercept =
            yMean -
            Array.from(coef).reduce((a, c, j) => a + c * (xMean[j] ?? 0), 0);
        }
        msePerAlpha[ai]![fold] = this.mse(coef, intercept, xVal, yVal);
      }
    }

    // Pick alpha with lowest mean CV MSE
    let bestAlpha = alphas[0]!;
    let bestMSE = Number.POSITIVE_INFINITY;
    for (let ai = 0; ai < alphas.length; ai++) {
      const meanMse =
        Array.from(msePerAlpha[ai]!).reduce((a, b) => a + b, 0) / this.cv;
      if (meanMse < bestMSE) {
        bestMSE = meanMse;
        bestAlpha = alphas[ai]!;
      }
    }

    this.alpha_ = bestAlpha;
    this.alphas_ = new Float64Array(alphas);
    this.msePathPerAlpha_ = msePerAlpha;

    const bestCoef = this.lassoPath(X, y, bestAlpha);
    this.coef_ = bestCoef;
    if (this.fitIntercept) {
      const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
      const xMean = new Float64Array(bestCoef.length).map(
        (_, j) => Array.from(X).reduce((a, row) => a + (row[j] ?? 0), 0) / n,
      );
      this.intercept_ =
        yMean -
        Array.from(bestCoef).reduce((a, c, j) => a + c * (xMean[j] ?? 0), 0);
    } else {
      this.intercept_ = 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("LassoLarsCV is not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let pred = this.intercept_ ?? 0;
      const xi = X[i]!;
      for (let j = 0; j < this.coef_.length; j++)
        pred += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      out[i] = pred;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let ssTot = 0;
    let ssRes = 0;
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}

export type LassoLarsICCriterion = "aic" | "bic";

export interface LassoLarsICOptions {
  criterion?: LassoLarsICCriterion;
  fitIntercept?: boolean;
  maxIter?: number;
  eps?: number;
  noiseVariance?: number;
}

/**
 * LASSO model fit with Lars using information criterion (AIC or BIC).
 */
export class LassoLarsIC {
  private criterion: LassoLarsICCriterion;
  private fitIntercept: boolean;
  private maxIter: number;
  private eps: number;
  private noiseVariance: number | null;

  coef_?: Float64Array;
  intercept_?: number;
  alpha_?: number;
  criterion_?: Float64Array;
  alphas_?: Float64Array;

  constructor(options: LassoLarsICOptions = {}) {
    this.criterion = options.criterion ?? "aic";
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 500;
    this.eps = options.eps ?? 2.220446049250313e-16;
    this.noiseVariance = options.noiseVariance ?? null;
  }

  private coordDescent(
    X: Float64Array[],
    y: Float64Array,
    alpha: number,
  ): Float64Array {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const coef = new Float64Array(p);
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          let pred = 0;
          for (let k = 0; k < p; k++) {
            if (k !== j) pred += (coef[k] ?? 0) * (X[i]![k] ?? 0);
          }
          rho += (X[i]![j] ?? 0) * ((y[i] ?? 0) - pred);
        }
        const xjNorm2 = X.reduce((s, row) => s + (row[j] ?? 0) ** 2, 0);
        const prev = coef[j] ?? 0;
        if (xjNorm2 < this.eps) {
          coef[j] = 0;
          continue;
        }
        const z = rho / xjNorm2;
        const thresh = (alpha * n) / xjNorm2;
        if (z > thresh) coef[j] = z - thresh;
        else if (z < -thresh) coef[j] = z + thresh;
        else coef[j] = 0;
        maxDelta = Math.max(maxDelta, Math.abs((coef[j] ?? 0) - prev));
      }
      if (maxDelta < this.eps) break;
    }
    return coef;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const alphas = [
      1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0,
    ];

    let bestAlpha = alphas[0]!;
    let bestIC = Number.POSITIVE_INFINITY;
    const icValues = new Float64Array(alphas.length);

    const noiseVar =
      this.noiseVariance ??
      (() => {
        const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
        return Array.from(y).reduce((a, v) => a + (v - yMean) ** 2, 0) / n;
      })();

    for (let ai = 0; ai < alphas.length; ai++) {
      const coef = this.coordDescent(X, y, alphas[ai]!);
      let intercept = 0;
      if (this.fitIntercept) {
        const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
        const xMean = new Float64Array(coef.length).map(
          (_, j) => Array.from(X).reduce((a, row) => a + (row[j] ?? 0), 0) / n,
        );
        intercept =
          yMean -
          Array.from(coef).reduce((a, c, j) => a + c * (xMean[j] ?? 0), 0);
      }
      // Residual sum of squares
      let rss = 0;
      for (let i = 0; i < n; i++) {
        let pred = intercept;
        for (let j = 0; j < coef.length; j++)
          pred += (coef[j] ?? 0) * (X[i]![j] ?? 0);
        rss += ((y[i] ?? 0) - pred) ** 2;
      }
      const df = Array.from(coef).filter((c) => Math.abs(c) > this.eps).length;
      const k = this.criterion === "bic" ? Math.log(n) : 2;
      const ic = n * Math.log(rss / n + noiseVar * 1e-6) + k * df;
      icValues[ai] = ic;
      if (ic < bestIC) {
        bestIC = ic;
        bestAlpha = alphas[ai]!;
      }
    }

    this.alpha_ = bestAlpha;
    this.alphas_ = new Float64Array(alphas);
    this.criterion_ = icValues;

    const bestCoef = this.coordDescent(X, y, bestAlpha);
    this.coef_ = bestCoef;
    if (this.fitIntercept) {
      const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
      const xMean = new Float64Array(bestCoef.length).map(
        (_, j) => Array.from(X).reduce((a, row) => a + (row[j] ?? 0), 0) / n,
      );
      this.intercept_ =
        yMean -
        Array.from(bestCoef).reduce((a, c, j) => a + c * (xMean[j] ?? 0), 0);
    } else {
      this.intercept_ = 0;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("LassoLarsIC is not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let pred = this.intercept_ ?? 0;
      const xi = X[i]!;
      for (let j = 0; j < this.coef_.length; j++)
        pred += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      out[i] = pred;
    }
    return out;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const n = y.length;
    let ssTot = 0;
    let ssRes = 0;
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < n; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
  }
}
