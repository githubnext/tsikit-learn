/**
 * HuberRegressor and Lars (Least Angle Regression).
 * Mirrors sklearn.linear_model.HuberRegressor and Lars.
 */

import { NotFittedError } from "../exceptions.js";

export interface HuberRegressorOptions {
  epsilon?: number;
  maxIter?: number;
  alpha?: number;
  tol?: number;
  fitIntercept?: boolean;
}

export class HuberRegressor {
  epsilon: number;
  maxIter: number;
  alpha: number;
  tol: number;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  outliers_: Uint8Array | null = null;
  nIter_: number = 0;

  constructor(options: HuberRegressorOptions = {}) {
    this.epsilon = options.epsilon ?? 1.35;
    this.maxIter = options.maxIter ?? 100;
    this.alpha = options.alpha ?? 0.0001;
    this.tol = options.tol ?? 1e-5;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    let w = new Float64Array(p);
    let b = this.fitIntercept ? 0 : 0;
    const lr = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxGrad = 0;
      const gradW = new Float64Array(p);
      let gradB = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i]!;
        const yi = y[i] ?? 0;
        let pred = b;
        for (let j = 0; j < p; j++) pred += (w[j] ?? 0) * (xi[j] ?? 0);

        const r = yi - pred;
        const absR = Math.abs(r);

        let huberGrad: number;
        if (absR <= this.epsilon) {
          huberGrad = -r; // MSE gradient
        } else {
          huberGrad = -this.epsilon * Math.sign(r); // absolute gradient
        }

        for (let j = 0; j < p; j++) {
          const g = huberGrad * (xi[j] ?? 0) + this.alpha * (w[j] ?? 0);
          gradW[j]! += g;
        }
        gradB += huberGrad;
      }

      for (let j = 0; j < p; j++) {
        const g = (gradW[j] ?? 0) / n;
        w[j]! -= lr * g;
        maxGrad = Math.max(maxGrad, Math.abs(g));
      }
      if (this.fitIntercept) b -= lr * gradB / n;

      this.nIter_ = iter + 1;
      if (maxGrad < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;

    // Mark outliers
    this.outliers_ = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      let pred = b;
      for (let j = 0; j < p; j++) pred += (w[j] ?? 0) * ((X[i]![j]) ?? 0);
      if (Math.abs((y[i] ?? 0) - pred) > this.epsilon) this.outliers_[i] = 1;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("HuberRegressor");
    return new Float64Array(
      X.map((xi) => {
        let pred = this.intercept_;
        for (let j = 0; j < xi.length; j++)
          pred += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
        return pred;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((preds[i] ?? 0) - (y[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot < 1e-10 ? 1 : 1 - ssRes / ssTot;
  }
}

export interface LarsOptions {
  nNonzeroCoefs?: number;
  fitIntercept?: boolean;
  normalize?: boolean;
}

export class Lars {
  nNonzeroCoefs: number;
  fitIntercept: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  alphas_: Float64Array | null = null;
  active_: number[] | null = null;
  nIter_: number = 0;

  constructor(options: LarsOptions = {}) {
    this.nNonzeroCoefs = options.nNonzeroCoefs ?? 500;
    this.fitIntercept = options.fitIntercept ?? true;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    // Center if fitIntercept
    let yMean = 0;
    const xMeans = new Float64Array(p);
    if (this.fitIntercept) {
      for (let i = 0; i < n; i++) yMean += (y[i] ?? 0) / n;
      for (let j = 0; j < p; j++) {
        for (let i = 0; i < n; i++) xMeans[j]! += (X[i]![j] ?? 0) / n;
      }
    }

    const Xc = X.map((row) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = (row[j] ?? 0) - (xMeans[j] ?? 0);
      return r;
    });
    const yc = new Float64Array(y.map((yi) => (yi ?? 0) - yMean));

    // LARS algorithm (simplified forward stagewise)
    const coef = new Float64Array(p);
    const residual = new Float64Array(yc);
    const active: number[] = [];
    const alphas: number[] = [];

    const maxIter = Math.min(this.nNonzeroCoefs, p);

    for (let step = 0; step < maxIter; step++) {
      // Find feature most correlated with residual
      let maxCorr = -Infinity;
      let bestJ = -1;
      for (let j = 0; j < p; j++) {
        if (active.includes(j)) continue;
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (Xc[i]![j] ?? 0) * (residual[i] ?? 0);
        corr = Math.abs(corr / n);
        if (corr > maxCorr) {
          maxCorr = corr;
          bestJ = j;
        }
      }
      if (bestJ < 0 || maxCorr < 1e-10) break;
      active.push(bestJ);
      alphas.push(maxCorr);

      // Simple OLS step along active set direction
      // Use Gram-Schmidt on active set (simplified)
      const XA = Xc.map((row) => new Float64Array(active.map((j) => row[j] ?? 0)));
      const gram: number[][] = active.map((_, a) =>
        active.map((_, b) => {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += (XA[i]![a] ?? 0) * (XA[i]![b] ?? 0);
          return dot / n;
        }),
      );

      const XAy = new Float64Array(active.length);
      for (let a = 0; a < active.length; a++) {
        for (let i = 0; i < n; i++) XAy[a]! += (XA[i]![a] ?? 0) * (residual[i] ?? 0);
        XAy[a]! /= n;
      }

      // Solve gram * w = XAy (Gauss-Seidel)
      const w = new Float64Array(active.length);
      for (let gs = 0; gs < 100; gs++) {
        for (let a = 0; a < active.length; a++) {
          let sum = XAy[a] ?? 0;
          for (let b = 0; b < active.length; b++) {
            if (b !== a) sum -= (gram[a]![b] ?? 0) * (w[b] ?? 0);
          }
          w[a] = sum / ((gram[a]![a] ?? 1) + 1e-8);
        }
      }

      // Update coefficients and residual
      for (let a = 0; a < active.length; a++) {
        coef[active[a]!] = w[a] ?? 0;
      }
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let j = 0; j < p; j++) pred += (coef[j] ?? 0) * (Xc[i]![j] ?? 0);
        residual[i] = (yc[i] ?? 0) - pred;
      }
    }

    this.coef_ = coef;
    this.intercept_ = this.fitIntercept
      ? yMean - (() => {
          let sum = 0;
          for (let j = 0; j < p; j++) sum += (coef[j] ?? 0) * (xMeans[j] ?? 0);
          return sum;
        })()
      : 0;
    this.alphas_ = new Float64Array(alphas);
    this.active_ = active;
    this.nIter_ = active.length;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new NotFittedError("Lars");
    return new Float64Array(
      X.map((xi) => {
        let pred = this.intercept_;
        for (let j = 0; j < xi.length; j++)
          pred += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
        return pred;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const preds = this.predict(X);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((preds[i] ?? 0) - (y[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return ssTot < 1e-10 ? 1 : 1 - ssRes / ssTot;
  }
}
