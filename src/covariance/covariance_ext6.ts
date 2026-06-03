/**
 * Extended covariance estimation: Toeplitz, Banded, and Heteroscedastic covariance.
 * Port of sklearn.covariance extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Toeplitz covariance estimator (assumes stationary covariance structure).
 */
export class ToeplitzCovarianceEstimator {
  private covariance_: Float64Array[] = [];
  private location_: Float64Array = new Float64Array(0);
  private fitted = false;

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    // Compute mean
    this.location_ = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) this.location_[j] = (this.location_[j] ?? 0) + (row[j] ?? 0) / n;

    // Compute autocovariance for each lag
    const cov = new Float64Array(p);
    for (let lag = 0; lag < p; lag++) {
      let s = 0;
      for (const row of X) {
        for (let j = 0; j < p - lag; j++) {
          s += ((row[j] ?? 0) - (this.location_[j] ?? 0)) * ((row[j + lag] ?? 0) - (this.location_[j + lag] ?? 0));
        }
      }
      cov[lag] = s / (n * (p - lag));
    }

    // Build Toeplitz matrix
    this.covariance_ = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: p }, (__, j) => cov[Math.abs(i - j)] ?? 0)
    );

    this.fitted = true;
    return this;
  }

  get covariance(): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("ToeplitzCovarianceEstimator not fitted");
    return this.covariance_;
  }

  get location(): Float64Array {
    if (!this.fitted) throw new NotFittedError("ToeplitzCovarianceEstimator not fitted");
    return this.location_;
  }
}

/**
 * Banded covariance estimator — regularizes by zeroing out off-diagonal entries beyond a bandwidth.
 */
export class BandedCovarianceEstimator {
  private bandwidth: number;
  private covariance_: Float64Array[] = [];
  private location_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { bandwidth?: number } = {}) {
    this.bandwidth = options.bandwidth ?? 1;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    this.location_ = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) this.location_[j] = (this.location_[j] ?? 0) + (row[j] ?? 0) / n;

    const fullCov = Array.from({ length: p }, () => new Float64Array(p));
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) {
          fullCov[j]![k] = (fullCov[j]?.[k] ?? 0) + ((row[j] ?? 0) - (this.location_[j] ?? 0)) * ((row[k] ?? 0) - (this.location_[k] ?? 0)) / n;
        }
      }
    }

    // Apply banding
    this.covariance_ = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: p }, (__, j) => Math.abs(i - j) <= this.bandwidth ? (fullCov[i]?.[j] ?? 0) : 0)
    );

    this.fitted = true;
    return this;
  }

  get covariance(): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("BandedCovarianceEstimator not fitted");
    return this.covariance_;
  }
}

/**
 * Heteroscedastic noise model — estimates separate noise variance per feature.
 */
export class HeteroscedasticNoise {
  private noiseVariances_: Float64Array = new Float64Array(0);
  private fitted = false;

  fit(X: Float64Array[], yTrue: Float64Array[], yPred: Float64Array[]): this {
    const n = X.length;
    const nOut = yTrue[0]?.length ?? 1;
    this.noiseVariances_ = new Float64Array(nOut);

    // Estimate noise variance as average squared residual per output
    for (let o = 0; o < nOut; o++) {
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const r = (yTrue[i]?.[o] ?? 0) - (yPred[i]?.[o] ?? 0);
        sumSq += r * r;
      }
      this.noiseVariances_[o] = sumSq / n;
    }

    this.fitted = true;
    return this;
  }

  get noiseVariances(): Float64Array {
    if (!this.fitted) throw new NotFittedError("HeteroscedasticNoise not fitted");
    return this.noiseVariances_;
  }

  weightedLoss(yTrue: Float64Array[], yPred: Float64Array[]): number {
    if (!this.fitted) throw new NotFittedError("HeteroscedasticNoise not fitted");
    let loss = 0;
    for (let i = 0; i < yTrue.length; i++) {
      for (let o = 0; o < (yTrue[i]?.length ?? 0); o++) {
        const r = (yTrue[i]?.[o] ?? 0) - (yPred[i]?.[o] ?? 0);
        loss += r * r / Math.max(this.noiseVariances_[o] ?? 1, 1e-10);
      }
    }
    return loss / yTrue.length;
  }
}

/**
 * Precision matrix estimation via nodewise regression (Meinshausen-Bühlmann).
 */
export class NodewiseRegressionPrecision {
  private betaMatrix_: Float64Array[] = [];
  private residualVariances_: Float64Array = new Float64Array(0);
  private fitted = false;
  private lambda: number;

  constructor(options: { lambda?: number } = {}) {
    this.lambda = options.lambda ?? 0.1;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    this.betaMatrix_ = Array.from({ length: p }, () => new Float64Array(p));
    this.residualVariances_ = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      // Regress j-th feature on all others with Lasso-like penalty (simplified ridge)
      const yj = Float64Array.from(X, row => row[j] ?? 0);
      const Xj = X.map(row => Float64Array.from({ length: p - 1 }, (_, k) => row[k >= j ? k + 1 : k] ?? 0));

      // Ridge regression
      const beta = this.ridgeRegress(Xj, yj, this.lambda);

      // Fill beta matrix
      for (let k = 0; k < p - 1; k++) {
        this.betaMatrix_[j]![k >= j ? k + 1 : k] = beta[k] ?? 0;
      }

      // Residual variance
      const preds = Xj.map(row => row.reduce((s, v, k) => s + v * (beta[k] ?? 0), 0));
      let resVar = 0;
      for (let i = 0; i < n; i++) resVar += ((yj[i] ?? 0) - (preds[i] ?? 0)) ** 2;
      this.residualVariances_[j] = resVar / n;
    }

    this.fitted = true;
    return this;
  }

  private ridgeRegress(X: Float64Array[], y: Float64Array, lambda: number): Float64Array {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    // Diagonal approximation of (X^T X + lambda I)^{-1} X^T y
    const diag = new Float64Array(p);
    const XtY = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) {
        diag[j] = (diag[j] ?? 0) + (X[i]?.[j] ?? 0) ** 2;
        XtY[j] = (XtY[j] ?? 0) + (X[i]?.[j] ?? 0) * (y[i] ?? 0);
      }
      diag[j] = (diag[j] ?? 0) + lambda;
    }
    return Float64Array.from({ length: p }, (_, j) => (XtY[j] ?? 0) / Math.max(diag[j] ?? 1, 1e-10));
  }

  getPrecisionMatrix(): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("NodewiseRegressionPrecision not fitted");
    const p = this.betaMatrix_.length;
    const precision = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        if (i === j) row[j] = 1 / Math.max(this.residualVariances_[i] ?? 1, 1e-10);
        else row[j] = -(this.betaMatrix_[i]?.[j] ?? 0) / Math.max(this.residualVariances_[i] ?? 1, 1e-10);
      }
      return row;
    });
    return precision;
  }
}
