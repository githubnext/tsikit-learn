/**
 * Gaussian Mixture Model.
 * Mirrors sklearn.mixture.GaussianMixture.
 */

import { NotFittedError } from "../exceptions.js";

export interface GaussianMixtureOptions {
  nComponents?: number;
  covarianceType?: "full" | "tied" | "diag" | "spherical";
  tol?: number;
  maxIter?: number;
  nInit?: number;
  regCovar?: number;
}

export class GaussianMixture {
  nComponents: number;
  covarianceType: "full" | "tied" | "diag" | "spherical";
  tol: number;
  maxIter: number;
  nInit: number;
  regCovar: number;

  weights_: Float64Array | null = null;
  means_: Float64Array[] | null = null;
  covariances_: Float64Array[][] | null = null;
  converged_: boolean = false;
  nIter_: number = 0;
  lowerBound_: number = Number.NEGATIVE_INFINITY;

  constructor(options: GaussianMixtureOptions = {}) {
    this.nComponents = options.nComponents ?? 1;
    this.covarianceType = options.covarianceType ?? "full";
    this.tol = options.tol ?? 1e-3;
    this.maxIter = options.maxIter ?? 100;
    this.nInit = options.nInit ?? 1;
    this.regCovar = options.regCovar ?? 1e-6;
  }

  private _logNormalPdf(x: Float64Array, mean: Float64Array, variance: number): number {
    const p = x.length;
    let sum = 0;
    for (let j = 0; j < p; j++) {
      sum += ((x[j] ?? 0) - (mean[j] ?? 0)) ** 2 / variance;
    }
    return -0.5 * (p * Math.log(2 * Math.PI * variance) + sum);
  }

  private _eStep(X: Float64Array[], means: Float64Array[], variances: number[], weights: Float64Array): Float64Array[] {
    const n = X.length;
    const k = this.nComponents;
    const resp: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
    for (let i = 0; i < n; i++) {
      const r = resp[i] as Float64Array;
      let sumR = 0;
      for (let c = 0; c < k; c++) {
        const logP = Math.log(weights[c] ?? 1 / k) + this._logNormalPdf(X[i] as Float64Array, means[c] as Float64Array, variances[c] ?? 1);
        r[c] = Math.exp(logP);
        sumR += r[c] ?? 0;
      }
      if (sumR === 0) sumR = 1e-10;
      for (let c = 0; c < k; c++) r[c] = (r[c] ?? 0) / sumR;
    }
    return resp;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = this.nComponents;

    // Initialize with k-means++ style
    const means: Float64Array[] = [];
    means.push(new Float64Array(X[Math.floor(Math.random() * n)] ?? new Float64Array(p)));
    for (let c = 1; c < k; c++) {
      const dists = X.map(xi => {
        let minD = Number.POSITIVE_INFINITY;
        for (const m of means) {
          let d = 0;
          for (let j = 0; j < p; j++) d += ((xi[j] ?? 0) - (m[j] ?? 0)) ** 2;
          if (d < minD) minD = d;
        }
        return minD;
      });
      const totalD = dists.reduce((a, b) => a + b, 0);
      let r = Math.random() * totalD;
      let idx = 0;
      for (let i = 0; i < n; i++) {
        r -= dists[i] ?? 0;
        if (r <= 0) { idx = i; break; }
      }
      means.push(new Float64Array(X[idx] ?? new Float64Array(p)));
    }

    const variances = new Float64Array(k).fill(1);
    const weights = new Float64Array(k).fill(1 / k);

    let prevLogLik = Number.NEGATIVE_INFINITY;
    for (let iter = 0; iter < this.maxIter; iter++) {
      // E step
      const resp = this._eStep(X, means, Array.from(variances), weights);

      // M step
      for (let c = 0; c < k; c++) {
        let Nc = 0;
        for (let i = 0; i < n; i++) Nc += (resp[i] as Float64Array)[c] ?? 0;
        weights[c] = Nc / n;
        // Update mean
        const newMean = new Float64Array(p);
        for (let i = 0; i < n; i++) {
          const r = (resp[i] as Float64Array)[c] ?? 0;
          for (let j = 0; j < p; j++) newMean[j] = (newMean[j] ?? 0) + r * ((X[i] as Float64Array)[j] ?? 0);
        }
        for (let j = 0; j < p; j++) newMean[j] = (newMean[j] ?? 0) / (Nc || 1);
        means[c] = newMean;
        // Update variance (spherical)
        let v = 0;
        for (let i = 0; i < n; i++) {
          const r = (resp[i] as Float64Array)[c] ?? 0;
          for (let j = 0; j < p; j++) v += r * ((X[i] as Float64Array)[j] ?? 0 - (newMean[j] ?? 0)) ** 2;
        }
        variances[c] = v / (Nc * p || 1) + this.regCovar;
      }

      // Compute log likelihood
      let logLik = 0;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < k; c++) {
          s += (weights[c] ?? 0) * Math.exp(this._logNormalPdf(X[i] as Float64Array, means[c] as Float64Array, variances[c] ?? 1));
        }
        logLik += Math.log(s || 1e-300);
      }

      this.nIter_ = iter + 1;
      if (Math.abs(logLik - prevLogLik) < this.tol) {
        this.converged_ = true;
        this.lowerBound_ = logLik;
        break;
      }
      prevLogLik = logLik;
    }

    this.weights_ = weights;
    this.means_ = means;
    this.covariances_ = means.map((_, c) => [new Float64Array(p).fill(variances[c] ?? 1)]);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const resp = this.predictProba(X);
    return Int32Array.from(resp.map(r => {
      let maxC = 0; let maxV = r[0] ?? 0;
      for (let c = 1; c < r.length; c++) { if ((r[c] ?? 0) > maxV) { maxV = r[c] ?? 0; maxC = c; } }
      return maxC;
    }));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.weights_ || !this.means_) throw new NotFittedError("GaussianMixture is not fitted.");
    const variances = (this.covariances_ as Float64Array[][]).map(c => (c[0] as Float64Array)[0] ?? 1);
    return this._eStep(X, this.means_, variances, this.weights_);
  }

  score(X: Float64Array[]): number {
    if (!this.weights_ || !this.means_) throw new NotFittedError("GaussianMixture is not fitted.");
    const variances = (this.covariances_ as Float64Array[][]).map(c => (c[0] as Float64Array)[0] ?? 1);
    let logLik = 0;
    for (const xi of X) {
      let s = 0;
      for (let c = 0; c < this.nComponents; c++) {
        s += (this.weights_[c] ?? 0) * Math.exp(this._logNormalPdf(xi, this.means_[c] as Float64Array, variances[c] ?? 1));
      }
      logLik += Math.log(s || 1e-300);
    }
    return logLik / X.length;
  }
}
