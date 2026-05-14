/**
 * BayesianGaussianMixture.
 * Mirrors sklearn.mixture.BayesianGaussianMixture.
 */

import { NotFittedError } from "../exceptions.js";

export interface BayesianGaussianMixtureOptions {
  nComponents?: number;
  maxIter?: number;
  tol?: number;
  weightConcentrationPrior?: number;
}

export class BayesianGaussianMixture {
  nComponents: number;
  maxIter: number;
  tol: number;
  weightConcentrationPrior: number;

  weights_: Float64Array | null = null;
  means_: Float64Array[] | null = null;
  covariances_: Float64Array[][] | null = null;
  converged_: boolean = false;
  nIter_: number = 0;

  constructor(options: BayesianGaussianMixtureOptions = {}) {
    this.nComponents = options.nComponents ?? 1;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-3;
    this.weightConcentrationPrior = options.weightConcentrationPrior ?? 1e-3;
  }

  private _logNormal(
    x: Float64Array,
    mean: Float64Array,
    cov: Float64Array[],
  ): number {
    const d = x.length;
    let logDet = 0;
    let mahal = 0;
    // Diagonal covariance approximation
    for (let j = 0; j < d; j++) {
      const sigma2 = cov[j]![j] ?? 1;
      logDet += Math.log(Math.max(sigma2, 1e-10));
      const diff = (x[j] ?? 0) - (mean[j] ?? 0);
      mahal += (diff * diff) / Math.max(sigma2, 1e-10);
    }
    return -0.5 * (d * Math.log(2 * Math.PI) + logDet + mahal);
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const K = this.nComponents;

    // Initialize means with random samples
    const means: Float64Array[] = Array.from({ length: K }, () => {
      const idx = Math.floor(Math.random() * n);
      return new Float64Array(X[idx]!);
    });

    // Initialize uniform responsibilities
    let resp = Array.from({ length: n }, () =>
      new Float64Array(K).map(() => 1 / K),
    );

    // Dirichlet concentration parameters
    let alpha = new Float64Array(K).fill(
      1 / K + this.weightConcentrationPrior,
    );

    let prevLogLik = -Infinity;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // M-step: compute weighted statistics
      const nk = new Float64Array(K);
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < K; k++) nk[k]! += resp[i]![k] ?? 0;
      }

      // Update alpha (Dirichlet params)
      for (let k = 0; k < K; k++) {
        alpha[k] = this.weightConcentrationPrior + (nk[k] ?? 0);
      }

      // Update means
      for (let k = 0; k < K; k++) {
        const m = new Float64Array(d);
        for (let i = 0; i < n; i++) {
          const r = resp[i]![k] ?? 0;
          for (let j = 0; j < d; j++) m[j]! += r * (X[i]![j] ?? 0);
        }
        const nkk = nk[k] ?? 1;
        for (let j = 0; j < d; j++) m[j] = m[j]! / nkk;
        means[k] = m;
      }

      // Update covariances (diagonal)
      const covs: Float64Array[][] = Array.from({ length: K }, () =>
        Array.from({ length: d }, () => new Float64Array(d)),
      );
      for (let k = 0; k < K; k++) {
        const nkk = Math.max(nk[k] ?? 0, 1e-10);
        for (let i = 0; i < n; i++) {
          const r = resp[i]![k] ?? 0;
          for (let j = 0; j < d; j++) {
            const diff = (X[i]![j] ?? 0) - (means[k]![j] ?? 0);
            covs[k]![j]![j]! += r * diff * diff;
          }
        }
        for (let j = 0; j < d; j++) {
          covs[k]![j]![j] = (covs[k]![j]![j] ?? 0) / nkk + 1e-6;
        }
      }

      // E-step: compute log weights
      const alphaSum = alpha.reduce((a, b) => a + b, 0);
      const logWeights = alpha.map(
        (a) => Math.log(a) - Math.log(alphaSum),
      );

      // Update responsibilities
      let logLik = 0;
      const newResp: Float64Array[] = [];
      for (let i = 0; i < n; i++) {
        const logProbs = new Float64Array(K);
        for (let k = 0; k < K; k++) {
          logProbs[k] =
            (logWeights[k] ?? 0) +
            this._logNormal(X[i]!, means[k]!, covs[k]!);
        }
        const maxLog = Math.max(...logProbs);
        const probs = logProbs.map((lp) => Math.exp(lp - maxLog));
        const sum = probs.reduce((a, b) => a + b, 0);
        const r = probs.map((p) => p / (sum || 1));
        newResp.push(r);
        logLik += Math.log(sum || 1e-10) + maxLog;
      }
      resp = newResp;

      // Check convergence
      if (Math.abs(logLik - prevLogLik) < this.tol) {
        this.converged_ = true;
        break;
      }
      prevLogLik = logLik;
      this.nIter_ = iter + 1;
    }

    // Finalize
    const alphaSum = alpha.reduce((a, b) => a + b, 0);
    this.weights_ = new Float64Array(alpha.map((a) => a / alphaSum));
    this.means_ = means;
    // simplified: store diagonal variances
    const covs: Float64Array[][] = Array.from({ length: K }, () =>
      Array.from({ length: d }, () => new Float64Array(d)),
    );
    const nk = new Float64Array(K);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) nk[k]! += resp[i]![k] ?? 0;
    }
    for (let k = 0; k < K; k++) {
      const nkk = Math.max(nk[k] ?? 0, 1e-10);
      for (let i = 0; i < n; i++) {
        const r = resp[i]![k] ?? 0;
        for (let j = 0; j < d; j++) {
          const diff = (X[i]![j] ?? 0) - (means[k]![j] ?? 0);
          covs[k]![j]![j]! += r * diff * diff;
        }
      }
      for (let j = 0; j < d; j++) {
        covs[k]![j]![j] = (covs[k]![j]![j] ?? 0) / nkk + 1e-6;
      }
    }
    this.covariances_ = covs;

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.weights_ || !this.means_ || !this.covariances_)
      throw new NotFittedError("BayesianGaussianMixture");

    return new Int32Array(
      X.map((x) => {
        let maxLogProb = -Infinity;
        let best = 0;
        for (let k = 0; k < this.nComponents; k++) {
          const lp =
            Math.log(this.weights_![k] ?? 1e-10) +
            this._logNormal(x, this.means_![k]!, this.covariances_![k]!);
          if (lp > maxLogProb) {
            maxLogProb = lp;
            best = k;
          }
        }
        return best;
      }),
    );
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.predict(X);
  }

  score(X: Float64Array[]): number {
    if (!this.weights_ || !this.means_ || !this.covariances_)
      throw new NotFittedError("BayesianGaussianMixture");
    let logLik = 0;
    for (const x of X) {
      let sum = 0;
      for (let k = 0; k < this.nComponents; k++) {
        sum +=
          (this.weights_![k] ?? 0) *
          Math.exp(this._logNormal(x, this.means_![k]!, this.covariances_![k]!));
      }
      logLik += Math.log(Math.max(sum, 1e-10));
    }
    return logLik / X.length;
  }
}
