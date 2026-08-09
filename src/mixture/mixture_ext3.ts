/**
 * Mixture model extensions: BayesianGMM extensions, StudentT mixture.
 * Mirrors sklearn.mixture advanced probabilistic models.
 */

import { BaseEstimator } from "../base.js";

export interface StudentTMixtureParams {
  n_components?: number;
  df?: number;
  max_iter?: number;
  tol?: number;
  random_state?: number | null;
}

/** Student-T Mixture Model: more robust than Gaussian mixture. */
export class StudentTMixture extends BaseEstimator {
  n_components: number;
  df: number;
  max_iter: number;
  tol: number;
  random_state: number | null;
  weights_: Float64Array = new Float64Array(0);
  means_: Float64Array[] = [];
  scales_: Float64Array[] = [];
  converged_ = false;
  n_iter_ = 0;

  constructor(params: StudentTMixtureParams = {}) {
    super();
    this.n_components = params.n_components ?? 3;
    this.df = params.df ?? 5;
    this.max_iter = params.max_iter ?? 100;
    this.tol = params.tol ?? 1e-3;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    const k = this.n_components;
    // Initialize with random means
    const seed = this.random_state ?? 42;
    this.weights_ = new Float64Array(k).fill(1 / k);
    this.means_ = Array.from({ length: k }, (_, c) => X[((seed + c * 37) * 1664525) % n]!.slice() as Float64Array);
    this.scales_ = Array.from({ length: k }, () => new Float64Array(nf).fill(1));
    let prevLogLik = -Number.POSITIVE_INFINITY;
    for (let iter = 0; iter < this.max_iter; iter++) {
      // E-step: compute responsibilities and u weights
      const resp = Array.from({ length: n }, () => new Float64Array(k));
      const uWeights = Array.from({ length: n }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        let sumProb = 0;
        for (let c = 0; c < k; c++) {
          let mhd = 0;
          for (let d = 0; d < nf; d++) mhd += ((X[i]?.[d] ?? 0) - (this.means_[c]?.[d] ?? 0)) ** 2 / Math.max(this.scales_[c]?.[d] ?? 1, 1e-10);
          const p = (this.weights_[c] ?? 0) * (1 + mhd / this.df) ** (-(this.df + nf) / 2);
          resp[i]![c] = p;
          sumProb += p;
        }
        if (sumProb > 0) for (let c = 0; c < k; c++) resp[i]![c] = (resp[i]![c] ?? 0) / sumProb;
        for (let c = 0; c < k; c++) {
          let mhd = 0;
          for (let d = 0; d < nf; d++) mhd += ((X[i]?.[d] ?? 0) - (this.means_[c]?.[d] ?? 0)) ** 2 / Math.max(this.scales_[c]?.[d] ?? 1, 1e-10);
          uWeights[i]![c] = (this.df + nf) / (this.df + mhd);
        }
      }
      // M-step
      const newWeights = new Float64Array(k);
      const newMeans: Float64Array[] = Array.from({ length: k }, () => new Float64Array(nf));
      const newScales: Float64Array[] = Array.from({ length: k }, () => new Float64Array(nf));
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          const r = resp[i]?.[c] ?? 0;
          const u = uWeights[i]?.[c] ?? 0;
          newWeights[c] = (newWeights[c] ?? 0) + r;
          for (let d = 0; d < nf; d++) newMeans[c]![d] = (newMeans[c]![d] ?? 0) + r * u * (X[i]?.[d] ?? 0);
        }
      }
      let logLik = 0;
      for (let c = 0; c < k; c++) {
        const wc = newWeights[c] ?? 1e-10;
        this.weights_[c] = wc / n;
        const ruSum = Array.from({ length: n }, (_, i) => (resp[i]?.[c] ?? 0) * (uWeights[i]?.[c] ?? 0)).reduce((s, v) => s + v, 0);
        for (let d = 0; d < nf; d++) newMeans[c]![d] = (newMeans[c]![d] ?? 0) / Math.max(ruSum, 1e-10);
        for (let i = 0; i < n; i++) {
          const r = resp[i]?.[c] ?? 0, u = uWeights[i]?.[c] ?? 0;
          for (let d = 0; d < nf; d++) newScales[c]![d] = (newScales[c]![d] ?? 0) + r * u * ((X[i]?.[d] ?? 0) - (newMeans[c]?.[d] ?? 0)) ** 2;
        }
        for (let d = 0; d < nf; d++) newScales[c]![d] = Math.max((newScales[c]![d] ?? 0) / Math.max(wc, 1e-10), 1e-6);
      }
      this.means_ = newMeans;
      this.scales_ = newScales;
      for (let i = 0; i < n; i++) {
        let p = 0;
        for (let c = 0; c < k; c++) {
          let mhd = 0;
          for (let d = 0; d < nf; d++) mhd += ((X[i]?.[d] ?? 0) - (this.means_[c]?.[d] ?? 0)) ** 2 / Math.max(this.scales_[c]?.[d] ?? 1, 1e-10);
          p += (this.weights_[c] ?? 0) * (1 + mhd / this.df) ** (-(this.df + nf) / 2);
        }
        logLik += Math.log(Math.max(p, 1e-300));
      }
      if (Math.abs(logLik - prevLogLik) < this.tol) { this.converged_ = true; this.n_iter_ = iter + 1; break; }
      prevLogLik = logLik;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const proba = this.predict_proba(X);
    return new Int32Array(proba.map((p) => {
      let best = 0, bestV = -1;
      for (let c = 0; c < p.length; c++) if ((p[c] ?? 0) > bestV) { best = c; bestV = p[c] ?? 0; }
      return best;
    }));
  }

  predict_proba(X: Float64Array[]): Float64Array[] {
    const k = this.n_components;
    const nf = this.means_[0]?.length ?? 0;
    return X.map((xi) => {
      const p = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let mhd = 0;
        for (let d = 0; d < nf; d++) mhd += ((xi[d] ?? 0) - (this.means_[c]?.[d] ?? 0)) ** 2 / Math.max(this.scales_[c]?.[d] ?? 1, 1e-10);
        p[c] = (this.weights_[c] ?? 0) * (1 + mhd / this.df) ** (-(this.df + nf) / 2);
      }
      const sum = Array.from(p).reduce((s, v) => s + v, 0);
      return sum > 0 ? new Float64Array(p.map((v) => v / sum)) : p;
    });
  }

  score(X: Float64Array[]): number {
    const k = this.n_components;
    const nf = this.means_[0]?.length ?? 0;
    let total = 0;
    for (const xi of X) {
      let p = 0;
      for (let c = 0; c < k; c++) {
        let mhd = 0;
        for (let d = 0; d < nf; d++) mhd += ((xi[d] ?? 0) - (this.means_[c]?.[d] ?? 0)) ** 2 / Math.max(this.scales_[c]?.[d] ?? 1, 1e-10);
        p += (this.weights_[c] ?? 0) * (1 + mhd / this.df) ** (-(this.df + nf) / 2);
      }
      total += Math.log(Math.max(p, 1e-300));
    }
    return total / X.length;
  }
}
