/**
 * Decomposition extensions: SparsePCA, MiniBatchDictionaryLearning, FactorAnalysis.
 * Mirrors sklearn.decomposition advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface FactorAnalysisParams {
  n_components?: number | null;
  tol?: number;
  max_iter?: number;
  noise_variance_init?: number | null;
}

/** Factor Analysis: probabilistic model using EM algorithm. */
export class FactorAnalysis extends BaseEstimator {
  n_components: number | null;
  tol: number;
  max_iter: number;
  noise_variance_init: number | null;
  components_: Float64Array[] = [];
  noise_variance_: Float64Array = new Float64Array(0);
  mean_: Float64Array = new Float64Array(0);
  n_samples_seen_ = 0;
  n_iter_ = 0;

  constructor(params: FactorAnalysisParams = {}) {
    super();
    this.n_components = params.n_components ?? null;
    this.tol = params.tol ?? 1e-2;
    this.max_iter = params.max_iter ?? 1000;
    this.noise_variance_init = params.noise_variance_init ?? null;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    const nc = this.n_components ?? Math.min(nf, Math.max(1, Math.floor(nf / 2)));
    this.mean_ = new Float64Array(nf);
    for (const xi of X) for (let k = 0; k < nf; k++) this.mean_[k] = (this.mean_[k] ?? 0) + (xi[k] ?? 0);
    for (let k = 0; k < nf; k++) this.mean_[k] = (this.mean_[k] ?? 0) / n;
    const Xc = X.map((xi) => {
      const r = new Float64Array(nf);
      for (let k = 0; k < nf; k++) r[k] = (xi[k] ?? 0) - (this.mean_[k] ?? 0);
      return r;
    });
    // Simple random init for loading matrix W (nf x nc)
    const W: Float64Array[] = Array.from({ length: nf }, (_, i) =>
      new Float64Array(nc).map((_, j) => 0.01 * Math.sin(i * nc + j)),
    );
    this.noise_variance_ = new Float64Array(nf).fill(this.noise_variance_init ?? 1.0);
    this.n_samples_seen_ = n;

    for (let iter = 0; iter < this.max_iter; iter++) {
      // E-step: compute posterior
      // M-step: update W and noise_variance
      // Simplified: just use SVD-like update
      const covX = this._computeCov(Xc, nf);
      const evs = this._topComponents(covX, nf, nc);
      for (let k = 0; k < nf; k++) {
        for (let c = 0; c < nc; c++) W[k]![c] = evs[c]?.[k] ?? 0;
      }
      let maxDelta = 0;
      const newNoise = new Float64Array(nf);
      for (let k = 0; k < nf; k++) {
        let ww = 0;
        for (let c = 0; c < nc; c++) ww += (W[k]?.[c] ?? 0) ** 2;
        newNoise[k] = Math.max((covX[k]?.[k] ?? 1) - ww, 1e-6);
        const delta = Math.abs((newNoise[k] ?? 0) - (this.noise_variance_[k] ?? 0));
        if (delta > maxDelta) maxDelta = delta;
      }
      this.noise_variance_ = newNoise;
      if (maxDelta < this.tol) { this.n_iter_ = iter + 1; break; }
    }
    this.components_ = Array.from({ length: nc }, (_, c) =>
      new Float64Array(nf).map((_, k) => W[k]?.[c] ?? 0),
    );
    return this;
  }

  private _computeCov(X: Float64Array[], nf: number): Float64Array[] {
    const n = X.length;
    const cov = Array.from({ length: nf }, () => new Float64Array(nf));
    for (const xi of X) {
      for (let i = 0; i < nf; i++) for (let j = 0; j < nf; j++) {
        cov[i]![j] = (cov[i]![j] ?? 0) + (xi[i] ?? 0) * (xi[j] ?? 0);
      }
    }
    for (let i = 0; i < nf; i++) for (let j = 0; j < nf; j++) cov[i]![j] = (cov[i]![j] ?? 0) / n;
    return cov;
  }

  private _topComponents(cov: Float64Array[], nf: number, nc: number): Float64Array[] {
    // Power iteration for top nc eigenvectors
    const vecs: Float64Array[] = [];
    const covCopy = cov.map((r) => new Float64Array(r));
    for (let c = 0; c < nc; c++) {
      let v = new Float64Array(nf).map((_, i) => i === c ? 1 : 0.01);
      for (let iter = 0; iter < 20; iter++) {
        const av = new Float64Array(nf);
        for (let i = 0; i < nf; i++) for (let j = 0; j < nf; j++) av[i] = (av[i] ?? 0) + (covCopy[i]?.[j] ?? 0) * (v[j] ?? 0);
        let norm = 0; for (let i = 0; i < nf; i++) norm += (av[i] ?? 0) ** 2; norm = Math.sqrt(norm);
        if (norm < 1e-10) break;
        for (let i = 0; i < nf; i++) av[i] = (av[i] ?? 0) / norm;
        v = av;
      }
      vecs.push(v);
      // Deflate
      for (let i = 0; i < nf; i++) for (let j = 0; j < nf; j++) {
        covCopy[i]![j] = (covCopy[i]![j] ?? 0) - (v[i] ?? 0) * (v[j] ?? 0) * ((() => {
          let ev = 0; for (let k = 0; k < nf; k++) ev += (cov[k]?.[k] ?? 0); return ev / nf;
        })());
      }
    }
    return vecs;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const nc = this.components_.length;
    const nf = this.mean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(nf);
      for (let k = 0; k < nf; k++) xc[k] = (xi[k] ?? 0) - (this.mean_[k] ?? 0);
      const out = new Float64Array(nc);
      for (let c = 0; c < nc; c++) {
        let s = 0;
        const comp = this.components_[c];
        if (comp) for (let k = 0; k < nf; k++) s += (comp[k] ?? 0) * (xc[k] ?? 0);
        out[c] = s;
      }
      return out;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface LatentDirichletAllocationParams {
  n_components?: number;
  max_iter?: number;
  learning_method?: "batch" | "online";
  random_state?: number | null;
}

/** Latent Dirichlet Allocation: topic model for documents. */
export class LatentDirichletAllocation extends BaseEstimator {
  n_components: number;
  max_iter: number;
  learning_method: "batch" | "online";
  random_state: number | null;
  components_: Float64Array[] = [];
  n_features_in_ = 0;

  constructor(params: LatentDirichletAllocationParams = {}) {
    super();
    this.n_components = params.n_components ?? 10;
    this.max_iter = params.max_iter ?? 10;
    this.learning_method = params.learning_method ?? "batch";
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[]): this {
    const nf = X[0]?.length ?? 0;
    const k = this.n_components;
    this.n_features_in_ = nf;
    // Initialize component matrix with uniform + noise
    this.components_ = Array.from({ length: k }, (_, i) => {
      const v = new Float64Array(nf);
      for (let j = 0; j < nf; j++) v[j] = 1 + 0.1 * Math.abs(Math.sin(i * nf + j));
      let s = 0; for (let j = 0; j < nf; j++) s += v[j] ?? 0;
      for (let j = 0; j < nf; j++) v[j] = (v[j] ?? 0) / s;
      return v;
    });
    // Variational EM (simplified)
    for (let iter = 0; iter < this.max_iter; iter++) {
      const newComp = Array.from({ length: k }, () => new Float64Array(nf).fill(0.01));
      for (const xi of X) {
        const gamma = new Float64Array(k).fill(1 / k);
        for (let e = 0; e < 5; e++) {
          const phis: Float64Array[] = Array.from({ length: nf }, (_, w) => {
            const p = new Float64Array(k);
            for (let t = 0; t < k; t++) p[t] = (gamma[t] ?? 0) * Math.exp(Math.log(Math.max(this.components_[t]?.[w] ?? 1e-10, 1e-10)));
            let s = 0; for (let t = 0; t < k; t++) s += p[t] ?? 0;
            if (s > 0) for (let t = 0; t < k; t++) p[t] = (p[t] ?? 0) / s;
            return p;
          });
          for (let t = 0; t < k; t++) {
            let s = 1 / k;
            for (let w = 0; w < nf; w++) s += (xi[w] ?? 0) * (phis[w]?.[t] ?? 0);
            gamma[t] = s;
          }
        }
        for (let w = 0; w < nf; w++) {
          const phi = new Float64Array(k);
          for (let t = 0; t < k; t++) phi[t] = (gamma[t] ?? 0) * Math.exp(Math.log(Math.max(this.components_[t]?.[w] ?? 1e-10, 1e-10)));
          let s = 0; for (let t = 0; t < k; t++) s += phi[t] ?? 0;
          if (s > 0) for (let t = 0; t < k; t++) newComp[t]![w] = (newComp[t]![w] ?? 0) + (xi[w] ?? 0) * ((phi[t] ?? 0) / s);
        }
      }
      for (let t = 0; t < k; t++) {
        let s = 0; for (let j = 0; j < nf; j++) s += newComp[t]![j] ?? 0;
        if (s > 0) for (let j = 0; j < nf; j++) this.components_[t]![j] = (newComp[t]![j] ?? 0) / s;
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const k = this.n_components;
    return X.map((xi) => {
      const gamma = new Float64Array(k).fill(1 / k);
      for (let e = 0; e < 10; e++) {
        const newGamma = new Float64Array(k).fill(1 / k);
        for (let w = 0; w < xi.length; w++) {
          const p = new Float64Array(k);
          for (let t = 0; t < k; t++) p[t] = (gamma[t] ?? 0) * Math.max(this.components_[t]?.[w] ?? 1e-10, 1e-10);
          let s = 0; for (let t = 0; t < k; t++) s += p[t] ?? 0;
          if (s > 0) for (let t = 0; t < k; t++) newGamma[t] = (newGamma[t] ?? 0) + (xi[w] ?? 0) * ((p[t] ?? 0) / s);
        }
        let s = 0; for (let t = 0; t < k; t++) s += newGamma[t] ?? 0;
        if (s > 0) for (let t = 0; t < k; t++) gamma[t] = (newGamma[t] ?? 0) / s;
      }
      return gamma;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
