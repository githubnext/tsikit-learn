/**
 * Mixture model extensions: diagonal GMM, BIC/AIC selection, GMM with covariance types.
 */

export class DiagonalGMM {
  private means_: Float64Array[] = [];
  private vars_: Float64Array[] = [];
  private weights_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(
    private readonly nComponents = 1,
    private readonly maxIter = 100,
    private readonly tol = 1e-4
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    const K = this.nComponents;
    // Initialize
    this.weights_ = new Float64Array(K).fill(1 / K);
    this.means_ = Array.from({ length: K }, (_, k) => {
      const idx = Math.floor((k / K) * n);
      return new Float64Array(X[idx] ?? new Float64Array(nF));
    });
    this.vars_ = Array.from({ length: K }, () => new Float64Array(nF).fill(1.0));
    let prevLogLik = -Number.POSITIVE_INFINITY;
    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step
      const gamma = this._eStep(X);
      // M-step
      const Nk = new Float64Array(K);
      for (const g of gamma) for (let k = 0; k < K; k++) Nk[k] = (Nk[k] ?? 0) + (g[k] ?? 0);
      for (let k = 0; k < K; k++) {
        const nk = Nk[k] ?? 1;
        this.weights_[k] = nk / n;
        const mean = new Float64Array(nF);
        for (let i = 0; i < n; i++) for (let f = 0; f < nF; f++) mean[f] = (mean[f] ?? 0) + (gamma[i]?.[k] ?? 0) * (X[i]?.[f] ?? 0) / nk;
        this.means_[k] = mean;
        const vari = new Float64Array(nF);
        for (let i = 0; i < n; i++) for (let f = 0; f < nF; f++) {
          const d = (X[i]?.[f] ?? 0) - (mean[f] ?? 0);
          vari[f] = (vari[f] ?? 0) + (gamma[i]?.[k] ?? 0) * d * d / nk;
        }
        for (let f = 0; f < nF; f++) vari[f] = Math.max(vari[f] ?? 1e-6, 1e-6);
        this.vars_[k] = vari;
      }
      const logLik = this._logLikelihood(X);
      if (Math.abs(logLik - prevLogLik) < this.tol) break;
      prevLogLik = logLik;
    }
    this.fitted = true;
    return this;
  }

  private _eStep(X: Float64Array[]): Float64Array[] {
    return X.map((x) => {
      const logProbs = new Float64Array(this.nComponents);
      for (let k = 0; k < this.nComponents; k++) {
        logProbs[k] = Math.log(Math.max(this.weights_[k] ?? 1e-10, 1e-10)) + this._logPdf(x, k);
      }
      const maxLog = Math.max(...logProbs);
      const probs = new Float64Array(logProbs.map((lp) => Math.exp(lp - maxLog)));
      const sum = probs.reduce((a, b) => a + b, 0);
      return new Float64Array(probs.map((p) => p / Math.max(sum, 1e-10)));
    });
  }

  private _logPdf(x: Float64Array, k: number): number {
    const mean = this.means_[k]!;
    const vari = this.vars_[k]!;
    let logP = 0;
    for (let f = 0; f < x.length; f++) {
      const d = (x[f] ?? 0) - (mean[f] ?? 0);
      const v = vari[f] ?? 1;
      logP += -0.5 * (Math.log(2 * Math.PI * v) + d * d / v);
    }
    return logP;
  }

  private _logLikelihood(X: Float64Array[]): number {
    return X.reduce((sum, x) => {
      const logProbs = new Float64Array(this.nComponents);
      for (let k = 0; k < this.nComponents; k++) {
        logProbs[k] = Math.log(Math.max(this.weights_[k] ?? 1e-10, 1e-10)) + this._logPdf(x, k);
      }
      const maxLog = Math.max(...logProbs);
      return sum + maxLog + Math.log(logProbs.reduce((a, lp) => a + Math.exp(lp - maxLog), 0));
    }, 0);
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new Error("Not fitted");
    return new Int32Array(X.map((x) => {
      let best = 0, bestP = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nComponents; k++) {
        const p = Math.log(Math.max(this.weights_[k] ?? 1e-10, 1e-10)) + this._logPdf(x, k);
        if (p > bestP) { bestP = p; best = k; }
      }
      return best;
    }));
  }

  bic(X: Float64Array[]): number {
    const nF = X[0]?.length ?? 1;
    const nParams = this.nComponents * (1 + nF + nF) - 1;
    return -2 * this._logLikelihood(X) + nParams * Math.log(X.length);
  }

  aic(X: Float64Array[]): number {
    const nF = X[0]?.length ?? 1;
    const nParams = this.nComponents * (1 + nF + nF) - 1;
    return -2 * this._logLikelihood(X) + 2 * nParams;
  }
}

export class GMMModelSelector {
  selectByBIC(X: Float64Array[], kRange: Int32Array): { bestK: number; bics: Float64Array } {
    const bics = new Float64Array(kRange.length);
    let bestK = kRange[0] ?? 1;
    let bestBIC = Number.POSITIVE_INFINITY;
    for (let i = 0; i < kRange.length; i++) {
      const k = kRange[i]!;
      const gmm = new DiagonalGMM(k).fit(X);
      const bic = gmm.bic(X);
      bics[i] = bic;
      if (bic < bestBIC) { bestBIC = bic; bestK = k; }
    }
    return { bestK, bics };
  }

  selectByAIC(X: Float64Array[], kRange: Int32Array): { bestK: number; aics: Float64Array } {
    const aics = new Float64Array(kRange.length);
    let bestK = kRange[0] ?? 1;
    let bestAIC = Number.POSITIVE_INFINITY;
    for (let i = 0; i < kRange.length; i++) {
      const k = kRange[i]!;
      const gmm = new DiagonalGMM(k).fit(X);
      const aic = gmm.aic(X);
      aics[i] = aic;
      if (aic < bestAIC) { bestAIC = aic; bestK = k; }
    }
    return { bestK, aics };
  }
}
