/**
 * Variational Inference Gaussian Mixture and Hidden Markov Model.
 */

export class BayesianGaussianMixtureVI {
  private weights_!: Float64Array;
  private means_!: Float64Array[];
  private covariances_!: Float64Array[][];
  private fitted_ = false;

  constructor(
    private nComponents = 5,
    private weightConcentrationPrior = 1e-3,
    private maxIter = 100,
    private tol = 1e-3
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const K = this.nComponents;
    // Initialize with random responsibilities
    let r = Array.from({ length: n }, () => {
      const vals = new Float64Array(K).map(() => Math.random());
      const s = vals.reduce((acc, v) => acc + v, 0);
      return new Float64Array(vals.map(v => v / s));
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      // M-step: update parameters
      const Nk = new Float64Array(K).map((_, k) => r.reduce((s, ri) => s + (ri[k] ?? 0), 0));
      const means = Array.from({ length: K }, (_, k) => {
        const wMean = new Float64Array(p);
        for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) wMean[j] = (wMean[j] ?? 0) + (r[i]![k] ?? 0) * (X[i]![j] ?? 0);
        return new Float64Array(wMean.map(v => v / ((Nk[k] ?? 1) + this.weightConcentrationPrior)));
      });

      const covs = Array.from({ length: K }, (_, k) => {
        const cov = Array.from({ length: p }, () => new Float64Array(p));
        for (let i = 0; i < n; i++) {
          const diff = new Float64Array(X[i]!.map((v, j) => v - (means[k]![j] ?? 0)));
          for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov[a]![b] = (cov[a]![b] ?? 0) + (r[i]![k] ?? 0) * (diff[a] ?? 0) * (diff[b] ?? 0);
        }
        const nk = Nk[k] ?? 1;
        return cov.map(row => new Float64Array(row.map((v, j) => v / nk + (j === 0 ? 1e-6 : 0))));
      });

      const alpha = new Float64Array(K).map((_, k) => (Nk[k] ?? 0) + this.weightConcentrationPrior);
      const alphaSum = alpha.reduce((s, v) => s + v, 0);
      const weights = new Float64Array(alpha.map(v => v / alphaSum));

      // E-step: update responsibilities
      const prevR = r;
      r = Array.from({ length: n }, (_, i) => {
        const logProbs = new Float64Array(K).map((_, k) => {
          const diff = new Float64Array(X[i]!.map((v, j) => v - (means[k]![j] ?? 0)));
          const cov = covs[k]!;
          const maha = diff.reduce((s, v, a) => s + v * cov[a]!.reduce((ss, c, b) => ss + c * (diff[b] ?? 0), 0), 0);
          return Math.log(weights[k] ?? 1e-300) - 0.5 * maha;
        });
        const maxL = Math.max(...logProbs);
        const expL = logProbs.map(l => Math.exp(l - maxL));
        const sumL = expL.reduce((s, v) => s + v, 0);
        return new Float64Array(expL.map(v => v / sumL));
      });

      const diff = r.reduce((s, ri, i) => s + ri.reduce((ss, v, k) => ss + (v - (prevR[i]![k] ?? 0)) ** 2, 0), 0);
      this.weights_ = weights;
      this.means_ = means;
      this.covariances_ = covs;
      if (diff < this.tol) break;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let best = 0, bestS = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nComponents; k++) {
        const diff = new Float64Array(x.map((v, j) => v - (this.means_[k]![j] ?? 0)));
        const maha = diff.reduce((s, v, a) => s + v * this.covariances_[k]![a]!.reduce((ss, c, b) => ss + c * (diff[b] ?? 0), 0), 0);
        const score = Math.log(this.weights_[k] ?? 1e-300) - 0.5 * maha;
        if (score > bestS) { bestS = score; best = k; }
      }
      return best;
    }));
  }

  get weights(): Float64Array { return this.weights_; }
  get means(): Float64Array[] { return this.means_; }
}
