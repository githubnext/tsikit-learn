/**
 * Dirichlet Process Gaussian Mixture (infinite mixture model).
 */

export class DirichletProcessGMM {
  private means_!: Float64Array[];
  private covariances_!: Float64Array[][];
  private weights_!: Float64Array;
  private nActiveComponents_ = 0;
  private fitted_ = false;

  constructor(
    private maxComponents = 20,
    private alpha = 1.0,
    private maxIter = 100,
    private tol = 1e-4
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const K = this.maxComponents;
    // Stick-breaking initialization
    let betas = new Float64Array(K).map(() => Math.random());
    const sticks = new Float64Array(K);
    let cumprod = 1;
    for (let k = 0; k < K; k++) {
      sticks[k] = betas[k]! * cumprod;
      cumprod *= (1 - (betas[k] ?? 0));
    }
    let weights = new Float64Array(sticks);
    const ws = weights.reduce((s, v) => s + v, 0);
    for (let k = 0; k < K; k++) weights[k] = (weights[k] ?? 0) / (ws + 1e-10);

    // Initialize cluster params from data
    let means = Array.from({ length: K }, (_, k) => new Float64Array(X[k % n]!));
    let covs = Array.from({ length: K }, () => Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = 1;
      return row;
    }));

    let prevLogLik = -Number.POSITIVE_INFINITY;
    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step
      const r = Array.from({ length: n }, (_, i) => {
        const logProbs = new Float64Array(K).map((_, k) => {
          const diff = new Float64Array(X[i]!.map((v, j) => v - (means[k]![j] ?? 0)));
          const maha = diff.reduce((s, v, a) => s + v * covs[k]![a]!.reduce((ss, c, b) => ss + c * (diff[b] ?? 0), 0), 0);
          return Math.log(weights[k]! + 1e-300) - 0.5 * maha;
        });
        const maxL = Math.max(...logProbs);
        const expL = logProbs.map(l => Math.exp(l - maxL));
        const sumL = expL.reduce((s, v) => s + v, 0);
        return new Float64Array(expL.map(v => v / (sumL + 1e-10)));
      });

      // M-step
      const Nk = new Float64Array(K).map((_, k) => r.reduce((s, ri) => s + (ri[k] ?? 0), 0) + 1e-10);
      means = Array.from({ length: K }, (_, k) => {
        const m = new Float64Array(p);
        for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (r[i]![k] ?? 0) * (X[i]![j] ?? 0);
        return new Float64Array(m.map(v => v / (Nk[k] ?? 1)));
      });
      covs = Array.from({ length: K }, (_, k) => {
        const cov = Array.from({ length: p }, () => new Float64Array(p));
        for (let i = 0; i < n; i++) {
          const diff = new Float64Array(X[i]!.map((v, j) => v - (means[k]![j] ?? 0)));
          for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov[a]![b] = (cov[a]![b] ?? 0) + (r[i]![k] ?? 0) * (diff[a] ?? 0) * (diff[b] ?? 0);
        }
        return cov.map((row, ii) => new Float64Array(row.map((v, j) => v / (Nk[k] ?? 1) + (ii === j ? 1e-6 : 0))));
      });

      // Update weights using DP stick-breaking posterior
      for (let k = 0; k < K; k++) {
        const nk = Nk[k] ?? 0;
        const nAfter = Nk.slice(k + 1).reduce((s, v) => s + v, 0);
        betas[k] = (nk + 1) / (nk + nAfter + this.alpha);
      }
      let cum = 1;
      for (let k = 0; k < K; k++) { sticks[k] = (betas[k] ?? 0) * cum; cum *= (1 - (betas[k] ?? 0)); }
      const sw = sticks.reduce((s, v) => s + v, 0);
      weights = new Float64Array(sticks.map(v => v / (sw + 1e-10)));

      const logLik = r.reduce((s, ri, i) => s + Math.log(ri.reduce((ss, v, k) => ss + v * (weights[k] ?? 1e-300), 0) + 1e-300), 0);
      if (Math.abs(logLik - prevLogLik) < this.tol) break;
      prevLogLik = logLik;
    }
    this.means_ = means;
    this.covariances_ = covs;
    this.weights_ = weights;
    this.nActiveComponents_ = Array.from(weights).filter(v => v > 0.01).length;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let best = 0, bestS = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.maxComponents; k++) {
        const diff = new Float64Array(x.map((v, j) => v - (this.means_[k]![j] ?? 0)));
        const maha = diff.reduce((s, v, a) => s + v * this.covariances_[k]![a]!.reduce((ss, c, b) => ss + c * (diff[b] ?? 0), 0), 0);
        const score = Math.log(this.weights_[k]! + 1e-300) - 0.5 * maha;
        if (score > bestS) { bestS = score; best = k; }
      }
      return best;
    }));
  }

  get weights(): Float64Array { return this.weights_; }
  get means(): Float64Array[] { return this.means_; }
  get nActiveComponents(): number { return this.nActiveComponents_; }
}
