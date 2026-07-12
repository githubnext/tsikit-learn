/**
 * Dirichlet Process Mixture Model (DPMM) via truncated variational inference.
 */

function gammaLn(x: number): number {
  // Stirling approximation for log(Gamma(x))
  if (x < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - gammaLn(1 - x);
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  let sum = c[0] ?? 0;
  for (let i = 1; i < g + 2; i++) sum += (c[i] ?? 0) / (x + i - 1);
  return 0.5 * Math.log(2 * Math.PI) + (x - 0.5) * Math.log(x + g - 0.5) - (x + g - 0.5) + Math.log(sum);
}

function digamma(x: number): number {
  if (x < 6) return digamma(x + 1) - 1 / x;
  return Math.log(x) - 1 / (2 * x) - 1 / (12 * x ** 2) + 1 / (120 * x ** 4);
}

export class BayesianGaussianMixture {
  nComponents: number;
  weightsConcentrationPrior: number;
  meanPriorPrecision: number;
  covariancePrior: number;
  maxIter: number;
  tol: number;
  randomState: number;
  converged_: boolean = false;
  nIter_: number = 0;
  weights_: Float64Array | null = null;
  means_: Float64Array[] | null = null;
  covariances_: Float64Array[] | null = null;
  responsibilities_: Float64Array[] | null = null;

  constructor(
    nComponents = 10,
    weightsConcentrationPrior = 0.001,
    meanPriorPrecision = 1.0,
    covariancePrior = 1.0,
    maxIter = 100,
    tol = 1e-4,
    randomState = 42,
  ) {
    this.nComponents = nComponents;
    this.weightsConcentrationPrior = weightsConcentrationPrior;
    this.meanPriorPrecision = meanPriorPrecision;
    this.covariancePrior = covariancePrior;
    this.maxIter = maxIter;
    this.tol = tol;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const k = this.nComponents;

    // Initialize variational parameters
    let alpha = new Float64Array(k).fill(this.weightsConcentrationPrior + n / k);
    let m = Array.from({ length: k }, () => Float64Array.from({ length: p }, () => (Math.random() - 0.5) * 2));
    let nu = new Float64Array(k).fill(p + 1);
    let R: Float64Array[] = Array.from({ length: n }, () => Float64Array.from({ length: k }, () => 1 / k));

    let prevLB = -Number.POSITIVE_INFINITY;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // M-step: update variational parameters
      const Nk = new Float64Array(k);
      const xk: Float64Array[] = Array.from({ length: k }, () => new Float64Array(p));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
          Nk[j]! += R[i]?.[j] ?? 0;
          for (let d = 0; d < p; d++) {
            (xk[j] as Float64Array)[d]! += ((R[i]?.[j] ?? 0) * (X[i]?.[d] ?? 0));
          }
        }
      }

      // Variational Bayes M-step
      alpha = Float64Array.from({ length: k }, (_, j) => this.weightsConcentrationPrior + (Nk[j] ?? 0));
      for (let j = 0; j < k; j++) {
        const nj = Nk[j] ?? 0;
        m[j] = Float64Array.from({ length: p }, (_, d) => nj > 0 ? ((xk[j] as Float64Array)[d] ?? 0) / nj : 0);
        nu[j] = p + 1 + nj;
      }

      // E-step: compute responsibilities
      const alphaSum = alpha.reduce((s, v) => s + v, 0);
      const logDirExpAlpha = Float64Array.from(alpha, (a) => digamma(a) - digamma(alphaSum));

      const newR: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        let logNormConst = -Number.POSITIVE_INFINITY;
        const logProbs = new Float64Array(k);
        for (let j = 0; j < k; j++) {
          const diff = Float64Array.from({ length: p }, (_, d) => (X[i]?.[d] ?? 0) - (m[j]?.[d] ?? 0));
          const mahal = diff.reduce((s, v, d) => s + v * v * (nu[j] ?? 1), 0);
          logProbs[j] = (logDirExpAlpha[j] ?? 0) - 0.5 * mahal - 0.5 * p * Math.log(2 * Math.PI * (this.covariancePrior / Math.max(nu[j] ?? 1, 1)));
          logNormConst = logNormConst === -Number.POSITIVE_INFINITY ? logProbs[j] ?? 0 : Math.max(logNormConst, logProbs[j] ?? 0);
        }
        let sumExp = 0;
        for (let j = 0; j < k; j++) {
          (newR[i] as Float64Array)[j] = Math.exp((logProbs[j] ?? 0) - logNormConst);
          sumExp += (newR[i] as Float64Array)[j] ?? 0;
        }
        for (let j = 0; j < k; j++) (newR[i] as Float64Array)[j]! /= Math.max(sumExp, 1e-12);
      }

      // Check ELBO (approximate)
      const lb = Array.from({ length: n }, (_, i) => {
        let s = 0;
        for (let j = 0; j < k; j++) s += (newR[i]?.[j] ?? 0) * (logDirExpAlpha[j] ?? 0);
        return s;
      }).reduce((a, b) => a + b, 0);

      R = newR;
      this.nIter_ = iter + 1;
      if (Math.abs(lb - prevLB) < this.tol) { this.converged_ = true; break; }
      prevLB = lb;
    }

    this.weights_ = Float64Array.from(alpha, (a) => a / alpha.reduce((s, v) => s + v, 0));
    this.means_ = m;
    this.covariances_ = Array.from({ length: k }, (_, j) => {
      const cov = new Float64Array(p);
      cov.fill(this.covariancePrior / Math.max(nu[j] ?? 1, 1));
      return cov;
    });
    this.responsibilities_ = R;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.weights_ || !this.means_) throw new Error("Not fitted");
    return Int32Array.from(X, (x) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let j = 0; j < this.nComponents; j++) {
        const w = this.weights_![j] ?? 0;
        const m = this.means_![j] as Float64Array;
        const diff2 = x.reduce((s, v, d) => s + (v - (m[d] ?? 0)) ** 2, 0);
        const score = Math.log(Math.max(w, 1e-15)) - diff2;
        if (score > bestScore) { bestScore = score; best = j; }
      }
      return best;
    });
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.weights_ || !this.means_) throw new Error("Not fitted");
    return X.map((x) => {
      const k = this.nComponents;
      const logProbs = Float64Array.from({ length: k }, (_, j) => {
        const w = this.weights_![j] ?? 0;
        const m = this.means_![j] as Float64Array;
        const diff2 = x.reduce((s, v, d) => s + (v - (m[d] ?? 0)) ** 2, 0);
        return Math.log(Math.max(w, 1e-15)) - diff2;
      });
      const maxLP = Math.max(...Array.from(logProbs));
      const exps = logProbs.map((v) => Math.exp(v - maxLP));
      const sumExp = exps.reduce((s, v) => s + v, 0);
      return exps.map((v) => v / Math.max(sumExp, 1e-12));
    });
  }
}

export class DirichletProcessMixture {
  truncationLevel: number;
  alpha: number;
  maxIter: number;
  tol: number;
  private _bgmm: BayesianGaussianMixture;
  weights_: Float64Array | null = null;
  means_: Float64Array[] | null = null;

  constructor(truncationLevel = 10, alpha = 1.0, maxIter = 100, tol = 1e-4) {
    this.truncationLevel = truncationLevel;
    this.alpha = alpha;
    this.maxIter = maxIter;
    this.tol = tol;
    this._bgmm = new BayesianGaussianMixture(truncationLevel, alpha / truncationLevel, 1.0, 1.0, maxIter, tol);
  }

  fit(X: Float64Array[]): this {
    this._bgmm.fit(X);
    this.weights_ = this._bgmm.weights_;
    this.means_ = this._bgmm.means_;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return this._bgmm.predict(X);
  }
}

export { gammaLn, digamma };
