/**
 * Extended Gaussian mixture models.
 * Port of sklearn.mixture extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Mixture of von Mises distributions (for directional data).
 */
export class VonMisesMixture {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private means_: Float64Array = new Float64Array(0);
  private concentrations_: Float64Array = new Float64Array(0);
  private weights_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; maxIter?: number; tol?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-6;
  }

  private vonMisesPdf(x: number, mu: number, kappa: number): number {
    // Approximated I0(kappa) using series
    let i0 = 1; let term = 1;
    for (let k = 1; k <= 20; k++) {
      term *= (kappa / 2) * (kappa / 2) / (k * k);
      i0 += term;
    }
    return Math.exp(kappa * Math.cos(x - mu)) / (2 * Math.PI * i0);
  }

  fit(X: Float64Array): this {
    const n = X.length;
    const k = this.nComponents;

    // Initialize
    this.means_ = Float64Array.from({ length: k }, (_, i) => ((2 * Math.PI * i) / k) - Math.PI);
    this.concentrations_ = new Float64Array(k).fill(1.0);
    this.weights_ = new Float64Array(k).fill(1 / k);

    const resp = Array.from({ length: n }, () => new Float64Array(k));

    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < k; c++) {
          const val = (this.weights_[c] ?? 0) * this.vonMisesPdf(X[i] ?? 0, this.means_[c] ?? 0, this.concentrations_[c] ?? 1);
          resp[i]![c] = val;
          sum += val;
        }
        for (let c = 0; c < k; c++) {
          resp[i]![c] = sum > 0 ? (resp[i]?.[c] ?? 0) / sum : 1 / k;
        }
      }

      // M-step
      const oldMeans = Float64Array.from(this.means_);
      for (let c = 0; c < k; c++) {
        let sumR = 0; let sinSum = 0; let cosSum = 0;
        for (let i = 0; i < n; i++) {
          const r = resp[i]?.[c] ?? 0;
          sumR += r;
          sinSum += r * Math.sin(X[i] ?? 0);
          cosSum += r * Math.cos(X[i] ?? 0);
        }
        this.weights_[c] = sumR / n;
        this.means_[c] = Math.atan2(sinSum, cosSum);
        // Update concentration (approximation)
        const R = Math.sqrt(sinSum * sinSum + cosSum * cosSum) / (sumR + 1e-10);
        this.concentrations_[c] = R < 0.53 ? 2 * R + R ** 3 + 5 * R ** 5 / 6 : R >= 0.85 ? 1 / (2 - 2 * R - (1 - R) ** 2) : -0.4 + 1.39 * R + 0.43 / (1 - R);
      }

      // Check convergence
      const diff = this.means_.reduce((s, v, i) => s + Math.abs(v - (oldMeans[i] ?? 0)), 0);
      if (diff < this.tol) break;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array): Int32Array {
    if (!this.fitted) throw new NotFittedError("VonMisesMixture not fitted");
    return Int32Array.from(X, x => {
      let best = 0; let bestP = -1;
      for (let c = 0; c < this.nComponents; c++) {
        const p = (this.weights_[c] ?? 0) * this.vonMisesPdf(x, this.means_[c] ?? 0, this.concentrations_[c] ?? 1);
        if (p > bestP) { bestP = p; best = c; }
      }
      return best;
    });
  }
}

/**
 * Mixture of Poisson distributions for count data.
 */
export class PoissonMixture {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private lambdas_: Float64Array = new Float64Array(0);
  private weights_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; maxIter?: number; tol?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-6;
  }

  private poissonLogPmf(x: number, lambda: number): number {
    // log(lambda^x * e^(-lambda) / x!)
    let logFact = 0;
    for (let i = 2; i <= x; i++) logFact += Math.log(i);
    return x * Math.log(lambda + 1e-10) - lambda - logFact;
  }

  fit(X: Int32Array): this {
    const n = X.length;
    const k = this.nComponents;
    const maxX = Math.max(...X);

    // Initialize lambdas spread across range
    this.lambdas_ = Float64Array.from({ length: k }, (_, i) => ((i + 1) * maxX) / (k + 1));
    this.weights_ = new Float64Array(k).fill(1 / k);

    const resp = Array.from({ length: n }, () => new Float64Array(k));

    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step
      for (let i = 0; i < n; i++) {
        let logSum = Number.NEGATIVE_INFINITY;
        const logR = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          logR[c] = Math.log(this.weights_[c] ?? 1e-10) + this.poissonLogPmf(X[i] ?? 0, this.lambdas_[c] ?? 1);
          logSum = logSum === Number.NEGATIVE_INFINITY ? (logR[c] ?? 0) : Math.log(Math.exp(logSum) + Math.exp(logR[c] ?? Number.NEGATIVE_INFINITY));
        }
        for (let c = 0; c < k; c++) {
          resp[i]![c] = Math.exp((logR[c] ?? 0) - logSum);
        }
      }

      // M-step
      const oldLambdas = Float64Array.from(this.lambdas_);
      for (let c = 0; c < k; c++) {
        let sumR = 0; let sumRX = 0;
        for (let i = 0; i < n; i++) {
          sumR += resp[i]?.[c] ?? 0;
          sumRX += (resp[i]?.[c] ?? 0) * (X[i] ?? 0);
        }
        this.weights_[c] = sumR / n;
        this.lambdas_[c] = sumR > 0 ? sumRX / sumR : 1;
      }

      const diff = this.lambdas_.reduce((s, v, i) => s + Math.abs(v - (oldLambdas[i] ?? 0)), 0);
      if (diff < this.tol) break;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Int32Array): Int32Array {
    if (!this.fitted) throw new NotFittedError("PoissonMixture not fitted");
    return Int32Array.from(X, x => {
      let best = 0; let bestLogP = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < this.nComponents; c++) {
        const logP = Math.log(this.weights_[c] ?? 1e-10) + this.poissonLogPmf(x, this.lambdas_[c] ?? 1);
        if (logP > bestLogP) { bestLogP = logP; best = c; }
      }
      return best;
    });
  }
}

/**
 * Skew-normal mixture model.
 */
export class SkewNormalMixture {
  private nComponents: number;
  private maxIter: number;
  private means_: Float64Array = new Float64Array(0);
  private stds_: Float64Array = new Float64Array(0);
  private skews_: Float64Array = new Float64Array(0);
  private weights_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; maxIter?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 100;
  }

  private normalPdf(x: number): number {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  }

  private normalCdf(x: number): number {
    // Approximation
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = this.normalPdf(x);
    const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - p : p;
  }

  private skewNormalPdf(x: number, mu: number, sigma: number, alpha: number): number {
    const z = (x - mu) / sigma;
    return (2 / sigma) * this.normalPdf(z) * this.normalCdf(alpha * z);
  }

  fit(X: Float64Array): this {
    const n = X.length;
    const k = this.nComponents;

    // Initialize using moment estimates
    const sorted = Float64Array.from(X).sort();
    this.means_ = Float64Array.from({ length: k }, (_, i) => sorted[Math.floor((i + 0.5) * n / k)] ?? 0);
    this.stds_ = new Float64Array(k).fill(1.0);
    this.skews_ = new Float64Array(k).fill(0.0);
    this.weights_ = new Float64Array(k).fill(1 / k);

    const resp = Array.from({ length: n }, () => new Float64Array(k));

    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < k; c++) {
          const p = (this.weights_[c] ?? 0) * this.skewNormalPdf(X[i] ?? 0, this.means_[c] ?? 0, this.stds_[c] ?? 1, this.skews_[c] ?? 0);
          resp[i]![c] = Math.max(p, 1e-300);
          sum += resp[i]?.[c] ?? 0;
        }
        for (let c = 0; c < k; c++) resp[i]![c] = (resp[i]?.[c] ?? 0) / sum;
      }

      // M-step (simplified: update means and weights, keep stds/skews fixed for stability)
      for (let c = 0; c < k; c++) {
        let sumR = 0; let sumRX = 0;
        for (let i = 0; i < n; i++) {
          sumR += resp[i]?.[c] ?? 0;
          sumRX += (resp[i]?.[c] ?? 0) * (X[i] ?? 0);
        }
        this.weights_[c] = sumR / n;
        this.means_[c] = sumR > 0 ? sumRX / sumR : this.means_[c] ?? 0;
        // Update std
        let sumRX2 = 0;
        for (let i = 0; i < n; i++) {
          const d = (X[i] ?? 0) - (this.means_[c] ?? 0);
          sumRX2 += (resp[i]?.[c] ?? 0) * d * d;
        }
        this.stds_[c] = sumR > 0 ? Math.sqrt(Math.max(sumRX2 / sumR, 1e-6)) : 1;
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array): Int32Array {
    if (!this.fitted) throw new NotFittedError("SkewNormalMixture not fitted");
    return Int32Array.from(X, x => {
      let best = 0; let bestP = -1;
      for (let c = 0; c < this.nComponents; c++) {
        const p = (this.weights_[c] ?? 0) * this.skewNormalPdf(x, this.means_[c] ?? 0, this.stds_[c] ?? 1, this.skews_[c] ?? 0);
        if (p > bestP) { bestP = p; best = c; }
      }
      return best;
    });
  }
}
