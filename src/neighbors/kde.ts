/**
 * Kernel Density Estimation.
 * Mirrors sklearn.neighbors.KernelDensity.
 */

import { NotFittedError } from "../exceptions.js";
import { BaseEstimator } from "../base.js";

export type KernelType =
  | "gaussian"
  | "tophat"
  | "epanechnikov"
  | "exponential"
  | "linear"
  | "cosine";

export interface KernelDensityParams {
  bandwidth?: number | "scott" | "silverman";
  algorithm?: "ball_tree" | "kd_tree" | "auto";
  kernel?: KernelType;
  metric?: string;
  atol?: number;
  rtol?: number;
  breadthFirst?: boolean;
  leafSize?: number;
}

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

function kernelVal(kernel: KernelType, d: number, h: number): number {
  const u = d / h;
  switch (kernel) {
    case "gaussian":
      return Math.exp(-0.5 * u * u);
    case "tophat":
      return u <= 1 ? 1.0 : 0.0;
    case "epanechnikov":
      return u <= 1 ? 0.75 * (1 - u * u) : 0.0;
    case "exponential":
      return Math.exp(-u);
    case "linear":
      return u <= 1 ? 1 - u : 0.0;
    case "cosine":
      return u <= 1 ? Math.cos((Math.PI / 2) * u) : 0.0;
  }
}

/**
 * Kernel Density Estimation.
 *
 * Mirrors sklearn.neighbors.KernelDensity.
 */
export class KernelDensity extends BaseEstimator {
  readonly bandwidthParam: number | "scott" | "silverman";
  readonly kernel: KernelType;
  readonly algorithm: string;
  readonly metric: string;
  readonly atol: number;
  readonly rtol: number;
  readonly leafSize: number;

  bandwidth_: number | null = null;
  fitX_: Float64Array[] | null = null;
  nFeaturesIn_: number | null = null;

  constructor(params: KernelDensityParams = {}) {
    super();
    this.bandwidthParam = params.bandwidth ?? 1.0;
    this.kernel = params.kernel ?? "gaussian";
    this.algorithm = params.algorithm ?? "auto";
    this.metric = params.metric ?? "euclidean";
    this.atol = params.atol ?? 0;
    this.rtol = params.rtol ?? 0;
    this.leafSize = params.leafSize ?? 40;
  }

  private _computeBandwidth(X: Float64Array[]): number {
    if (typeof this.bandwidthParam === "number") return this.bandwidthParam;
    const n = X.length;
    const p = X[0]?.length ?? 1;
    // Compute std per feature, average
    let meanStd = 0;
    for (let j = 0; j < p; j++) {
      let sum = 0;
      let sum2 = 0;
      for (let i = 0; i < n; i++) {
        const v = X[i]?.[j] ?? 0;
        sum += v;
        sum2 += v * v;
      }
      const std = Math.sqrt(Math.max(0, sum2 / n - (sum / n) ** 2));
      meanStd += std;
    }
    meanStd /= p;
    if (this.bandwidthParam === "scott") {
      return meanStd * Math.pow(n, -1 / (p + 4));
    }
    // silverman
    return meanStd * Math.pow(n * (p + 2) / 4, -1 / (p + 4));
  }

  fit(X: Float64Array[]): this {
    this.fitX_ = X;
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    this.bandwidth_ = this._computeBandwidth(X);
    return this;
  }

  /**
   * Score samples: log-density of each sample.
   */
  scoresSamples(X: Float64Array[]): Float64Array {
    if (this.fitX_ === null || this.bandwidth_ === null)
      throw new NotFittedError("KernelDensity");
    const trainX = this.fitX_;
    const h = this.bandwidth_;
    const kernel = this.kernel;
    const n = trainX.length;
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let logSum = -Number.POSITIVE_INFINITY;
      const xi = X[i] ?? new Float64Array(0);
      for (let j = 0; j < n; j++) {
        const d = euclidean(xi, trainX[j] ?? new Float64Array(0));
        const k = kernelVal(kernel, d, h);
        if (k > 0) {
          const logK = Math.log(k);
          if (logSum === -Number.POSITIVE_INFINITY) {
            logSum = logK;
          } else {
            const m = Math.max(logSum, logK);
            logSum = m + Math.log(Math.exp(logSum - m) + Math.exp(logK - m));
          }
        }
      }
      out[i] = logSum - Math.log(n);
    }
    return out;
  }

  score(X: Float64Array[]): number {
    const logDensities = this.scoresSamples(X);
    let sum = 0;
    for (let i = 0; i < logDensities.length; i++) sum += logDensities[i] ?? 0;
    return sum;
  }

  sample(nSamples = 1, randomState?: number): Float64Array[] {
    if (this.fitX_ === null || this.bandwidth_ === null)
      throw new NotFittedError("KernelDensity");
    const trainX = this.fitX_;
    const n = trainX.length;
    const p = this.nFeaturesIn_ ?? 1;
    const h = this.bandwidth_;
    let seed = randomState ?? 42;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 4294967296;
    };
    const randNorm = () => {
      const u1 = 1 - rand();
      const u2 = rand();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const result: Float64Array[] = [];
    for (let i = 0; i < nSamples; i++) {
      const baseIdx = Math.floor(rand() * n);
      const base = trainX[baseIdx] ?? new Float64Array(p);
      const sample = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        sample[j] = (base[j] ?? 0) + h * randNorm();
      }
      result.push(sample);
    }
    return result;
  }
}
