/**
 * Additional feature selection: GenericUnivariateSelect, VarianceThreshold extensions.
 * Mirrors sklearn.feature_selection extras.
 */

import { NotFittedError } from "../exceptions.js";

export function chiSquared(
  X: Float64Array[],
  y: Int32Array,
): { chi2: Float64Array; pValues: Float64Array } {
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
  const nClasses = classes.length;

  const chi2 = new Float64Array(nFeatures);
  const pValues = new Float64Array(nFeatures).fill(1);

  for (let j = 0; j < nFeatures; j++) {
    // Build observed contingency table
    const classCounts = new Float64Array(nClasses);
    const featureSum = new Float64Array(nClasses);
    let totalSum = 0;

    for (let i = 0; i < nSamples; i++) {
      const cIdx = classes.indexOf(y[i] ?? 0);
      if (cIdx >= 0) {
        classCounts[cIdx] = (classCounts[cIdx] ?? 0) + 1;
        featureSum[cIdx] = (featureSum[cIdx] ?? 0) + (X[i]?.[j] ?? 0);
      }
      totalSum += X[i]?.[j] ?? 0;
    }

    // Chi-squared statistic
    let stat = 0;
    for (let c = 0; c < nClasses; c++) {
      const expected = ((classCounts[c] ?? 0) * totalSum) / nSamples;
      if (expected > 0) {
        stat += ((featureSum[c] ?? 0) - expected) ** 2 / expected;
      }
    }
    chi2[j] = stat;
    // Approximate p-value using chi-sq distribution CDF (df = nClasses - 1)
    const df = nClasses - 1;
    if (df > 0) {
      pValues[j] = 1 - incompletGamma(df / 2, stat / 2);
    }
  }

  return { chi2, pValues };
}

function incompletGamma(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x > 1 + a) {
    // Use continued fraction
    let f = 1 / x;
    let c = f;
    let d = 0;
    for (let i = 0; i < 100; i++) {
      const an = (i % 2 === 0 ? -(a + i / 2) * x : (1 + i / 2) * x);
      d = 1 + an * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + an / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      f *= c * d;
      if (Math.abs(c * d - 1) < 1e-7) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - logGamma(a)) * f;
  }
  // Series expansion
  let sum = 1 / a;
  let term = 1 / a;
  for (let i = 1; i < 100; i++) {
    term *= x / (a + i);
    sum += term;
    if (Math.abs(term) < 1e-10) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

function logGamma(x: number): number {
  // Stirling approximation
  if (x <= 0) return 0;
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let result = 0.99999999999980993;
  const c = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  for (let i = 0; i < c.length; i++) result += (c[i] ?? 0) / (x + i + 1);
  const t = x + c.length - 0.5;
  return Math.log(2 * Math.PI) / 2 + Math.log(result) + (x + 0.5) * Math.log(t) - t;
}

export class SelectPercentile {
  scoreFunc: (X: Float64Array[], y: Int32Array) => { scores?: Float64Array; pValues?: Float64Array };
  percentile: number;

  scores_: Float64Array | null = null;
  pValues_: Float64Array | null = null;
  private selectedMask_: Uint8Array | null = null;

  constructor(
    scoreFunc: (X: Float64Array[], y: Int32Array) => { scores?: Float64Array; pValues?: Float64Array },
    percentile = 50,
  ) {
    this.scoreFunc = scoreFunc;
    this.percentile = percentile;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const result = this.scoreFunc(X, y);
    this.scores_ = result.scores ?? result.pValues ?? new Float64Array(X[0]?.length ?? 0);
    this.pValues_ = result.pValues ?? null;

    const nFeatures = this.scores_.length;
    const threshold = this._computeThreshold(Array.from(this.scores_));
    this.selectedMask_ = new Uint8Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      if ((this.scores_[j] ?? 0) >= threshold) this.selectedMask_[j] = 1;
    }
    return this;
  }

  private _computeThreshold(scores: number[]): number {
    const sorted = scores.slice().sort((a, b) => a - b);
    const idx = Math.floor((1 - this.percentile / 100) * sorted.length);
    return sorted[Math.max(0, idx)] ?? 0;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedMask_) throw new NotFittedError("SelectPercentile is not fitted");
    const selected = Array.from(this.selectedMask_).map((v, i) => ({ v, i })).filter((x) => x.v).map((x) => x.i);
    return X.map((row) => {
      const out = new Float64Array(selected.length);
      for (let k = 0; k < selected.length; k++) out[k] = row[selected[k] ?? 0] ?? 0;
      return out;
    });
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupport(): Uint8Array {
    if (!this.selectedMask_) throw new NotFittedError("SelectPercentile is not fitted");
    return this.selectedMask_;
  }
}

export class VarianceThresholdExt {
  threshold: number;
  variances_: Float64Array | null = null;
  private selectedMask_: Uint8Array | null = null;

  constructor(threshold = 0.0) {
    this.threshold = threshold;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const means = new Float64Array(nFeatures);
    const vars = new Float64Array(nFeatures);

    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) means[j] = (means[j] ?? 0) + (row[j] ?? 0);
    }
    for (let j = 0; j < nFeatures; j++) means[j] = (means[j] ?? 0) / n;

    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        vars[j] = (vars[j] ?? 0) + ((row[j] ?? 0) - (means[j] ?? 0)) ** 2;
      }
    }
    for (let j = 0; j < nFeatures; j++) vars[j] = (vars[j] ?? 0) / n;

    this.variances_ = vars;
    this.selectedMask_ = new Uint8Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      if ((vars[j] ?? 0) > this.threshold) this.selectedMask_[j] = 1;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedMask_) throw new NotFittedError("VarianceThresholdExt is not fitted");
    const selected = Array.from(this.selectedMask_).map((v, i) => ({ v, i })).filter((x) => x.v).map((x) => x.i);
    return X.map((row) => {
      const out = new Float64Array(selected.length);
      for (let k = 0; k < selected.length; k++) out[k] = row[selected[k] ?? 0] ?? 0;
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
