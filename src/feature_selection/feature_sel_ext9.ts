/**
 * Feature selection extensions: SelectFwe, GenericUnivariateSelect extensions.
 * Mirrors sklearn.feature_selection advanced selectors.
 */

import { BaseEstimator } from "../base.js";

/** Chi-squared test for feature selection. */
export function chi2Ext(
  X: Float64Array[],
  y: Int32Array,
): { statistics: Float64Array; pValues: Float64Array } {
  const n = X.length;
  const nf = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
  const statistics = new Float64Array(nf);
  const pValues = new Float64Array(nf);

  for (let k = 0; k < nf; k++) {
    // Build contingency table
    const observed: number[][] = Array.from({ length: 2 }, () => new Array<number>(classes.length).fill(0));
    const colVals = Array.from({ length: n }, (_, i) => X[i]?.[k] ?? 0);
    const median = colVals.slice().sort((a, b) => a - b)[Math.floor(n / 2)] ?? 0;
    for (let i = 0; i < n; i++) {
      const row = (colVals[i] ?? 0) >= median ? 1 : 0;
      const col = classes.indexOf(y[i] ?? 0);
      if (col >= 0) observed[row]![col] = (observed[row]![col] ?? 0) + 1;
    }
    const rowSums = observed.map((r) => r.reduce((a, b) => a + b, 0));
    const colSums = classes.map((_, c) => observed.reduce((s, r) => s + (r[c] ?? 0), 0));
    let chi2 = 0;
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < classes.length; c++) {
        const expected = (rowSums[r] ?? 0) * (colSums[c] ?? 0) / n;
        if (expected > 0) {
          const obs = observed[r]?.[c] ?? 0;
          chi2 += (obs - expected) ** 2 / expected;
        }
      }
    }
    statistics[k] = chi2;
    // Approximate p-value using chi2 distribution (df = classes.length - 1)
    pValues[k] = 1 - _chi2CDF(chi2, classes.length - 1);
  }
  return { statistics, pValues };

  function _chi2CDF(x: number, df: number): number {
    if (x <= 0) return 0;
    // Regularized incomplete gamma function approximation
    return _gammainc(df / 2, x / 2);
  }

  function _gammainc(a: number, x: number): number {
    if (x <= 0) return 0;
    let sum = 1 / a;
    let term = 1 / a;
    for (let n = 1; n <= 100; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < 1e-10) break;
    }
    return Math.min(1, sum * Math.exp(-x + a * Math.log(x) - _lgamma(a)));
  }

  function _lgamma(z: number): number {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = z, x = z, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const ci of c) { y++; ser += ci / y; }
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
}

export interface SelectFweParams {
  alpha?: number;
  score_func?: ((X: Float64Array[], y: Int32Array) => { statistics: Float64Array; pValues: Float64Array }) | null;
}

/** SelectFwe: select features based on family-wise error rate. */
export class SelectFwe extends BaseEstimator {
  alpha: number;
  scores_: Float64Array = new Float64Array(0);
  pvalues_: Float64Array = new Float64Array(0);
  selected_: boolean[] = [];

  constructor(params: SelectFweParams = {}) {
    super();
    this.alpha = params.alpha ?? 0.05;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const nf = X[0]?.length ?? 0;
    // F-statistic based selection
    const n = X.length;
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    const k = classes.length;
    this.scores_ = new Float64Array(nf);
    this.pvalues_ = new Float64Array(nf);
    for (let f = 0; f < nf; f++) {
      const overall = Array.from({ length: n }, (_, i) => X[i]?.[f] ?? 0);
      let overallMean = 0;
      for (const v of overall) overallMean += v;
      overallMean /= n;
      let bss = 0, wss = 0;
      for (const c of classes) {
        const group = overall.filter((_, i) => (y[i] ?? -1) === c);
        const gm = group.reduce((s, v) => s + v, 0) / (group.length || 1);
        bss += group.length * (gm - overallMean) ** 2;
        for (const v of group) wss += (v - gm) ** 2;
      }
      const fStat = (bss / Math.max(k - 1, 1)) / (wss / Math.max(n - k, 1));
      this.scores_[f] = fStat;
      // Approximate p-value
      this.pvalues_[f] = Math.exp(-0.5 * fStat);
    }
    // Bonferroni correction
    const threshold = this.alpha / nf;
    this.selected_ = Array.from(this.pvalues_).map((p) => p <= threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const indices = this.selected_.map((s, i) => s ? i : -1).filter((i) => i >= 0);
    return X.map((xi) => new Float64Array(indices.map((i) => xi[i] ?? 0)));
  }

  fit_transform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export interface VarianceThresholdExtParams {
  threshold?: number;
}

/** VarianceThreshold: remove features with low variance. */
export class VarianceThresholdExt extends BaseEstimator {
  threshold: number;
  variances_: Float64Array = new Float64Array(0);
  selected_: boolean[] = [];

  constructor(params: VarianceThresholdExtParams = {}) {
    super();
    this.threshold = params.threshold ?? 0.0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    this.variances_ = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      let mean = 0;
      for (let i = 0; i < n; i++) mean += X[i]?.[k] ?? 0;
      mean /= n;
      let variance = 0;
      for (let i = 0; i < n; i++) variance += ((X[i]?.[k] ?? 0) - mean) ** 2;
      this.variances_[k] = variance / n;
    }
    this.selected_ = Array.from(this.variances_).map((v) => v > this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const indices = this.selected_.map((s, i) => s ? i : -1).filter((i) => i >= 0);
    return X.map((xi) => new Float64Array(indices.map((i) => xi[i] ?? 0)));
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  get_support(indices = false): number[] | boolean[] {
    if (indices) return this.selected_.map((s, i) => s ? i : -1).filter((i) => i >= 0);
    return this.selected_;
  }
}

/** SelectPercentile: select features by percentile of highest scores. */
export class SelectPercentileExt extends BaseEstimator {
  percentile: number;
  scores_: Float64Array = new Float64Array(0);
  selected_: boolean[] = [];

  constructor(percentile = 50) {
    super();
    this.percentile = percentile;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const nf = X[0]?.length ?? 0;
    const n = X.length;
    const classes = [...new Set(Array.from(y))];
    const k = classes.length;
    this.scores_ = new Float64Array(nf);
    for (let f = 0; f < nf; f++) {
      let overallMean = 0;
      for (let i = 0; i < n; i++) overallMean += X[i]?.[f] ?? 0;
      overallMean /= n;
      let bss = 0, wss = 0;
      for (const c of classes) {
        const group = Array.from({ length: n }, (_, i) => (y[i] ?? -1) === c ? (X[i]?.[f] ?? 0) : null).filter((v) => v !== null) as number[];
        const gm = group.reduce((s, v) => s + v, 0) / (group.length || 1);
        bss += group.length * (gm - overallMean) ** 2;
        for (const v of group) wss += (v - gm) ** 2;
      }
      this.scores_[f] = (bss / Math.max(k - 1, 1)) / (wss / Math.max(n - k, 1) || 1);
    }
    const sorted = Array.from(this.scores_).sort((a, b) => b - a);
    const cutoff = sorted[Math.floor((1 - this.percentile / 100) * nf)] ?? 0;
    this.selected_ = Array.from(this.scores_).map((s) => s >= cutoff);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const indices = this.selected_.map((s, i) => s ? i : -1).filter((i) => i >= 0);
    return X.map((xi) => new Float64Array(indices.map((i) => xi[i] ?? 0)));
  }

  fit_transform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
