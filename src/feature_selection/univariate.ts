/**
 * Feature selection utilities.
 * Mirrors sklearn.feature_selection: SelectKBest, SelectPercentile,
 * VarianceThreshold, chi2, f_classif, f_regression.
 */

import { NotFittedError } from "../exceptions.js";

export type ScoreFn = (X: Float64Array[], y: Float64Array) => [Float64Array, Float64Array];

/** F-score for classification (ANOVA F-test). */
export function fClassif(X: Float64Array[], y: Float64Array): [Float64Array, Float64Array] {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const uniqueClasses = Array.from(new Set(Array.from(y)));
  const k = uniqueClasses.length;

  const fScores = new Float64Array(p);
  const pValues = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const overall = Array.from(X).map((xi) => xi[j] ?? 0);
    const grandMean = overall.reduce((a, b) => a + b, 0) / n;

    let ssBetween = 0;
    let ssWithin = 0;

    for (const cls of uniqueClasses) {
      const groupVals = Array.from(y)
        .map((yi, i) => (yi === cls ? (X[i] ?? new Float64Array(p))[j] ?? 0 : null))
        .filter((v): v is number => v !== null);
      const groupMean = groupVals.reduce((a, b) => a + b, 0) / (groupVals.length || 1);
      ssBetween += groupVals.length * (groupMean - grandMean) ** 2;
      ssWithin += groupVals.reduce((s, v) => s + (v - groupMean) ** 2, 0);
    }

    const dfBetween = k - 1;
    const dfWithin = n - k;
    const msBetween = dfBetween > 0 ? ssBetween / dfBetween : 0;
    const msWithin = dfWithin > 0 ? ssWithin / dfWithin : 1e-10;

    fScores[j] = msWithin > 0 ? msBetween / msWithin : 0;
    // Approximate p-value (simplified: not exact F distribution CDF)
    pValues[j] = Math.exp(-(fScores[j] ?? 0) / 2);
  }

  return [fScores, pValues];
}

/** F-score for regression. */
export function fRegression(X: Float64Array[], y: Float64Array): [Float64Array, Float64Array] {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const yMean = Array.from(y).reduce((a, b) => a + b, 0) / n;

  const fScores = new Float64Array(p);
  const pValues = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const xVals = Array.from(X).map((xi) => xi[j] ?? 0);
    const xMean = xVals.reduce((a, b) => a + b, 0) / n;

    let ssXY = 0;
    let ssXX = 0;
    for (let i = 0; i < n; i++) {
      const dx = (xVals[i] ?? 0) - xMean;
      ssXY += dx * ((y[i] ?? 0) - yMean);
      ssXX += dx ** 2;
    }

    if (ssXX === 0) {
      fScores[j] = 0;
      pValues[j] = 1;
      continue;
    }

    const slope = ssXY / ssXX;
    const intercept = yMean - slope * xMean;

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < n; i++) {
      const pred = slope * (xVals[i] ?? 0) + intercept;
      ssRes += ((y[i] ?? 0) - pred) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }

    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    fScores[j] = r2 > 0 && r2 < 1 ? (r2 / 1) / ((1 - r2) / (n - 2)) : 0;
    pValues[j] = Math.exp(-(fScores[j] ?? 0) / 2);
  }

  return [fScores, pValues];
}

/** Chi-squared test statistic for non-negative features. */
export function chi2(X: Float64Array[], y: Float64Array): [Float64Array, Float64Array] {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const uniqueClasses = Array.from(new Set(Array.from(y)));

  const chiScores = new Float64Array(p);
  const pValues = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    let chi = 0;
    for (const cls of uniqueClasses) {
      const classIdx = Array.from(y).map((yi, i) => yi === cls ? i : -1).filter(i => i >= 0);
      const expected = classIdx.length / n;
      for (let i of classIdx) {
        const obs = (X[i] ?? new Float64Array(p))[j] ?? 0;
        const exp = expected * Array.from(X).reduce((s, xi) => s + (xi[j] ?? 0), 0) / n;
        if (exp > 0) chi += (obs - exp) ** 2 / exp;
      }
    }
    chiScores[j] = chi;
    pValues[j] = Math.exp(-chi / 2);
  }

  return [chiScores, pValues];
}

export class SelectKBest {
  k: number;
  scoreFunc: ScoreFn;

  scores_: Float64Array | null = null;
  pValues_: Float64Array | null = null;
  selectedIndices_: number[] | null = null;

  constructor(
    scoreFunc: ScoreFn = fClassif,
    options: { k?: number } = {},
  ) {
    this.scoreFunc = scoreFunc;
    this.k = options.k ?? 10;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const [scores, pValues] = this.scoreFunc(X, y);
    this.scores_ = scores;
    this.pValues_ = pValues;

    const k = Math.min(this.k, scores.length);
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    this.selectedIndices_ = indices.slice(0, k).sort((a, b) => a - b);

    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.selectedIndices_ === null) throw new NotFittedError("SelectKBest");
    const sel = this.selectedIndices_;
    return X.map((xi) => new Float64Array(sel.map((j) => xi[j] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupport(): boolean[] {
    if (this.selectedIndices_ === null || this.scores_ === null)
      throw new NotFittedError("SelectKBest");
    const n = this.scores_.length;
    const selected = new Set(this.selectedIndices_);
    return Array.from({ length: n }, (_, i) => selected.has(i));
  }
}

export class SelectPercentile {
  percentile: number;
  scoreFunc: ScoreFn;

  scores_: Float64Array | null = null;
  selectedIndices_: number[] | null = null;

  constructor(
    scoreFunc: ScoreFn = fClassif,
    options: { percentile?: number } = {},
  ) {
    this.scoreFunc = scoreFunc;
    this.percentile = options.percentile ?? 10;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const [scores] = this.scoreFunc(X, y);
    this.scores_ = scores;

    const k = Math.max(1, Math.round((this.percentile / 100) * scores.length));
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    this.selectedIndices_ = indices.slice(0, k).sort((a, b) => a - b);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.selectedIndices_ === null) throw new NotFittedError("SelectPercentile");
    const sel = this.selectedIndices_;
    return X.map((xi) => new Float64Array(sel.map((j) => xi[j] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class VarianceThreshold {
  threshold: number;

  variances_: Float64Array | null = null;
  selectedIndices_: number[] | null = null;

  constructor(options: { threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0.0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    const variances = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let mean = 0;
      for (const xi of X) mean += xi[j] ?? 0;
      mean /= n;
      let variance = 0;
      for (const xi of X) variance += ((xi[j] ?? 0) - mean) ** 2;
      variances[j] = variance / n;
    }

    this.variances_ = variances;
    this.selectedIndices_ = Array.from({ length: p }, (_, i) => i).filter(
      (i) => (variances[i] ?? 0) > this.threshold,
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.selectedIndices_ === null) throw new NotFittedError("VarianceThreshold");
    const sel = this.selectedIndices_;
    return X.map((xi) => new Float64Array(sel.map((j) => xi[j] ?? 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
