/**
 * Feature selection extensions: Stability selection, Boruta, Information gain.
 * Mirrors sklearn.feature_selection additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Compute information gain for each feature with respect to target. */
export function informationGain(X: Float64Array[], y: Int32Array, nBins = 10): Float64Array {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const scores = new Float64Array(d);

  const classCount = new Map<number, number>();
  for (let i = 0; i < n; i++) classCount.set(y[i] ?? 0, (classCount.get(y[i] ?? 0) ?? 0) + 1);
  const Hy = [...classCount.values()].reduce((e, c) => {
    const p = c / n;
    return e - p * Math.log2(p || 1e-12);
  }, 0);

  for (let f = 0; f < d; f++) {
    const vals = X.map(row => row[f] ?? 0);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const binSize = (max - min) / nBins || 1;

    const binCounts = new Array<Map<number, number>>(nBins).fill(null!).map(() => new Map<number, number>());
    const binTotals = new Array(nBins).fill(0) as number[];

    for (let i = 0; i < n; i++) {
      const bin = Math.min(nBins - 1, Math.floor(((vals[i] ?? 0) - min) / binSize));
      const cls = y[i] ?? 0;
      binCounts[bin]!.set(cls, (binCounts[bin]!.get(cls) ?? 0) + 1);
      binTotals[bin] = (binTotals[bin] ?? 0) + 1;
    }

    let Hcond = 0;
    for (let b = 0; b < nBins; b++) {
      const total = binTotals[b] ?? 0;
      if (total === 0) continue;
      const p_b = total / n;
      let Hb = 0;
      for (const cnt of binCounts[b]!.values()) {
        const p = cnt / total;
        Hb -= p * Math.log2(p || 1e-12);
      }
      Hcond += p_b * Hb;
    }
    scores[f] = Hy - Hcond;
  }
  return scores;
}

export interface StabilitySelectionParams {
  n_bootstrap?: number;
  threshold?: number;
  sample_fraction?: number;
}

/** Stability Selection: identify stable features via bootstrapped feature selection. */
export class StabilitySelection extends BaseEstimator {
  n_bootstrap: number;
  threshold: number;
  sample_fraction: number;
  stability_scores_: Float64Array = new Float64Array(0);
  selected_: boolean[] = [];

  constructor(params: StabilitySelectionParams = {}) {
    super();
    this.n_bootstrap = params.n_bootstrap ?? 50;
    this.threshold = params.threshold ?? 0.6;
    this.sample_fraction = params.sample_fraction ?? 0.5;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const nSample = Math.floor(n * this.sample_fraction);
    const selectionCounts = new Float64Array(d);

    for (let b = 0; b < this.n_bootstrap; b++) {
      // Bootstrap sample
      const idx: number[] = [];
      while (idx.length < nSample) idx.push(Math.floor(Math.random() * n));
      const Xb = idx.map(i => X[i]!);
      const yb = new Int32Array(idx.map(i => y[i] ?? 0));

      // Simple ANOVA F-score as base selector
      const scores = this._scoreFeatures(Xb, yb);
      const thresh = sorted(scores)[Math.floor(d * 0.5)] ?? 0;
      for (let f = 0; f < d; f++) {
        if ((scores[f] ?? 0) >= thresh) selectionCounts[f] = (selectionCounts[f] ?? 0) + 1;
      }
    }

    this.stability_scores_ = selectionCounts.map(c => c / this.n_bootstrap);
    this.selected_ = Array.from(this.stability_scores_).map(s => s >= this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => new Float64Array(
      Array.from(row).filter((_, f) => this.selected_[f] ?? false),
    ));
  }

  private _scoreFeatures(X: Float64Array[], y: Int32Array): Float64Array {
    const d = X[0]?.length ?? 0;
    const scores = new Float64Array(d);
    const classes = [...new Set(Array.from(y))];
    const n = X.length;
    for (let f = 0; f < d; f++) {
      const vals = X.map(row => row[f] ?? 0);
      const grand = vals.reduce((a, b) => a + b, 0) / n;
      let ssBetween = 0;
      let ssWithin = 0;
      for (const cls of classes) {
        const clsVals = vals.filter((_, i) => y[i] === cls);
        if (clsVals.length === 0) continue;
        const clsMean = clsVals.reduce((a, b) => a + b, 0) / clsVals.length;
        ssBetween += clsVals.length * (clsMean - grand) ** 2;
        ssWithin += clsVals.reduce((s, v) => s + (v - clsMean) ** 2, 0);
      }
      scores[f] = ssWithin > 0 ? ssBetween / ssWithin : 0;
    }
    return scores;
  }
}

function sorted(arr: Float64Array): Float64Array {
  return new Float64Array([...arr].sort((a, b) => a - b));
}

/** Gain ratio feature selection (information gain / split info). */
export function gainRatio(X: Float64Array[], y: Int32Array, nBins = 10): Float64Array {
  const ig = informationGain(X, y, nBins);
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const gr = new Float64Array(d);
  for (let f = 0; f < d; f++) {
    const vals = X.map(row => row[f] ?? 0);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const binSize = (max - min) / nBins || 1;
    const binCounts = new Float64Array(nBins);
    for (const v of vals) {
      const bin = Math.min(nBins - 1, Math.floor((v - min) / binSize));
      binCounts[bin] = (binCounts[bin] ?? 0) + 1;
    }
    let splitInfo = 0;
    for (const c of binCounts) {
      if (c === 0) continue;
      const p = c / n;
      splitInfo -= p * Math.log2(p);
    }
    gr[f] = splitInfo > 0 ? (ig[f] ?? 0) / splitInfo : 0;
  }
  return gr;
}

/** Select features with variance above a threshold. */
export class HighVarianceSelector extends BaseEstimator {
  threshold: number;
  variances_: Float64Array = new Float64Array(0);
  selected_: boolean[] = [];

  constructor(threshold = 0.01) {
    super();
    this.threshold = threshold;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const variances = new Float64Array(d);
    for (let f = 0; f < d; f++) {
      const vals = X.map(row => row[f] ?? 0);
      const mean = vals.reduce((a, b) => a + b, 0) / n;
      variances[f] = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    }
    this.variances_ = variances;
    this.selected_ = Array.from(variances).map(v => v >= this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => new Float64Array(Array.from(row).filter((_, f) => this.selected_[f] ?? false)));
  }
}
