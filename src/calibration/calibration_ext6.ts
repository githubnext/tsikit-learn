/**
 * Isotonic calibration and histogram-based calibration.
 */

import { BaseEstimator } from "../base.js";

export interface CalibrationBase extends BaseEstimator {
  predictProba(X: Float64Array[]): Float64Array[];
}

function isotonicRegression(y: number[]): number[] {
  const n = y.length;
  const result = [...y];
  // Pool Adjacent Violators (PAV) algorithm
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < n - 1; i++) {
      if ((result[i] ?? 0) > (result[i + 1] ?? 0)) {
        const avg = ((result[i] ?? 0) + (result[i + 1] ?? 0)) / 2;
        result[i] = avg;
        result[i + 1] = avg;
        changed = true;
      }
    }
  }
  return result;
}

export class IsotonicCalibrator extends BaseEstimator implements CalibrationBase {
  private isotonics_!: Array<{ xKnots: number[]; yKnots: number[] }>;
  private fitted_ = false;

  fit(_X: Float64Array[]): this { return this; }

  fitCalibration(
    baseEstimator: { predictProba(X: Float64Array[]): Float64Array[] },
    XCal: Float64Array[],
    yCal: Int32Array
  ): this {
    const probas = baseEstimator.predictProba(XCal);
    const nClasses = probas[0]?.length ?? 2;
    this.isotonics_ = [];
    for (let c = 0; c < nClasses; c++) {
      const scores = probas.map(p => p[c] ?? 0);
      const yBinary = Array.from(yCal).map(v => v === c ? 1 : 0);
      // Sort by score
      const order = Array.from(scores.keys()).sort((a, b) => (scores[a] ?? 0) - (scores[b] ?? 0));
      const xSorted = order.map(i => scores[i] ?? 0);
      const ySorted = order.map(i => yBinary[i] ?? 0);
      const yIso = isotonicRegression(ySorted);
      this.isotonics_.push({ xKnots: xSorted, yKnots: yIso });
    }
    this.fitted_ = true;
    return this;
  }

  private _interpolate(x: number, xKnots: number[], yKnots: number[]): number {
    if (x <= (xKnots[0] ?? 0)) return yKnots[0] ?? 0;
    if (x >= (xKnots[xKnots.length - 1] ?? 1)) return yKnots[yKnots.length - 1] ?? 1;
    let lo = 0, hi = xKnots.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if ((xKnots[mid] ?? 0) <= x) lo = mid; else hi = mid;
    }
    const t = ((xKnots[hi] ?? 0) - (xKnots[lo] ?? 0)) > 0
      ? (x - (xKnots[lo] ?? 0)) / ((xKnots[hi] ?? 0) - (xKnots[lo] ?? 0))
      : 0;
    return (yKnots[lo] ?? 0) * (1 - t) + (yKnots[hi] ?? 0) * t;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => {
      const calibrated = new Float64Array(row.length).map((_, c) =>
        this._interpolate(row[c] ?? 0, this.isotonics_[c]!.xKnots, this.isotonics_[c]!.yKnots)
      );
      const sum = calibrated.reduce((s, v) => s + v, 0);
      return sum > 0 ? new Float64Array(calibrated.map(v => v / sum)) : calibrated;
    });
  }
}

export class HistogramCalibrator {
  private bins_!: Float64Array;
  private binMeans_!: Float64Array;
  private fitted_ = false;

  constructor(private nBins = 10) {}

  fit(scores: Float64Array, yTrue: Int32Array): this {
    const min = Math.min(...scores), max = Math.max(...scores);
    this.bins_ = new Float64Array(this.nBins + 1).map((_, k) => min + k * (max - min) / this.nBins);
    const binCounts = new Float64Array(this.nBins);
    const binPositive = new Float64Array(this.nBins);
    for (let i = 0; i < scores.length; i++) {
      const b = Math.min(Math.floor(((scores[i] ?? 0) - min) / ((max - min) / this.nBins)), this.nBins - 1);
      binCounts[b] = (binCounts[b] ?? 0) + 1;
      binPositive[b] = (binPositive[b] ?? 0) + (yTrue[i] ?? 0);
    }
    this.binMeans_ = new Float64Array(this.nBins).map((_, b) =>
      (binCounts[b] ?? 0) > 0 ? (binPositive[b] ?? 0) / (binCounts[b] ?? 1) : 0.5
    );
    this.fitted_ = true;
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const min = this.bins_[0] ?? 0, max = this.bins_[this.nBins] ?? 1;
    const range = max - min;
    return new Float64Array(scores.map(s => {
      const b = Math.min(Math.floor(((s) - min) / (range / this.nBins)), this.nBins - 1);
      return this.binMeans_[b] ?? 0.5;
    }));
  }
}
