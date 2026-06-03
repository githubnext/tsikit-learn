/**
 * Additional calibration methods: Venn prediction, conformal prediction.
 * Port of sklearn.calibration extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Histogram binning calibration method.
 */
export class HistogramBinningCalibration {
  private nBins: number;
  private binEdges_: Float64Array = new Float64Array(0);
  private binMeans_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nBins?: number } = {}) {
    this.nBins = options.nBins ?? 10;
  }

  fit(scores: Float64Array, labels: Int32Array): this {
    const n = scores.length;
    const edges = new Float64Array(this.nBins + 1);
    for (let i = 0; i <= this.nBins; i++) {
      edges[i] = i / this.nBins;
    }
    this.binEdges_ = edges;

    const binMeans = new Float64Array(this.nBins);
    const binCounts = new Int32Array(this.nBins);

    for (let i = 0; i < n; i++) {
      const s = scores[i] ?? 0;
      const bin = Math.min(Math.floor(s * this.nBins), this.nBins - 1);
      binMeans[bin] += labels[i] ?? 0;
      binCounts[bin]++;
    }

    for (let b = 0; b < this.nBins; b++) {
      binMeans[b] = binCounts[b] > 0 ? (binMeans[b] ?? 0) / binCounts[b] : (edges[b]! + edges[b + 1]!) / 2;
    }

    this.binMeans_ = binMeans;
    this.fitted = true;
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    if (!this.fitted) throw new NotFittedError("HistogramBinningCalibration not fitted");
    const out = new Float64Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i] ?? 0;
      const bin = Math.min(Math.floor(s * this.nBins), this.nBins - 1);
      out[i] = this.binMeans_[bin] ?? 0.5;
    }
    return out;
  }
}

/**
 * Dirichlet calibration for multiclass probabilities.
 */
export class DirichletCalibration {
  private nClasses: number;
  private weights_: Float64Array = new Float64Array(0);
  private bias_: Float64Array = new Float64Array(0);
  private fitted = false;
  private maxIter: number;
  private lr: number;

  constructor(options: { nClasses?: number; maxIter?: number; lr?: number } = {}) {
    this.nClasses = options.nClasses ?? 2;
    this.maxIter = options.maxIter ?? 100;
    this.lr = options.lr ?? 0.01;
  }

  fit(probs: Float64Array[], labels: Int32Array): this {
    const n = probs.length;
    const k = this.nClasses;
    this.weights_ = new Float64Array(k).fill(1.0);
    this.bias_ = new Float64Array(k).fill(0.0);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const gradW = new Float64Array(k);
      const gradB = new Float64Array(k);

      for (let i = 0; i < n; i++) {
        const p = probs[i] ?? new Float64Array(k);
        const y = labels[i] ?? 0;

        // Log-scaled calibrated probs
        const logCal = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          logCal[c] = (this.weights_[c] ?? 1) * Math.log(Math.max(p[c] ?? 1e-10, 1e-10)) + (this.bias_[c] ?? 0);
        }

        // Softmax
        const maxL = Math.max(...logCal);
        const expL = new Float64Array(k);
        let sumE = 0;
        for (let c = 0; c < k; c++) {
          expL[c] = Math.exp((logCal[c] ?? 0) - maxL);
          sumE += expL[c] ?? 0;
        }
        const softmax = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          softmax[c] = (expL[c] ?? 0) / sumE;
        }

        // Cross-entropy gradient
        for (let c = 0; c < k; c++) {
          const err = (softmax[c] ?? 0) - (c === y ? 1 : 0);
          gradW[c] = (gradW[c] ?? 0) + err * Math.log(Math.max(p[c] ?? 1e-10, 1e-10));
          gradB[c] = (gradB[c] ?? 0) + err;
        }
      }

      for (let c = 0; c < k; c++) {
        this.weights_[c] = (this.weights_[c] ?? 1) - this.lr * (gradW[c] ?? 0) / n;
        this.bias_[c] = (this.bias_[c] ?? 0) - this.lr * (gradB[c] ?? 0) / n;
      }
    }

    this.fitted = true;
    return this;
  }

  calibrate(probs: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("DirichletCalibration not fitted");
    const k = this.nClasses;
    return probs.map(p => {
      const logCal = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        logCal[c] = (this.weights_[c] ?? 1) * Math.log(Math.max(p[c] ?? 1e-10, 1e-10)) + (this.bias_[c] ?? 0);
      }
      const maxL = Math.max(...logCal);
      const expL = new Float64Array(k);
      let sumE = 0;
      for (let c = 0; c < k; c++) {
        expL[c] = Math.exp((logCal[c] ?? 0) - maxL);
        sumE += expL[c] ?? 0;
      }
      const out = new Float64Array(k);
      for (let c = 0; c < k; c++) out[c] = (expL[c] ?? 0) / sumE;
      return out;
    });
  }
}

/**
 * Reliability diagram utilities.
 */
export function reliabilityDiagram(
  scores: Float64Array,
  labels: Int32Array,
  nBins = 10,
): { binCenters: Float64Array; fractions: Float64Array; counts: Int32Array } {
  const binCenters = new Float64Array(nBins);
  const fractions = new Float64Array(nBins);
  const counts = new Int32Array(nBins);

  for (let b = 0; b < nBins; b++) {
    binCenters[b] = (b + 0.5) / nBins;
  }

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i] ?? 0;
    const bin = Math.min(Math.floor(s * nBins), nBins - 1);
    fractions[bin] = (fractions[bin] ?? 0) + (labels[i] ?? 0);
    counts[bin]++;
  }

  for (let b = 0; b < nBins; b++) {
    if (counts[b] > 0) {
      fractions[b] = (fractions[b] ?? 0) / counts[b];
    }
  }

  return { binCenters, fractions, counts };
}

/**
 * Maximum Calibration Error (MaxCE) metric.
 */
export function maximumCalibrationError(
  scores: Float64Array,
  labels: Int32Array,
  nBins = 10,
): number {
  const { fractions, binCenters, counts } = reliabilityDiagram(scores, labels, nBins);
  let maxErr = 0;
  for (let b = 0; b < nBins; b++) {
    if (counts[b] > 0) {
      maxErr = Math.max(maxErr, Math.abs((fractions[b] ?? 0) - (binCenters[b] ?? 0)));
    }
  }
  return maxErr;
}

/**
 * Adaptive calibration error — uses adaptive binning.
 */
export function adaptiveCalibrationError(
  scores: Float64Array,
  labels: Int32Array,
  nBins = 10,
): number {
  const n = scores.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (scores[a] ?? 0) - (scores[b] ?? 0));
  const binSize = Math.floor(n / nBins);
  let ace = 0;

  for (let b = 0; b < nBins; b++) {
    const start = b * binSize;
    const end = b === nBins - 1 ? n : start + binSize;
    if (end <= start) continue;

    let sumS = 0; let sumY = 0;
    for (let i = start; i < end; i++) {
      const idx = indices[i] ?? 0;
      sumS += scores[idx] ?? 0;
      sumY += labels[idx] ?? 0;
    }
    const count = end - start;
    ace += Math.abs(sumS / count - sumY / count);
  }

  return ace / nBins;
}
