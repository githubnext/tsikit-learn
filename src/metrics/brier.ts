/**
 * Brier score, DET curve, and related probability calibration metrics.
 * Ported from sklearn.metrics
 */

/**
 * Compute the Brier score loss for binary/multiclass classification.
 * Lower is better (0 = perfect, 1 = worst).
 *
 * @param yTrue True binary labels (0 or 1)
 * @param yProb Predicted probabilities for the positive class
 * @param sampleWeight Optional per-sample weights
 * @returns Brier score (scalar)
 */
export function brierScoreLoss(
  yTrue: Int32Array | number[],
  yProb: Float64Array | number[],
  sampleWeight?: Float64Array | number[],
): number {
  const n = yTrue.length;
  if (n === 0) return 0;

  let weightSum = 0;
  let score = 0;

  for (let i = 0; i < n; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    const diff = (yTrue[i] ?? 0) - (yProb[i] ?? 0);
    score += w * diff * diff;
    weightSum += w;
  }

  return weightSum > 0 ? score / weightSum : 0;
}

export interface DetCurveResult {
  /** False negative rates */
  fnr: Float64Array;
  /** False positive rates */
  fpr: Float64Array;
  /** Threshold values at each point */
  thresholds: Float64Array;
}

/**
 * Compute the Detection Error Tradeoff (DET) curve.
 * The DET curve plots False Negative Rate (FNR) vs False Positive Rate (FPR).
 *
 * @param yTrue True binary labels (0 or 1)
 * @param yScore Scores/probabilities for the positive class
 * @returns DetCurveResult with fnr, fpr, and thresholds arrays
 */
export function detCurve(
  yTrue: Int32Array | number[],
  yScore: Float64Array | number[],
): DetCurveResult {
  const n = yTrue.length;

  // Sort by descending score
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0),
  );

  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < n; i++) {
    if ((yTrue[i] ?? 0) === 1) nPos++;
    else nNeg++;
  }

  const fnrArr: number[] = [];
  const fprArr: number[] = [];
  const thresholds: number[] = [];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    const idx = order[i]!;
    if ((yTrue[idx] ?? 0) === 1) {
      tp++;
    } else {
      fp++;
    }
    const threshold = yScore[idx] ?? 0;
    const fnr = nPos > 0 ? (nPos - tp) / nPos : 0;
    const fpr = nNeg > 0 ? fp / nNeg : 0;
    fnrArr.push(fnr);
    fprArr.push(fpr);
    thresholds.push(threshold);
  }

  return {
    fnr: new Float64Array(fnrArr),
    fpr: new Float64Array(fprArr),
    thresholds: new Float64Array(thresholds),
  };
}

export interface CalibrationCurveResult {
  /** Mean predicted probability in each bin */
  probPred: Float64Array;
  /** Fraction of positives in each bin */
  probTrue: Float64Array;
  /** Bin indices for each sample */
  binIds: Int32Array;
}

/**
 * Compute the calibration curve (reliability diagram).
 *
 * @param yTrue True binary labels (0 or 1)
 * @param yProb Predicted probabilities for the positive class
 * @param nBins Number of bins to use
 * @returns CalibrationCurveResult
 */
export function calibrationCurve(
  yTrue: Int32Array | number[],
  yProb: Float64Array | number[],
  nBins = 5,
): CalibrationCurveResult {
  const n = yTrue.length;
  const binCounts = new Int32Array(nBins);
  const binPosSum = new Float64Array(nBins);
  const binProbSum = new Float64Array(nBins);
  const binIds = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const p = Math.max(0, Math.min(1, yProb[i] ?? 0));
    const binIdx = Math.min(nBins - 1, Math.floor(p * nBins));
    binIds[i] = binIdx;
    binCounts[binIdx]!++;
    binPosSum[binIdx]! += yTrue[i] ?? 0;
    binProbSum[binIdx]! += p;
  }

  const probPred: number[] = [];
  const probTrue: number[] = [];
  for (let b = 0; b < nBins; b++) {
    const count = binCounts[b] ?? 0;
    if (count > 0) {
      probPred.push((binProbSum[b] ?? 0) / count);
      probTrue.push((binPosSum[b] ?? 0) / count);
    }
  }

  return {
    probPred: new Float64Array(probPred),
    probTrue: new Float64Array(probTrue),
    binIds,
  };
}

/**
 * Log loss (cross-entropy loss) for probabilistic predictions.
 * @param yTrue True labels (integers)
 * @param yProb Predicted probabilities (nSamples x nClasses)
 * @param eps Small value for clipping to avoid log(0)
 */
export function logLoss(
  yTrue: Int32Array | number[],
  yProb: Float64Array[],
  eps = 1e-15,
): number {
  const n = yTrue.length;
  if (n === 0) return 0;

  const nClasses = yProb[0]?.length ?? 2;
  let loss = 0;

  for (let i = 0; i < n; i++) {
    const trueClass = yTrue[i] ?? 0;
    const probs = yProb[i]!;
    const p = Math.max(eps, Math.min(1 - eps, probs[trueClass] ?? eps));
    loss -= Math.log(p);
  }

  if (nClasses === 2) {
    // Binary case: add contribution from negative class
    for (let i = 0; i < n; i++) {
      const trueClass = yTrue[i] ?? 0;
      if (trueClass === 0) {
        const probs = yProb[i]!;
        const p1 = Math.max(eps, Math.min(1 - eps, probs[1] ?? eps));
        loss -= Math.log(1 - p1);
      }
    }
  }

  return loss / n;
}

/**
 * Compute the Expected Calibration Error (ECE).
 * Measures how well predicted probabilities match observed frequencies.
 */
export function expectedCalibrationError(
  yTrue: Int32Array | number[],
  yProb: Float64Array | number[],
  nBins = 10,
): number {
  const n = yTrue.length;
  if (n === 0) return 0;

  const result = calibrationCurve(yTrue, yProb, nBins);
  const binCounts = new Int32Array(nBins);
  for (let i = 0; i < n; i++) {
    binCounts[result.binIds[i] ?? 0]!++;
  }

  let ece = 0;
  for (let b = 0; b < result.probPred.length; b++) {
    const count = binCounts[b] ?? 0;
    ece +=
      (count / n) *
      Math.abs((result.probTrue[b] ?? 0) - (result.probPred[b] ?? 0));
  }
  return ece;
}
