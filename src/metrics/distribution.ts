/**
 * Distribution and probabilistic metrics.
 * Port of sklearn.metrics._distribution
 */

/**
 * Brier score loss for probabilistic predictions.
 * Port of sklearn.metrics.brier_score_loss
 */
export function brierScoreLoss(
  yTrue: Int32Array,
  yProba: Float64Array,
  sampleWeight: Float64Array | null = null,
  posLabel: number | null = null,
): number {
  const n = yTrue.length;
  const posLbl = posLabel ?? Math.max(...yTrue);
  let score = 0;
  let totalWeight = 0;
  for (let i = 0; i < n; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    const p = yProba[i] ?? 0;
    const t = yTrue[i] === posLbl ? 1 : 0;
    score += w * (p - t) ** 2;
    totalWeight += w;
  }
  return score / (totalWeight || 1);
}

/**
 * Log loss (cross-entropy loss).
 * Port of sklearn.metrics.log_loss (probabilistic version).
 */
export function multiclassLogLoss(
  yTrue: Int32Array,
  yProba: Float64Array[],
  normalize = true,
  sampleWeight: Float64Array | null = null,
  labels: Int32Array | null = null,
  eps = 1e-15,
): number {
  const n = yTrue.length;
  const classSet = labels
    ? Array.from(labels)
    : Array.from(new Set(Array.from(yTrue))).sort((a, b) => a - b);
  const classMap = new Map(classSet.map((c, i) => [c, i]));
  let loss = 0;
  let totalWeight = 0;
  for (let i = 0; i < n; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    const ci = classMap.get(yTrue[i]!)!;
    const p = Math.max(eps, Math.min(1 - eps, yProba[i]?.[ci] ?? eps));
    loss += w * Math.log(p);
    totalWeight += w;
  }
  const result = -loss;
  return normalize ? result / (totalWeight || 1) : result;
}

/**
 * Calibration curve data for reliability diagrams.
 * Port of sklearn.calibration.calibration_curve
 */
export interface CalibrationCurveResult {
  fracOfPositives: Float64Array;
  meanPredictedValue: Float64Array;
}

export function calibrationCurve(
  yTrue: Int32Array,
  yProba: Float64Array,
  normalize = false,
  nBins = 5,
  strategy: "uniform" | "quantile" = "uniform",
): CalibrationCurveResult {
  const n = yTrue.length;
  let probas = yProba;
  if (normalize) {
    const minP = Math.min(...probas);
    const maxP = Math.max(...probas);
    const range = maxP - minP;
    probas = new Float64Array(probas.map((p) => (p - minP) / (range || 1)));
  }

  let binEdges: Float64Array;
  if (strategy === "quantile") {
    const sorted = Float64Array.from(probas).sort();
    binEdges = new Float64Array(nBins + 1);
    for (let b = 0; b <= nBins; b++) {
      const idx = Math.min(Math.floor((b / nBins) * n), n - 1);
      binEdges[b] = sorted[idx]!;
    }
  } else {
    binEdges = new Float64Array(nBins + 1);
    for (let b = 0; b <= nBins; b++) binEdges[b] = b / nBins;
  }

  const fracOfPositives = new Float64Array(nBins);
  const meanPredictedValue = new Float64Array(nBins);
  const counts = new Float64Array(nBins);
  const positives = new Float64Array(nBins);

  for (let i = 0; i < n; i++) {
    const p = probas[i]!;
    // Find bin
    let binIdx = nBins - 1;
    for (let b = 0; b < nBins; b++) {
      if (p < (binEdges[b + 1] ?? 1)) {
        binIdx = b;
        break;
      }
    }
    counts[binIdx]!++;
    meanPredictedValue[binIdx]! += p;
    if (yTrue[i] === 1) positives[binIdx]!++;
  }

  for (let b = 0; b < nBins; b++) {
    if ((counts[b] ?? 0) > 0) {
      fracOfPositives[b] = (positives[b] ?? 0) / counts[b]!;
      meanPredictedValue[b] = (meanPredictedValue[b] ?? 0) / counts[b]!;
    }
  }

  return { fracOfPositives, meanPredictedValue };
}

/**
 * Expected Calibration Error (ECE).
 */
export function expectedCalibrationError(
  yTrue: Int32Array,
  yProba: Float64Array,
  nBins = 10,
): number {
  const { fracOfPositives, meanPredictedValue } = calibrationCurve(
    yTrue,
    yProba,
    false,
    nBins,
  );
  const n = yTrue.length;
  let ece = 0;
  for (let b = 0; b < nBins; b++) {
    const binSize = yTrue.filter((_, i) => {
      const p = yProba[i]!;
      return p >= b / nBins && p < (b + 1) / nBins;
    }).length;
    ece +=
      (binSize / n) *
      Math.abs((fracOfPositives[b] ?? 0) - (meanPredictedValue[b] ?? 0));
  }
  return ece;
}

/**
 * Reliability diagram data (same as calibration curve, alias).
 */
export const reliabilityDiagram = calibrationCurve;

/**
 * KL divergence between two probability distributions.
 */
export function klDivergence(
  p: Float64Array,
  q: Float64Array,
  eps = 1e-15,
): number {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = Math.max(eps, p[i]!);
    const qi = Math.max(eps, q[i]!);
    kl += pi * Math.log(pi / qi);
  }
  return kl;
}

/**
 * Jensen-Shannon divergence.
 */
export function jsDivergence(
  p: Float64Array,
  q: Float64Array,
  eps = 1e-15,
): number {
  const m = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) m[i] = ((p[i] ?? 0) + (q[i] ?? 0)) / 2;
  return (klDivergence(p, m, eps) + klDivergence(q, m, eps)) / 2;
}
