/**
 * Extended model evaluation: learning_curve extensions, bias-variance decomposition,
 * threshold optimization, and calibration metrics.
 */

/** Bias-variance decomposition for regression (using bootstrap). */
export interface BiasVarianceResult {
  avgLoss: number;
  avgBias2: number;
  avgVar: number;
  avgNoise: number;
}

export function biasVarianceDecomposition(
  predictFns: Array<(X: Float64Array[]) => Float64Array>,
  XTest: Float64Array[],
  yTest: Float64Array,
): BiasVarianceResult {
  const n = XTest.length;
  const k = predictFns.length;

  // Matrix of predictions: preds[i][j] = pred of estimator i on sample j
  const preds = predictFns.map((fn) => fn(XTest));

  const meanPred = new Float64Array(n).map((_, j) => {
    let sum = 0;
    for (let i = 0; i < k; i++) sum += preds[i]?.[j] ?? 0;
    return sum / k;
  });

  let avgBias2 = 0;
  let avgVar = 0;
  let avgLoss = 0;
  for (let j = 0; j < n; j++) {
    const yj = yTest[j] ?? 0;
    const mj = meanPred[j] ?? 0;
    avgBias2 += (mj - yj) ** 2;
    let varJ = 0;
    for (let i = 0; i < k; i++) varJ += ((preds[i]?.[j] ?? 0) - mj) ** 2;
    varJ /= k;
    avgVar += varJ;
    avgLoss += (mj - yj) ** 2 + varJ;
  }
  avgBias2 /= n;
  avgVar /= n;
  avgLoss /= n;

  return { avgLoss, avgBias2, avgVar, avgNoise: avgLoss - avgBias2 - avgVar };
}

/** Learning curve: scores for training sizes. */
export interface LearningCurveResult {
  trainSizes: Int32Array;
  trainScores: Float64Array[];
  testScores: Float64Array[];
}

export function learningCurveData(
  trainSizesFrac: number[],
  trainScoresAll: Float64Array[],
  testScoresAll: Float64Array[],
  nTotal: number,
): LearningCurveResult {
  const trainSizes = Int32Array.from(
    trainSizesFrac.map((f) => Math.round(f * nTotal)),
  );
  return { trainSizes, trainScores: trainScoresAll, testScores: testScoresAll };
}

/** Threshold optimization: find best classification threshold for a metric. */
export function optimizeThreshold(
  yTrue: Int32Array,
  yScores: Float64Array,
  metric: "f1" | "accuracy" | "balanced_accuracy" = "f1",
): number {
  const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
  let bestThreshold = 0.5;
  let bestScore = 0;

  for (const t of thresholds) {
    const yPred = Int32Array.from(yScores.map((s) => (s >= t ? 1 : 0)));
    let score = 0;
    if (metric === "accuracy") {
      let correct = 0;
      for (let i = 0; i < yTrue.length; i++)
        if ((yTrue[i] ?? 0) === (yPred[i] ?? 0)) correct++;
      score = correct / yTrue.length;
    } else if (metric === "f1") {
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if ((yTrue[i] ?? 0) === 1 && (yPred[i] ?? 0) === 1) tp++;
        else if ((yTrue[i] ?? 0) === 0 && (yPred[i] ?? 0) === 1) fp++;
        else if ((yTrue[i] ?? 0) === 1 && (yPred[i] ?? 0) === 0) fn++;
      }
      const prec = tp / (tp + fp + 1e-10);
      const rec = tp / (tp + fn + 1e-10);
      score = (2 * prec * rec) / (prec + rec + 1e-10);
    } else {
      // balanced_accuracy
      const classes = [0, 1];
      let sumRecall = 0;
      for (const c of classes) {
        let tp = 0;
        let total = 0;
        for (let i = 0; i < yTrue.length; i++) {
          if ((yTrue[i] ?? 0) === c) {
            total++;
            if ((yPred[i] ?? 0) === c) tp++;
          }
        }
        if (total > 0) sumRecall += tp / total;
      }
      score = sumRecall / classes.length;
    }
    if (score > bestScore) {
      bestScore = score;
      bestThreshold = t;
    }
  }
  return bestThreshold;
}

/** Expected Calibration Error (ECE). */
export function expectedCalibrationError(
  yTrue: Int32Array,
  yProba: Float64Array,
  nBins = 10,
): number {
  const n = yTrue.length;
  const binEdges = Array.from({ length: nBins + 1 }, (_, i) => i / nBins);
  let ece = 0;

  for (let b = 0; b < nBins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;
    const inBin: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = yProba[i] ?? 0;
      if (p >= lo && p < hi) inBin.push(i);
    }
    if (inBin.length === 0) continue;
    const avgConf =
      inBin.reduce((s, i) => s + (yProba[i] ?? 0), 0) / inBin.length;
    const avgAcc =
      inBin.filter((i) => (yTrue[i] ?? 0) === 1).length / inBin.length;
    ece += (inBin.length / n) * Math.abs(avgConf - avgAcc);
  }
  return ece;
}

/** Maximum Calibration Error (MCE). */
export function maximumCalibrationError(
  yTrue: Int32Array,
  yProba: Float64Array,
  nBins = 10,
): number {
  const n = yTrue.length;
  const binEdges = Array.from({ length: nBins + 1 }, (_, i) => i / nBins);
  let mce = 0;

  for (let b = 0; b < nBins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;
    const inBin: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = yProba[i] ?? 0;
      if (p >= lo && p < hi) inBin.push(i);
    }
    if (inBin.length === 0) continue;
    const avgConf =
      inBin.reduce((s, i) => s + (yProba[i] ?? 0), 0) / inBin.length;
    const avgAcc =
      inBin.filter((i) => (yTrue[i] ?? 0) === 1).length / inBin.length;
    const err = Math.abs(avgConf - avgAcc);
    if (err > mce) mce = err;
  }
  return mce;
}
