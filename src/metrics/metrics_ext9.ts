/**
 * Metrics extensions: additional scoring functions.
 * Mirrors sklearn.metrics: balanced_accuracy, top_k_accuracy, hamming_loss, etc.
 */

/** Balanced accuracy: mean of sensitivity and specificity per class. */
export function balancedAccuracyScore(
  yTrue: Int32Array,
  yPred: Int32Array,
  adjusted = false,
): number {
  const classes = [...new Set([...yTrue, ...yPred])].sort((a, b) => a - b);
  let sum = 0;
  for (const c of classes) {
    let tp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? -1) === c) {
        if ((yPred[i] ?? -1) === c) tp++; else fn++;
      }
    }
    const denom = tp + fn;
    sum += denom > 0 ? tp / denom : 0;
  }
  const score = sum / classes.length;
  if (adjusted) return (score - 1 / classes.length) / (1 - 1 / classes.length);
  return score;
}

/** Top-k accuracy score. */
export function topKAccuracyScore(
  yTrue: Int32Array,
  yScore: Float64Array[],
  k = 2,
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const scores = yScore[i];
    if (!scores) continue;
    const topK = Array.from(scores)
      .map((v, j) => ({ v, j }))
      .sort((a, b) => b.v - a.v)
      .slice(0, k)
      .map((x) => x.j);
    if (topK.includes(yTrue[i] ?? -1)) correct++;
  }
  return correct / yTrue.length;
}

/** Hamming loss: fraction of labels incorrectly predicted. */
export function hammingLoss(
  yTrue: Int32Array,
  yPred: Int32Array,
): number {
  let wrong = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? -1) !== (yPred[i] ?? -1)) wrong++;
  }
  return wrong / yTrue.length;
}

/** Zero-one loss: fraction of samples misclassified. */
export function zeroOneLoss(
  yTrue: Int32Array,
  yPred: Int32Array,
  normalize = true,
): number {
  let wrong = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? -1) !== (yPred[i] ?? -1)) wrong++;
  }
  return normalize ? wrong / yTrue.length : wrong;
}

/** Jaccard score: intersection over union. */
export function jaccardScore(
  yTrue: Int32Array,
  yPred: Int32Array,
  average: "macro" | "micro" | "binary" = "binary",
): number {
  const classes = [...new Set([...yTrue, ...yPred])].sort((a, b) => a - b);
  if (average === "micro") {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = yTrue[i] ?? -1, p = yPred[i] ?? -1;
      if (t === p) tp++; else { fp++; fn++; }
    }
    return (tp + fp + fn) === 0 ? 0 : tp / (tp + fp + fn);
  }
  let s = 0;
  for (const c of classes) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = yTrue[i] ?? -1, p = yPred[i] ?? -1;
      if (t === c && p === c) tp++;
      else if (t !== c && p === c) fp++;
      else if (t === c && p !== c) fn++;
    }
    s += (tp + fp + fn) === 0 ? 0 : tp / (tp + fp + fn);
  }
  return s / classes.length;
}

/** Cohen's kappa: agreement measure corrected for chance. */
export function cohensKappa(
  yTrue: Int32Array,
  yPred: Int32Array,
): number {
  const n = yTrue.length;
  const classes = [...new Set([...yTrue, ...yPred])].sort((a, b) => a - b);
  const k = classes.length;
  const conf = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const classIdx = new Map(classes.map((c, i) => [c, i]));
  for (let i = 0; i < n; i++) {
    const ti = classIdx.get(yTrue[i] ?? -1) ?? 0;
    const pi = classIdx.get(yPred[i] ?? -1) ?? 0;
    conf[ti]![pi] = (conf[ti]![pi] ?? 0) + 1;
  }
  let po = 0;
  for (let i = 0; i < k; i++) po += (conf[i]?.[i] ?? 0) / n;
  let pe = 0;
  for (let i = 0; i < k; i++) {
    let rsum = 0, csum = 0;
    for (let j = 0; j < k; j++) { rsum += conf[i]?.[j] ?? 0; csum += conf[j]?.[i] ?? 0; }
    pe += (rsum / n) * (csum / n);
  }
  return pe === 1 ? 0 : (po - pe) / (1 - pe);
}

/** Matthews correlation coefficient. */
export function matthewsCorrCoefMulti(
  yTrue: Int32Array,
  yPred: Int32Array,
): number {
  return cohensKappa(yTrue, yPred);
}

/** Brier score loss for probabilistic predictions. */
export function brierScoreLossExt(
  yTrue: Int32Array,
  yProba: Float64Array,
  posLabel = 1,
): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = (yTrue[i] ?? 0) === posLabel ? 1 : 0;
    const diff = t - (yProba[i] ?? 0);
    s += diff * diff;
  }
  return s / yTrue.length;
}

/** D^2 Tweedie score. */
export function d2TweedieScore(
  yTrue: Float64Array,
  yPred: Float64Array,
  power = 0,
): number {
  const n = yTrue.length;
  let dev = 0, devNull = 0;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[i] ?? 0;
  mean /= n;
  const tweedie = (a: number, b: number): number => {
    if (power === 0) return (a - b) ** 2;
    if (power === 1) return a * Math.log(Math.max(a, 1e-10) / Math.max(b, 1e-10)) - a + b;
    if (power === 2) return Math.log(Math.max(a, 1e-10) / Math.max(b, 1e-10)) - (a - b) / Math.max(b, 1e-10);
    return (Math.max(a, 0) ** (2 - power)) / ((1 - power) * (2 - power)) - a * (Math.max(b, 0) ** (1 - power)) / (1 - power) + (Math.max(b, 0) ** (2 - power)) / (2 - power);
  };
  for (let i = 0; i < n; i++) {
    dev += tweedie(yTrue[i] ?? 0, yPred[i] ?? 0);
    devNull += tweedie(yTrue[i] ?? 0, mean);
  }
  return devNull === 0 ? 0 : 1 - dev / devNull;
}
