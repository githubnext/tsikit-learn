/**
 * Extended metrics: zero_one_loss, balanced_accuracy, hamming_loss,
 * jaccard_score, cohen_kappa_score, and additional regression metrics.
 */

/** Zero-one loss: fraction of misclassified samples (or count if normalize=false). */
export function zeroOneLoss(
  yTrue: Int32Array,
  yPred: Int32Array,
  normalize = true,
): number {
  let count = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) count++;
  }
  return normalize ? count / yTrue.length : count;
}

/** Balanced accuracy: average recall per class (handles imbalanced datasets). */
export function balancedAccuracyScore(yTrue: Int32Array, yPred: Int32Array): number {
  const classes = [...new Set(Array.from(yTrue))];
  let sumRecall = 0;
  let classCount = 0;
  for (const c of classes) {
    let tp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) {
        if ((yPred[i] ?? 0) === c) tp++;
        else fn++;
      }
    }
    if (tp + fn > 0) {
      sumRecall += tp / (tp + fn);
      classCount++;
    }
  }
  return classCount > 0 ? sumRecall / classCount : 0;
}

/** Hamming loss: fraction of labels differing between yTrue and yPred. */
export function hammingLoss(yTrue: Int32Array, yPred: Int32Array): number {
  let diff = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) diff++;
  }
  return diff / yTrue.length;
}

/** Jaccard similarity score. */
export function jaccardScore(yTrue: Int32Array, yPred: Int32Array, average: "binary" | "macro" = "binary"): number {
  if (average === "binary") {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = yTrue[i] ?? 0;
      const p = yPred[i] ?? 0;
      if (t === 1 && p === 1) tp++;
      else if (t === 0 && p === 1) fp++;
      else if (t === 1 && p === 0) fn++;
    }
    return tp / (tp + fp + fn + 1e-10);
  }
  const classes = [...new Set([...Array.from(yTrue), ...Array.from(yPred)])];
  let sum = 0;
  for (const c of classes) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = (yTrue[i] ?? 0) === c ? 1 : 0;
      const p = (yPred[i] ?? 0) === c ? 1 : 0;
      if (t === 1 && p === 1) tp++;
      else if (t === 0 && p === 1) fp++;
      else if (t === 1 && p === 0) fn++;
    }
    sum += tp / (tp + fp + fn + 1e-10);
  }
  return sum / classes.length;
}

/** Cohen's kappa score: agreement between two annotators. */
export function cohenKappaScore(yTrue: Int32Array, yPred: Int32Array): number {
  const n = yTrue.length;
  const classes = [...new Set([...Array.from(yTrue), ...Array.from(yPred)])].sort((a, b) => a - b);
  const nC = classes.length;
  const classIdx = new Map(classes.map((c, i) => [c, i]));

  // Confusion matrix
  const cm = Array.from({ length: nC }, () => new Float64Array(nC));
  for (let i = 0; i < n; i++) {
    const ti = classIdx.get(yTrue[i] ?? 0) ?? 0;
    const pi = classIdx.get(yPred[i] ?? 0) ?? 0;
    const row = cm[ti];
    if (row !== undefined) row[pi] = (row[pi] ?? 0) + 1;
  }

  let po = 0;
  for (let i = 0; i < nC; i++) po += (cm[i]?.[i] ?? 0);
  po /= n;

  let pe = 0;
  for (let i = 0; i < nC; i++) {
    let rowSum = 0, colSum = 0;
    for (let j = 0; j < nC; j++) {
      rowSum += cm[i]?.[j] ?? 0;
      colSum += cm[j]?.[i] ?? 0;
    }
    pe += (rowSum / n) * (colSum / n);
  }

  return (po - pe) / (1 - pe + 1e-10);
}

/** Mean Poisson deviance. */
export function meanPoissonDeviance(yTrue: Float64Array, yPred: Float64Array): number {
  let dev = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yi = Math.max(yTrue[i] ?? 0, 0);
    const yHat = Math.max(yPred[i] ?? 1e-10, 1e-10);
    dev += 2 * (yi * Math.log((yi + 1e-10) / yHat) - (yi - yHat));
  }
  return dev / yTrue.length;
}

/** Mean gamma deviance. */
export function meanGammaDeviance(yTrue: Float64Array, yPred: Float64Array): number {
  let dev = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yi = Math.max(yTrue[i] ?? 1e-10, 1e-10);
    const yHat = Math.max(yPred[i] ?? 1e-10, 1e-10);
    dev += 2 * (Math.log(yHat / yi) + yi / yHat - 1);
  }
  return dev / yTrue.length;
}

/** Mean Tweedie deviance. */
export function meanTweedieDeviance(yTrue: Float64Array, yPred: Float64Array, power = 0): number {
  let dev = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yi = yTrue[i] ?? 0;
    const yHat = Math.max(yPred[i] ?? 1e-10, 1e-10);
    if (power === 0) dev += (yi - yHat) ** 2;
    else if (power === 1) dev += 2 * (yi * Math.log((yi + 1e-10) / yHat) - (yi - yHat));
    else if (power === 2) dev += 2 * (Math.log(yHat / (yi + 1e-10)) + yi / yHat - 1);
    else dev += 2 * (
      (yi ** (2 - power)) / ((1 - power) * (2 - power)) -
      yi * yHat ** (1 - power) / (1 - power) +
      yHat ** (2 - power) / (2 - power)
    );
  }
  return dev / yTrue.length;
}

/** Max error: maximum residual error between y and yPred. */
export function maxError(yTrue: Float64Array, yPred: Float64Array): number {
  let maxErr = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const err = Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

/** Mean absolute percentage error. */
export function meanAbsolutePercentageError(yTrue: Float64Array, yPred: Float64Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yi = yTrue[i] ?? 0;
    if (Math.abs(yi) < 1e-10) continue;
    sum += Math.abs((yi - (yPred[i] ?? 0)) / yi);
    count++;
  }
  return count > 0 ? sum / count : 0;
}
