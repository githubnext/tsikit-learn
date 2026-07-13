/**
 * Multiclass classification metrics extensions.
 */

export function topKAccuracy(y: Int32Array, yProba: Float64Array[], k = 5): number {
  const n = y.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const proba = yProba[i] as Float64Array;
    const sortedIdx = Array.from({ length: proba.length }, (_, j) => j)
      .sort((a, b) => (proba[b] ?? 0) - (proba[a] ?? 0));
    if (sortedIdx.slice(0, k).includes(y[i] ?? -1)) correct++;
  }
  return n > 0 ? correct / n : 0;
}

export function averagePrecisionScore(y: Int32Array, yScore: Float64Array): number {
  const n = y.length;
  const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let ap = 0, nPos = 0;
  for (let i = 0; i < n; i++) {
    if (y[sortedIdx[i] as number] === 1) {
      nPos++;
      ap += nPos / (i + 1);
    }
  }
  const totalPos = Array.from(y).filter((v) => v === 1).length;
  return totalPos > 0 ? ap / totalPos : 0;
}

export function rocAucScore(y: Int32Array, yScore: Float64Array): number {
  const n = y.length;
  const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tpCount = 0, fpCount = 0;
  let auc = 0;
  let prevTp = 0, prevFp = 0;
  const nPos = Array.from(y).filter((v) => v === 1).length;
  const nNeg = n - nPos;
  for (let i = 0; i < n; i++) {
    const idx = sortedIdx[i] as number;
    if (y[idx] === 1) tpCount++; else fpCount++;
    const tpr = nPos > 0 ? tpCount / nPos : 0;
    const fpr = nNeg > 0 ? fpCount / nNeg : 0;
    const prevTpr = nPos > 0 ? prevTp / nPos : 0;
    const prevFpr = nNeg > 0 ? prevFp / nNeg : 0;
    auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
    prevTp = tpCount;
    prevFp = fpCount;
  }
  return auc;
}

export function rocCurve(y: Int32Array, yScore: Float64Array): { fpr: Float64Array; tpr: Float64Array; thresholds: Float64Array } {
  const n = y.length;
  const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const nPos = Array.from(y).filter((v) => v === 1).length;
  const nNeg = n - nPos;
  const fprs: number[] = [0];
  const tprs: number[] = [0];
  const thresholds: number[] = [];
  let tp = 0, fp = 0;
  for (let i = 0; i < n; i++) {
    const idx = sortedIdx[i] as number;
    if (y[idx] === 1) tp++; else fp++;
    fprs.push(nNeg > 0 ? fp / nNeg : 0);
    tprs.push(nPos > 0 ? tp / nPos : 0);
    thresholds.push(yScore[idx] ?? 0);
  }
  return {
    fpr: Float64Array.from(fprs),
    tpr: Float64Array.from(tprs),
    thresholds: Float64Array.from(thresholds),
  };
}

export function precisionRecallCurve(y: Int32Array, yScore: Float64Array): { precision: Float64Array; recall: Float64Array; thresholds: Float64Array } {
  const n = y.length;
  const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const nPos = Array.from(y).filter((v) => v === 1).length;
  const precisions: number[] = [];
  const recalls: number[] = [];
  const thresholds: number[] = [];
  let tp = 0, fp = 0;
  for (let i = 0; i < n; i++) {
    const idx = sortedIdx[i] as number;
    if (y[idx] === 1) tp++; else fp++;
    precisions.push(tp + fp > 0 ? tp / (tp + fp) : 1);
    recalls.push(nPos > 0 ? tp / nPos : 0);
    thresholds.push(yScore[idx] ?? 0);
  }
  return {
    precision: Float64Array.from([1, ...precisions]),
    recall: Float64Array.from([0, ...recalls]),
    thresholds: Float64Array.from(thresholds),
  };
}

export function cohensKappa(y1: Int32Array, y2: Int32Array): number {
  const n = y1.length;
  const classes = Array.from(new Set([...Array.from(y1), ...Array.from(y2)]));
  const k = classes.length;
  const confMatrix = Array.from({ length: k }, () => new Int32Array(k));
  for (let i = 0; i < n; i++) {
    const r = classes.indexOf(y1[i] ?? 0);
    const c = classes.indexOf(y2[i] ?? 0);
    if (r >= 0 && c >= 0) { const arr = confMatrix[r] as Int32Array; arr[c] = (arr[c] ?? 0) + 1; }
  }
  const rowSums = confMatrix.map((row) => Array.from(row).reduce((a, b) => a + b, 0));
  const colSums = classes.map((_, j) => confMatrix.reduce((s, row) => s + ((row as Int32Array)[j] ?? 0), 0));
  const observed = classes.reduce((s, _, i) => s + ((confMatrix[i] as Int32Array)[i] ?? 0), 0) / n;
  const expected = classes.reduce((s, _, i) => s + ((rowSums[i] ?? 0) * (colSums[i] ?? 0)), 0) / (n * n);
  return expected === 1 ? 0 : (observed - expected) / (1 - expected);
}

export function balancedAccuracy(y: Int32Array, yPred: Int32Array): number {
  const classes = Array.from(new Set(Array.from(y)));
  let sum = 0;
  for (const cls of classes) {
    const mask = Array.from(y).map((v) => v === cls);
    const total = mask.filter(Boolean).length;
    const correct = mask.filter((m, i) => m && yPred[i] === cls).length;
    if (total > 0) sum += correct / total;
  }
  return classes.length > 0 ? sum / classes.length : 0;
}

export function hammingLoss(y: Int32Array, yPred: Int32Array): number {
  const n = y.length;
  let incorrect = 0;
  for (let i = 0; i < n; i++) if (y[i] !== yPred[i]) incorrect++;
  return n > 0 ? incorrect / n : 0;
}

export function zeroOneLoss(y: Int32Array, yPred: Int32Array, normalize = true): number {
  const n = y.length;
  let incorrect = 0;
  for (let i = 0; i < n; i++) if (y[i] !== yPred[i]) incorrect++;
  return normalize ? incorrect / n : incorrect;
}

export function classLikelihoodRatios(y: Int32Array, yPred: Int32Array): { positiveClass: number; negativeClass: number } {
  const n = y.length;
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < n; i++) {
    const actual = y[i] === 1;
    const predicted = yPred[i] === 1;
    if (actual && predicted) tp++;
    else if (!actual && predicted) fp++;
    else if (!actual && !predicted) tn++;
    else fn++;
  }
  const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
  const tnr = fp + tn > 0 ? tn / (fp + tn) : 0;
  const fnr = tp + fn > 0 ? fn / (tp + fn) : 0;
  return {
    positiveClass: fpr > 0 ? tpr / fpr : Number.POSITIVE_INFINITY,
    negativeClass: tnr > 0 ? fnr / tnr : Number.POSITIVE_INFINITY,
  };
}
