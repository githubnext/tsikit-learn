/**
 * Multilabel classification metrics.
 */

/** Jaccard similarity score averaged over samples. */
export function jaccardScore(
  yTrue: Float64Array[],
  yPred: Float64Array[],
): number {
  const n = Math.min(yTrue.length, yPred.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? new Float64Array(0);
    const yp = yPred[i] ?? new Float64Array(0);
    const len = Math.min(yt.length, yp.length);
    let inter = 0;
    let union = 0;
    for (let j = 0; j < len; j++) {
      const a = (yt[j] ?? 0) > 0.5 ? 1 : 0;
      const b = (yp[j] ?? 0) > 0.5 ? 1 : 0;
      inter += a & b;
      union += a | b;
    }
    total += union === 0 ? 1 : inter / union;
  }
  return total / n;
}

/** Hamming loss: fraction of labels that are incorrectly predicted. */
export function hammingLoss(
  yTrue: Float64Array[],
  yPred: Float64Array[],
): number {
  const n = Math.min(yTrue.length, yPred.length);
  if (n === 0) return 0;
  const nLabels = (yTrue[0] ?? new Float64Array(0)).length;
  if (nLabels === 0) return 0;
  let wrong = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? new Float64Array(0);
    const yp = yPred[i] ?? new Float64Array(0);
    for (let j = 0; j < nLabels; j++) {
      const a = (yt[j] ?? 0) > 0.5 ? 1 : 0;
      const b = (yp[j] ?? 0) > 0.5 ? 1 : 0;
      if (a !== b) wrong++;
    }
  }
  return wrong / (n * nLabels);
}

/**
 * Coverage error: average number of labels that have to be included in the
 * final prediction to cover all true labels.
 */
export function coverageError(
  yTrue: Float64Array[],
  yScore: Float64Array[],
): number {
  const n = Math.min(yTrue.length, yScore.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? new Float64Array(0);
    const ys = yScore[i] ?? new Float64Array(0);
    const nLabels = yt.length;
    // sort indices by score descending
    const order = Array.from({ length: nLabels }, (_, k) => k);
    order.sort((a, b) => (ys[b] ?? 0) - (ys[a] ?? 0));
    let maxRank = 0;
    for (let j = 0; j < nLabels; j++) {
      if ((yt[order[j] ?? 0] ?? 0) > 0.5) maxRank = j + 1;
    }
    total += maxRank;
  }
  return total / n;
}

/** Label ranking average precision. */
export function labelRankingAveragePrecision(
  yTrue: Float64Array[],
  yScore: Float64Array[],
): number {
  const n = Math.min(yTrue.length, yScore.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? new Float64Array(0);
    const ys = yScore[i] ?? new Float64Array(0);
    const nLabels = yt.length;
    const order = Array.from({ length: nLabels }, (_, k) => k);
    order.sort((a, b) => (ys[b] ?? 0) - (ys[a] ?? 0));
    let nRelevant = 0;
    let sum = 0;
    for (let j = 0; j < nLabels; j++) {
      if ((yt[order[j] ?? 0] ?? 0) > 0.5) {
        nRelevant++;
        sum += nRelevant / (j + 1);
      }
    }
    const totalRelevant = Array.from(yt).filter((v) => v > 0.5).length;
    if (totalRelevant > 0) total += sum / totalRelevant;
  }
  return total / n;
}

/** Label ranking loss: fraction of label pairs that are incorrectly ordered. */
export function labelRankingLoss(
  yTrue: Float64Array[],
  yScore: Float64Array[],
): number {
  const n = Math.min(yTrue.length, yScore.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? new Float64Array(0);
    const ys = yScore[i] ?? new Float64Array(0);
    const nLabels = yt.length;
    let relevant = 0;
    let irrelevant = 0;
    let wrong = 0;
    for (let j = 0; j < nLabels; j++) {
      if ((yt[j] ?? 0) > 0.5) relevant++;
      else irrelevant++;
    }
    if (relevant === 0 || irrelevant === 0) continue;
    for (let j = 0; j < nLabels; j++) {
      if ((yt[j] ?? 0) <= 0.5) continue;
      for (let k = 0; k < nLabels; k++) {
        if ((yt[k] ?? 0) > 0.5) continue;
        if ((ys[j] ?? 0) <= (ys[k] ?? 0)) wrong++;
      }
    }
    total += wrong / (relevant * irrelevant);
  }
  return total / n;
}
