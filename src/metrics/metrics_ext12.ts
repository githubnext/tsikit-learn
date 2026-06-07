/**
 * Multi-label classification metrics extensions.
 */

export function jaccard_score_multilabel(y: number[][], yPred: number[][]): number {
  const n = y.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const yi = new Set(y[i]);
    const pi = new Set(yPred[i]);
    let intersect = 0;
    for (const v of yi) if (pi.has(v)) intersect++;
    const union = yi.size + pi.size - intersect;
    total += union > 0 ? intersect / union : 0;
  }
  return n > 0 ? total / n : 0;
}

export function labelRankingAveragePrecisionExt(y: number[][], yScore: Float64Array[][]): number {
  const n = y.length;
  let ap = 0;
  for (let i = 0; i < n; i++) {
    const relevantSet = new Set(y[i]);
    const scores = yScore[i] as Float64Array;
    const sortedIdx = Array.from({ length: scores.length }, (_, j) => j)
      .sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    let cumRel = 0, sumAP = 0, rank = 0;
    for (const idx of sortedIdx) {
      rank++;
      if (relevantSet.has(idx)) {
        cumRel++;
        sumAP += cumRel / rank;
      }
    }
    ap += relevantSet.size > 0 ? sumAP / relevantSet.size : 0;
  }
  return n > 0 ? ap / n : 0;
}

export function multiLabelF1(y: Int32Array[], yPred: Int32Array[], average: "micro" | "macro" | "samples" = "micro"): number {
  const n = y.length;
  if (n === 0) return 0;
  const nLabels = y[0]?.length ?? 0;

  if (average === "micro") {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nLabels; j++) {
        const yt = y[i]?.[j] ?? 0;
        const yp = yPred[i]?.[j] ?? 0;
        if (yt === 1 && yp === 1) tp++;
        else if (yt === 0 && yp === 1) fp++;
        else if (yt === 1 && yp === 0) fn++;
      }
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    return prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  }

  if (average === "macro") {
    let totalF1 = 0;
    for (let j = 0; j < nLabels; j++) {
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < n; i++) {
        const yt = y[i]?.[j] ?? 0;
        const yp = yPred[i]?.[j] ?? 0;
        if (yt === 1 && yp === 1) tp++;
        else if (yt === 0 && yp === 1) fp++;
        else if (yt === 1 && yp === 0) fn++;
      }
      const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
      const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
      totalF1 += prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
    }
    return nLabels > 0 ? totalF1 / nLabels : 0;
  }

  // samples
  let totalF1 = 0;
  for (let i = 0; i < n; i++) {
    let tp = 0, fp = 0, fn = 0;
    for (let j = 0; j < nLabels; j++) {
      const yt = y[i]?.[j] ?? 0;
      const yp = yPred[i]?.[j] ?? 0;
      if (yt === 1 && yp === 1) tp++;
      else if (yt === 0 && yp === 1) fp++;
      else if (yt === 1 && yp === 0) fn++;
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    totalF1 += prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  }
  return totalF1 / n;
}

export function exactMatchRatio(y: Int32Array[], yPred: Int32Array[]): number {
  const n = y.length;
  let matches = 0;
  for (let i = 0; i < n; i++) {
    const row = y[i] as Int32Array;
    const predRow = yPred[i] as Int32Array;
    if (row.length === predRow.length && row.every((v, j) => v === predRow[j])) matches++;
  }
  return n > 0 ? matches / n : 0;
}

export function subsetAccuracy(y: Int32Array[], yPred: Int32Array[]): number {
  return exactMatchRatio(y, yPred);
}

export function multiLabelConfusionMatrix(y: Int32Array[], yPred: Int32Array[]): Array<{ tp: number; fp: number; tn: number; fn: number }> {
  const nLabels = y[0]?.length ?? 0;
  const n = y.length;
  return Array.from({ length: nLabels }, (_, j) => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < n; i++) {
      const yt = y[i]?.[j] ?? 0;
      const yp = yPred[i]?.[j] ?? 0;
      if (yt === 1 && yp === 1) tp++;
      else if (yt === 0 && yp === 1) fp++;
      else if (yt === 0 && yp === 0) tn++;
      else fn++;
    }
    return { tp, fp, tn, fn };
  });
}

export function coverageError(y: Int32Array[], yScore: Float64Array[][]): number {
  const n = y.length;
  const nLabels = y[0]?.length ?? 0;
  let totalCoverage = 0;
  for (let i = 0; i < n; i++) {
    const relevantLabels = Array.from({ length: nLabels }, (_, j) => j).filter((j) => (y[i]?.[j] ?? 0) === 1);
    if (relevantLabels.length === 0) continue;
    const scores = yScore[i] as Float64Array;
    const sortedIdx = Array.from({ length: scores.length }, (_, j) => j).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    const maxRank = Math.max(...relevantLabels.map((j) => sortedIdx.indexOf(j) + 1));
    totalCoverage += maxRank;
  }
  return n > 0 ? totalCoverage / n : 0;
}

export function ndcgScoreMultiLabel(y: Int32Array[], yScore: Float64Array[][], k: number | null = null): number {
  const n = y.length;
  const nLabels = y[0]?.length ?? 0;
  const topK = k ?? nLabels;
  let totalNdcg = 0;
  for (let i = 0; i < n; i++) {
    const scores = yScore[i] as Float64Array;
    const sortedIdx = Array.from({ length: scores.length }, (_, j) => j).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0)).slice(0, topK);
    let dcg = 0;
    for (let rank = 0; rank < sortedIdx.length; rank++) {
      dcg += (y[i]?.[sortedIdx[rank] as number] ?? 0) / Math.log2(rank + 2);
    }
    const idealLabels = Array.from({ length: nLabels }, (_, j) => y[i]?.[j] ?? 0).sort((a, b) => b - a).slice(0, topK);
    let idcg = 0;
    for (let rank = 0; rank < idealLabels.length; rank++) {
      idcg += (idealLabels[rank] ?? 0) / Math.log2(rank + 2);
    }
    totalNdcg += idcg > 0 ? dcg / idcg : 0;
  }
  return n > 0 ? totalNdcg / n : 0;
}
