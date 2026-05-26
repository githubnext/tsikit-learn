/**
 * Extended metrics: liftScore, gainCurve, detCurve, balancedAccuracy, topKAccuracy
 */

export function liftScore(yTrue: Int32Array, yScore: Float64Array, positiveLabel = 1): Float64Array {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const baseRate = Array.from(yTrue).filter((v) => v === positiveLabel).length / n;
  const lift = new Float64Array(n);
  let tp = 0;
  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === positiveLabel) tp++;
    const precision = tp / (i + 1);
    lift[i] = baseRate > 0 ? precision / baseRate : 0;
  }
  return lift;
}

export function gainCurve(yTrue: Int32Array, yScore: Float64Array, positiveLabel = 1): {
  percentages: Float64Array;
  gains: Float64Array;
} {
  const n = yTrue.length;
  const nPos = Array.from(yTrue).filter((v) => v === positiveLabel).length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const percentages = new Float64Array(n + 1);
  const gains = new Float64Array(n + 1);
  let tp = 0;
  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === positiveLabel) tp++;
    percentages[i + 1] = (i + 1) / n;
    gains[i + 1] = nPos > 0 ? tp / nPos : 0;
  }
  return { percentages, gains };
}

export function detCurve(yTrue: Int32Array, yScore: Float64Array): {
  fpr: Float64Array;
  fnr: Float64Array;
  thresholds: Float64Array;
} {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const nPos = Array.from(yTrue).filter((v) => v === 1).length;
  const nNeg = n - nPos;
  const fprs: number[] = [];
  const fnrs: number[] = [];
  const thresholds: number[] = [];
  let tp = 0, fp = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i]!;
    if ((yTrue[idx] ?? 0) === 1) tp++;
    else fp++;
    fprs.push(nNeg > 0 ? fp / nNeg : 0);
    fnrs.push(nPos > 0 ? (nPos - tp) / nPos : 0);
    thresholds.push(yScore[idx] ?? 0);
  }
  return {
    fpr: new Float64Array(fprs),
    fnr: new Float64Array(fnrs),
    thresholds: new Float64Array(thresholds),
  };
}

export function balancedAccuracyScore(
  yTrue: Int32Array,
  yPred: Int32Array,
  adjusted = false
): number {
  const classes = [...new Set(Array.from(yTrue))];
  const recallPerClass = classes.map((c) => {
    const truePositives = Array.from(yTrue).filter((_, i) => yTrue[i] === c && yPred[i] === c).length;
    const totalPositives = Array.from(yTrue).filter((v) => v === c).length;
    return totalPositives > 0 ? truePositives / totalPositives : 0;
  });
  const score = recallPerClass.reduce((a, b) => a + b, 0) / classes.length;
  if (adjusted) {
    const chance = 1 / classes.length;
    return (score - chance) / (1 - chance);
  }
  return score;
}

export function topKAccuracyScore(
  yTrue: Int32Array,
  yScore: Float64Array[],
  k = 5
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const scores = yScore[i] ?? new Float64Array(0);
    const topK = Array.from(scores)
      .map((v, j) => ({ v, j }))
      .sort((a, b) => b.v - a.v)
      .slice(0, k)
      .map((x) => x.j);
    if (topK.includes(yTrue[i] ?? -1)) correct++;
  }
  return correct / yTrue.length;
}

export function averagePrecisionScore(yTrue: Int32Array, yScore: Float64Array): number {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tp = 0, ap = 0;
  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) {
      tp++;
      ap += tp / (i + 1);
    }
  }
  const nPos = Array.from(yTrue).filter((v) => v === 1).length;
  return nPos > 0 ? ap / nPos : 0;
}

export function brierScoreMulticlass(yTrue: Int32Array, yProba: Float64Array[], nClasses: number): number {
  let score = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const proba = yProba[i] ?? new Float64Array(nClasses);
    for (let c = 0; c < nClasses; c++) {
      const indicator = (yTrue[i] ?? -1) === c ? 1 : 0;
      score += (indicator - (proba[c] ?? 0)) ** 2;
    }
  }
  return score / yTrue.length;
}

export function rocAucMulticlass(
  yTrue: Int32Array,
  yScore: Float64Array[],
  average: "macro" | "weighted" = "macro"
): number {
  const classes = [...new Set(Array.from(yTrue))].sort((a, b) => a - b);
  const n = yTrue.length;
  const aucs = classes.map((c) => {
    const binaryTrue = new Int32Array(yTrue.map((v) => (v === c ? 1 : 0)));
    const binaryScore = new Float64Array(yScore.map((proba) => proba[c] ?? 0));
    return aucFromBinary(binaryTrue, binaryScore);
  });
  if (average === "macro") return aucs.reduce((a, b) => a + b, 0) / aucs.length;
  const weights = classes.map((c) => Array.from(yTrue).filter((v) => v === c).length / n);
  return aucs.reduce((acc, auc, i) => acc + auc * (weights[i] ?? 0), 0);
}

function aucFromBinary(yTrue: Int32Array, yScore: Float64Array): number {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tp = 0, fp = 0, auc = 0;
  let prevFp = 0, prevTp = 0;
  const nPos = Array.from(yTrue).filter((v) => v === 1).length;
  const nNeg = n - nPos;
  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) tp++;
    else fp++;
    if (i === n - 1 || (yScore[indices[i]!] ?? 0) !== (yScore[indices[i + 1]!] ?? 0)) {
      auc += (fp - prevFp) * (tp + prevTp) / 2;
      prevFp = fp; prevTp = tp;
    }
  }
  return nPos > 0 && nNeg > 0 ? auc / (nPos * nNeg) : 0;
}
