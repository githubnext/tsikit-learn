/**
 * Metrics extensions: average_precision_score, PrecisionRecallCurve, Cohen's Kappa
 * Port of sklearn.metrics extensions
 */

export function averagePrecisionScore(
  yTrue: Int32Array | number[],
  yScore: Float64Array | number[],
  posLabel = 1
): number {
  const n = yTrue.length;
  const pairs = Array.from({ length: n }, (_, i) => ({ score: yScore[i] ?? 0, label: (yTrue[i] ?? 0) === posLabel ? 1 : 0 }));
  pairs.sort((a, b) => b.score - a.score);
  let ap = 0;
  let tp = 0;
  let fp = 0;
  let prevPrec = 1;
  for (let i = 0; i < n; i++) {
    if (pairs[i]!.label === 1) {
      tp++;
      const prec = tp / (tp + fp + 1e-15);
      const rec = tp / (pairs.filter(p => p.label === 1).length + 1e-15);
      ap += (prec + prevPrec) / 2 * (rec - (tp - 1) / (pairs.filter(p => p.label === 1).length + 1e-15));
      prevPrec = prec;
    } else {
      fp++;
    }
  }
  return ap;
}

export function precisionRecallCurve(
  yTrue: Int32Array | number[],
  probas: Float64Array | number[],
  posLabel = 1
): { precision: Float64Array; recall: Float64Array; thresholds: Float64Array } {
  const n = yTrue.length;
  const pairs = Array.from({ length: n }, (_, i) => ({ score: probas[i] ?? 0, label: (yTrue[i] ?? 0) === posLabel ? 1 : 0 }));
  pairs.sort((a, b) => b.score - a.score);
  const nPos = pairs.filter(p => p.label === 1).length;
  const precisions: number[] = [1];
  const recalls: number[] = [0];
  const thresholds: number[] = [];
  let tp = 0;
  let fp = 0;
  for (let i = 0; i < n; i++) {
    if (pairs[i]!.label === 1) tp++;
    else fp++;
    precisions.push(tp / (tp + fp + 1e-15));
    recalls.push(tp / (nPos + 1e-15));
    thresholds.push(pairs[i]!.score);
  }
  return {
    precision: Float64Array.from(precisions),
    recall: Float64Array.from(recalls),
    thresholds: Float64Array.from(thresholds),
  };
}

export function cohensKappa(
  y1: Int32Array | number[],
  y2: Int32Array | number[],
  weights: "linear" | "quadratic" | null = null
): number {
  const n = y1.length;
  const classes = new Set<number>();
  for (let i = 0; i < n; i++) { classes.add(y1[i] ?? 0); classes.add(y2[i] ?? 0); }
  const classArr = [...classes].sort((a, b) => a - b);
  const c = classArr.length;
  const classIdx = new Map(classArr.map((cls, i) => [cls, i]));
  const conf = Array.from({ length: c }, () => new Float64Array(c));
  for (let i = 0; i < n; i++) {
    const r = classIdx.get(y1[i] ?? 0) ?? 0;
    const cc = classIdx.get(y2[i] ?? 0) ?? 0;
    conf[r]![cc] = (conf[r]![cc] ?? 0) + 1;
  }
  const rowSum = new Float64Array(c);
  const colSum = new Float64Array(c);
  for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) {
    rowSum[i] = (rowSum[i] ?? 0) + (conf[i]![j] ?? 0);
    colSum[j] = (colSum[j] ?? 0) + (conf[i]![j] ?? 0);
  }
  const weightMat = Array.from({ length: c }, (_, i) => Float64Array.from({ length: c }, (_, j) => {
    if (!weights) return i === j ? 0 : 1;
    if (weights === "linear") return Math.abs(i - j) / (c - 1 + 1e-15);
    return ((i - j) / (c - 1 + 1e-15)) ** 2;
  }));
  let po = 0;
  let pe = 0;
  for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) {
    const w = 1 - (weightMat[i]![j] ?? 0);
    po += w * (conf[i]![j] ?? 0) / n;
    pe += w * (rowSum[i] ?? 0) * (colSum[j] ?? 0) / (n * n);
  }
  return (po - pe) / (1 - pe + 1e-15);
}

export function balancedAccuracyScore(yTrue: Int32Array | number[], yPred: Int32Array | number[]): number {
  const classes = new Set<number>();
  for (const v of yTrue) classes.add(v);
  let total = 0;
  for (const c of classes) {
    let tp = 0;
    let support = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) {
        support++;
        if ((yPred[i] ?? 0) === c) tp++;
      }
    }
    if (support > 0) total += tp / support;
  }
  return total / (classes.size + 1e-15);
}

export function topKAccuracyScore(
  yTrue: Int32Array | number[],
  yScoreMatrix: Float64Array[],
  k = 5
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const scores = yScoreMatrix[i];
    if (!scores) continue;
    const topK = Array.from({ length: scores.length }, (_, j) => ({ idx: j, score: scores[j] ?? 0 }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(x => x.idx);
    if (topK.includes(yTrue[i] ?? 0)) correct++;
  }
  return correct / (yTrue.length + 1e-15);
}

export function hammingLoss(yTrue: Int32Array | number[], yPred: Int32Array | number[]): number {
  let mismatches = 0;
  for (let i = 0; i < yTrue.length; i++) if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) mismatches++;
  return mismatches / (yTrue.length + 1e-15);
}

export function zerOneScore(yTrue: Int32Array | number[], yPred: Int32Array | number[]): number {
  return 1 - hammingLoss(yTrue, yPred);
}

export function d2TweedieLoss(yTrue: Float64Array, yPred: Float64Array, power = 1.5): number {
  let loss = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const y = yTrue[i] ?? 0;
    const mu = Math.max(yPred[i] ?? 0, 1e-15);
    if (power === 1) {
      loss += y * Math.log((y + 1e-15) / mu) - (y - mu);
    } else if (power === 2) {
      loss += Math.log(mu / (y + 1e-15)) + (y - mu) / (mu + 1e-15);
    } else {
      loss += (y ** (2 - power)) / ((1 - power) * (2 - power)) - y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power);
    }
  }
  return 2 * loss / (yTrue.length + 1e-15);
}
