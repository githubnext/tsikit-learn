/**
 * Top-K accuracy and precision-recall area metrics.
 */

export function topKAccuracyScore(yTrue: Int32Array, yScores: Float64Array[], k = 3): number {
  const n = yTrue.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const scores = Array.from(yScores.map(s => s[i] ?? 0));
    const topK = scores
      .map((v, j) => ({ v, j }))
      .sort((a, b) => b.v - a.v)
      .slice(0, k)
      .map(x => x.j);
    if (topK.includes(yTrue[i] ?? 0)) correct++;
  }
  return n > 0 ? correct / n : 0;
}

export function precisionRecallAreaUnderCurve(yTrue: Int32Array, yScore: Float64Array): number {
  const n = yTrue.length;
  const sorted = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tp = 0, fp = 0;
  let prevP = 0, prevR = 0, area = 0;
  const totalPos = Array.from(yTrue).filter(v => v === 1).length;
  for (const idx of sorted) {
    if (yTrue[idx] === 1) tp++; else fp++;
    const p = tp / (tp + fp);
    const r = totalPos > 0 ? tp / totalPos : 0;
    area += (r - prevR) * (p + prevP) / 2;
    prevP = p; prevR = r;
  }
  return area;
}

export function averagePrecisionScore(yTrue: Int32Array, yScore: Float64Array): number {
  return precisionRecallAreaUnderCurve(yTrue, yScore);
}

export function dcgScoreExt(yTrue: Float64Array, yScore: Float64Array, k = 10): number {
  const n = Math.min(yTrue.length, yScore.length);
  const indices = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0))
    .slice(0, k);
  return indices.reduce((sum, idx, i) => sum + (yTrue[idx] ?? 0) / Math.log2(i + 2), 0);
}
