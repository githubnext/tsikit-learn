/**
 * Additional curve-based metrics: DCG, cumulative gain, detection error tradeoff.
 * Complements ranking.ts with additional curve utilities.
 */

/**
 * Discounted Cumulative Gain (DCG) score.
 * Mirrors sklearn.metrics.dcg_score.
 */
export function dcgScore(
  yTrue: Float64Array,
  yScore: Float64Array,
  k?: number,
  ignoreties = false,
): number {
  const n = yTrue.length;
  const limit = k ?? n;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let dcg = 0;
  for (let i = 0; i < Math.min(limit, n); i++) {
    const gain = (2 ** (yTrue[order[i]!] ?? 0)) - 1;
    dcg += gain / Math.log2(i + 2);
  }
  return dcg;
}

/**
 * Compute cumulative gain curve.
 * Returns percentiles (0→1) and cumulative gains.
 */
export function cumulativeGainCurve(
  yTrue: Float64Array,
  yScore: Float64Array,
): { percentiles: Float64Array; gains: Float64Array } {
  const n = yTrue.length;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const totalGain = Array.from(yTrue).reduce((s, v) => s + v, 0) || 1;
  const percentiles = new Float64Array(n + 1);
  const gains = new Float64Array(n + 1);
  let cumGain = 0;
  for (let i = 0; i < n; i++) {
    cumGain += yTrue[order[i]!] ?? 0;
    percentiles[i + 1] = (i + 1) / n;
    gains[i + 1] = cumGain / totalGain;
  }
  return { percentiles, gains };
}

/**
 * Detection Error Tradeoff (DET) curve.
 * Returns false negative rates, false positive rates, and thresholds.
 */
export function detCurve(
  yTrue: Int32Array | number[],
  yScore: Float64Array,
): { fnr: Float64Array; fpr: Float64Array; thresholds: Float64Array } {
  const n = yTrue.length;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const totalPos = Array.from(yTrue).filter((v) => v > 0).length;
  const totalNeg = n - totalPos;

  const fprs: number[] = [];
  const fnrs: number[] = [];
  const thresholds: number[] = [];
  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    const label = yTrue[order[i]!] ?? 0;
    if (label > 0) tp++;
    else fp++;
    fprs.push(fp / (totalNeg || 1));
    fnrs.push((totalPos - tp) / (totalPos || 1));
    thresholds.push(yScore[order[i]!] ?? 0);
  }

  return {
    fpr: new Float64Array(fprs),
    fnr: new Float64Array(fnrs),
    thresholds: new Float64Array(thresholds),
  };
}

/**
 * Compute top-k accuracy score.
 * Mirrors sklearn.metrics.top_k_accuracy_score.
 */
export function topKAccuracyScore(
  yTrue: Int32Array | number[],
  yScore: Float64Array[],
  k = 1,
): number {
  const n = yTrue.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const scores = yScore[i]!;
    const nClasses = scores.length;
    const order = Array.from({ length: nClasses }, (_, j) => j)
      .sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    const topK = order.slice(0, k);
    if (topK.includes(yTrue[i] as number)) correct++;
  }
  return correct / n;
}
