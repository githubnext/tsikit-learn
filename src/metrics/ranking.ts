/**
 * Ranking metrics: ROC-AUC, PR-AUC, average_precision_score.
 * Mirrors sklearn.metrics ranking metrics.
 */

export interface RocCurveResult {
  fpr: Float64Array;
  tpr: Float64Array;
  thresholds: Float64Array;
}

export interface PrCurveResult {
  precision: Float64Array;
  recall: Float64Array;
  thresholds: Float64Array;
}

/**
 * Compute ROC curve (FPR, TPR, thresholds) for binary classification.
 */
export function rocCurve(
  yTrue: Int32Array | number[],
  yScore: Float64Array | number[],
  posLabel: number = 1,
): RocCurveResult {
  const n = yTrue.length;
  const pairs = Array.from({ length: n }, (_, i) => ({
    score: yScore[i] ?? 0,
    label: (yTrue[i] ?? 0) === posLabel ? 1 : 0,
  })).sort((a, b) => b.score - a.score);

  const nPos = pairs.filter((p) => p.label === 1).length;
  const nNeg = n - nPos;

  const fprs: number[] = [0];
  const tprs: number[] = [0];
  const thresholds: number[] = [1.0 + (pairs[0]?.score ?? 0)];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    if ((pairs[i]?.label ?? 0) === 1) tp++;
    else fp++;

    // Add point at each threshold change
    if (i === n - 1 || (pairs[i]?.score ?? 0) !== (pairs[i + 1]?.score ?? 0)) {
      fprs.push(nNeg > 0 ? fp / nNeg : 0);
      tprs.push(nPos > 0 ? tp / nPos : 0);
      thresholds.push(pairs[i]?.score ?? 0);
    }
  }

  return {
    fpr: new Float64Array(fprs),
    tpr: new Float64Array(tprs),
    thresholds: new Float64Array(thresholds),
  };
}

/**
 * Compute Area Under the ROC Curve (AUC-ROC).
 */
export function rocAucScore(
  yTrue: Int32Array | number[],
  yScore: Float64Array | number[],
  posLabel: number = 1,
): number {
  const { fpr, tpr } = rocCurve(yTrue, yScore, posLabel);
  return _auc(fpr, tpr);
}

function _auc(x: Float64Array, y: Float64Array): number {
  let area = 0;
  for (let i = 1; i < x.length; i++) {
    area += ((x[i] ?? 0) - (x[i - 1] ?? 0)) * ((y[i] ?? 0) + (y[i - 1] ?? 0)) / 2;
  }
  return Math.abs(area);
}

/**
 * Compute precision-recall curve.
 */
export function precisionRecallCurve(
  yTrue: Int32Array | number[],
  probas: Float64Array | number[],
  posLabel: number = 1,
): PrCurveResult {
  const n = yTrue.length;
  const pairs = Array.from({ length: n }, (_, i) => ({
    score: probas[i] ?? 0,
    label: (yTrue[i] ?? 0) === posLabel ? 1 : 0,
  })).sort((a, b) => b.score - a.score);

  const nPos = pairs.filter((p) => p.label === 1).length;

  const precisions: number[] = [];
  const recalls: number[] = [];
  const thresholds: number[] = [];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    if ((pairs[i]?.label ?? 0) === 1) tp++;
    else fp++;

    if (i === n - 1 || (pairs[i]?.score ?? 0) !== (pairs[i + 1]?.score ?? 0)) {
      precisions.push(tp / (tp + fp));
      recalls.push(nPos > 0 ? tp / nPos : 0);
      if (i < n - 1) thresholds.push(pairs[i]?.score ?? 0);
    }
  }

  precisions.push(1);
  recalls.push(0);

  return {
    precision: new Float64Array(precisions.reverse()),
    recall: new Float64Array(recalls.reverse()),
    thresholds: new Float64Array(thresholds.reverse()),
  };
}

/**
 * Compute average precision score (area under precision-recall curve).
 */
export function averagePrecisionScore(
  yTrue: Int32Array | number[],
  yScore: Float64Array | number[],
  posLabel: number = 1,
): number {
  const { precision, recall } = precisionRecallCurve(yTrue, yScore, posLabel);
  let ap = 0;
  for (let i = 1; i < recall.length; i++) {
    ap +=
      Math.abs((recall[i] ?? 0) - (recall[i - 1] ?? 0)) * (precision[i] ?? 0);
  }
  return ap;
}

/**
 * Compute AUC (area under curve) using the trapezoidal rule.
 */
export function auc(x: Float64Array | number[], y: Float64Array | number[]): number {
  const xArr = x instanceof Float64Array ? x : new Float64Array(x);
  const yArr = y instanceof Float64Array ? y : new Float64Array(y);
  return _auc(xArr, yArr);
}

/**
 * Compute NDCG (Normalized Discounted Cumulative Gain) at k.
 */
export function ndcgScore(
  yTrue: Float64Array | number[],
  yScore: Float64Array | number[],
  k?: number,
): number {
  const n = yTrue.length;
  const kk = k ?? n;

  const sortedByScore = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0))
    .slice(0, kk);

  const sortedByTrue = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (yTrue[b] ?? 0) - (yTrue[a] ?? 0))
    .slice(0, kk);

  const dcg = sortedByScore.reduce(
    (sum, idx, rank) =>
      sum + ((yTrue[idx] ?? 0) / Math.log2(rank + 2)),
    0,
  );

  const idealDcg = sortedByTrue.reduce(
    (sum, idx, rank) =>
      sum + ((yTrue[idx] ?? 0) / Math.log2(rank + 2)),
    0,
  );

  return idealDcg < 1e-10 ? 0 : dcg / idealDcg;
}
