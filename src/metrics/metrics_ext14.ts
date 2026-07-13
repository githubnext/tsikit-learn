/**
 * Fairness metrics — sklearn fairness / bias evaluation metrics port.
 */

export function demographicParityDifference(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const rates = groups.map((g) => {
    const mask = Array.from(sensitiveAttr).map((s) => s === g);
    const total = mask.filter(Boolean).length;
    const positive = mask.filter((m, i) => m && yPred[i] === 1).length;
    return total > 0 ? positive / total : 0;
  });
  return rates.length >= 2 ? Math.max(...rates) - Math.min(...rates) : 0;
}

export function demographicParityRatio(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const rates = groups.map((g) => {
    const mask = Array.from(sensitiveAttr).map((s) => s === g);
    const total = mask.filter(Boolean).length;
    const positive = mask.filter((m, i) => m && yPred[i] === 1).length;
    return total > 0 ? positive / total : 0;
  });
  const maxRate = Math.max(...rates);
  const minRate = Math.min(...rates);
  return maxRate > 0 ? minRate / maxRate : 0;
}

export function equalizedOddsDifference(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const tprDiff = (() => {
    const tprs = groups.map((g) => {
      const mask = Array.from(sensitiveAttr).map((s, i) => s === g && y[i] === 1);
      const total = mask.filter(Boolean).length;
      const tp = mask.filter((m, i) => m && yPred[i] === 1).length;
      return total > 0 ? tp / total : 0;
    });
    return tprs.length >= 2 ? Math.max(...tprs) - Math.min(...tprs) : 0;
  })();
  const fprDiff = (() => {
    const fprs = groups.map((g) => {
      const mask = Array.from(sensitiveAttr).map((s, i) => s === g && y[i] === 0);
      const total = mask.filter(Boolean).length;
      const fp = mask.filter((m, i) => m && yPred[i] === 1).length;
      return total > 0 ? fp / total : 0;
    });
    return fprs.length >= 2 ? Math.max(...fprs) - Math.min(...fprs) : 0;
  })();
  return Math.max(Math.abs(tprDiff), Math.abs(fprDiff));
}

export function equalOpportunityDifference(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const tprs = groups.map((g) => {
    const mask = Array.from(sensitiveAttr).map((s, i) => s === g && y[i] === 1);
    const total = mask.filter(Boolean).length;
    const tp = mask.filter((m, i) => m && yPred[i] === 1).length;
    return total > 0 ? tp / total : 0;
  });
  return tprs.length >= 2 ? Math.max(...tprs) - Math.min(...tprs) : 0;
}

export function predictiveParity(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const ppvs = groups.map((g) => {
    const mask = Array.from(sensitiveAttr).map((s, i) => s === g && yPred[i] === 1);
    const total = mask.filter(Boolean).length;
    const tp = mask.filter((m, i) => m && y[i] === 1).length;
    return total > 0 ? tp / total : 0;
  });
  return ppvs.length >= 2 ? Math.max(...ppvs) - Math.min(...ppvs) : 0;
}

export function disparateImpact(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  return demographicParityRatio(y, yPred, sensitiveAttr);
}

export function selectionRate(yPred: Int32Array, sensitiveAttr: Int32Array): Map<number, number> {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const rates = new Map<number, number>();
  for (const g of groups) {
    const mask = Array.from(sensitiveAttr).map((s) => s === g);
    const total = mask.filter(Boolean).length;
    const selected = mask.filter((m, i) => m && yPred[i] === 1).length;
    rates.set(g, total > 0 ? selected / total : 0);
  }
  return rates;
}

export function groupMetricDifference(
  metricFn: (y: Int32Array, yPred: Int32Array) => number,
  y: Int32Array,
  yPred: Int32Array,
  sensitiveAttr: Int32Array,
): number {
  const groups = Array.from(new Set(Array.from(sensitiveAttr)));
  const scores = groups.map((g) => {
    const idxs = Array.from({ length: y.length }, (_, i) => i).filter((i) => sensitiveAttr[i] === g);
    const yG = Int32Array.from(idxs, (i) => y[i] ?? 0);
    const yPredG = Int32Array.from(idxs, (i) => yPred[i] ?? 0);
    return metricFn(yG, yPredG);
  });
  return scores.length >= 2 ? Math.max(...scores) - Math.min(...scores) : 0;
}

export function biasAmplificationScore(y: Int32Array, yPred: Int32Array, sensitiveAttr: Int32Array): number {
  // Measure how much the model amplifies existing biases
  const dpData = demographicParityDifference(y, y, sensitiveAttr);
  const dpModel = demographicParityDifference(y, yPred, sensitiveAttr);
  return dpData > 0 ? dpModel / dpData : 1;
}

export function counterfactualFairness(
  modelFn: (X: Float64Array[]) => Int32Array,
  X: Float64Array[],
  sensitiveFeatureIdx: number,
  nCounterfactuals = 100,
): number {
  const n = Math.min(X.length, nCounterfactuals);
  const original = modelFn(X.slice(0, n));
  const counterfactual = X.slice(0, n).map((row) => {
    const r = new Float64Array(row);
    r[sensitiveFeatureIdx] = 1 - (r[sensitiveFeatureIdx] ?? 0);
    return r;
  });
  const cfPreds = modelFn(counterfactual);
  let diff = 0;
  for (let i = 0; i < n; i++) if (original[i] !== cfPreds[i]) diff++;
  return n > 0 ? diff / n : 0;
}
