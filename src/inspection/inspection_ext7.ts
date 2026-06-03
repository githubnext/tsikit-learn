/**
 * Extended inspection utilities: feature importance, model complexity.
 * Port of sklearn.inspection extensions.
 */

/**
 * H-statistic for measuring feature interaction strength.
 * Friedman's H-statistic for detecting feature interactions.
 */
export function hStatistic(
  model: { predict(X: Float64Array[]): Float64Array },
  X: Float64Array[],
  featureIdx1: number,
  featureIdx2: number,
  nGrid = 20,
): number {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;

  // Get unique values for each feature
  const vals1 = Array.from(new Set(Array.from(X, row => row[featureIdx1] ?? 0))).sort((a, b) => a - b);
  const vals2 = Array.from(new Set(Array.from(X, row => row[featureIdx2] ?? 0))).sort((a, b) => a - b);

  // Sample grid points
  const grid1 = sampleGrid(vals1, nGrid);
  const grid2 = sampleGrid(vals2, nGrid);

  // PDP for feature pair
  let pdpJointVar = 0;
  const pdpJoint: number[] = [];

  for (const v1 of grid1) {
    for (const v2 of grid2) {
      let sum = 0;
      for (const row of X) {
        const modified = Float64Array.from({ length: nFeatures }, (_, j) => {
          if (j === featureIdx1) return v1;
          if (j === featureIdx2) return v2;
          return row[j] ?? 0;
        });
        const pred = model.predict([modified]);
        sum += pred[0] ?? 0;
      }
      pdpJoint.push(sum / n);
    }
  }

  // PDPs for individual features
  const pdp1: number[] = grid1.map(v => {
    let sum = 0;
    for (const row of X) {
      const modified = Float64Array.from({ length: nFeatures }, (_, j) => j === featureIdx1 ? v : row[j] ?? 0);
      sum += (model.predict([modified])[0] ?? 0);
    }
    return sum / n;
  });

  const pdp2: number[] = grid2.map(v => {
    let sum = 0;
    for (const row of X) {
      const modified = Float64Array.from({ length: nFeatures }, (_, j) => j === featureIdx2 ? v : row[j] ?? 0);
      sum += (model.predict([modified])[0] ?? 0);
    }
    return sum / n;
  });

  // Mean of pdps
  const mean1 = pdp1.reduce((s, v) => s + v, 0) / pdp1.length;
  const mean2 = pdp2.reduce((s, v) => s + v, 0) / pdp2.length;

  // H-statistic numerator
  let h2num = 0; let h2den = 0;
  for (let i = 0; i < grid1.length; i++) {
    for (let j = 0; j < grid2.length; j++) {
      const joint = pdpJoint[i * grid2.length + j] ?? 0;
      const ind1 = (pdp1[i] ?? 0) - mean1;
      const ind2 = (pdp2[j] ?? 0) - mean2;
      const diff = joint - ind1 - ind2;
      h2num += diff * diff;
      h2den += joint * joint;
    }
  }

  return h2den > 0 ? Math.sqrt(h2num / h2den) : 0;
}

function sampleGrid(vals: number[], nGrid: number): number[] {
  if (vals.length <= nGrid) return vals;
  const step = (vals.length - 1) / (nGrid - 1);
  return Array.from({ length: nGrid }, (_, i) => vals[Math.round(i * step)] ?? 0);
}

/**
 * Accumulated Local Effects (ALE) plot — unbiased alternative to PDP.
 */
export function accumulatedLocalEffects(
  model: { predict(X: Float64Array[]): Float64Array },
  X: Float64Array[],
  featureIdx: number,
  nBins = 20,
): { binCenters: Float64Array; ale: Float64Array } {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;

  // Sort by feature value
  const sorted = Array.from(X).sort((a, b) => (a[featureIdx] ?? 0) - (b[featureIdx] ?? 0));
  const sortedVals = sorted.map(row => row[featureIdx] ?? 0);

  const minVal = sortedVals[0] ?? 0;
  const maxVal = sortedVals[sortedVals.length - 1] ?? 1;
  const binEdges = Float64Array.from({ length: nBins + 1 }, (_, k) => minVal + k * (maxVal - minVal) / nBins);

  const ale = new Float64Array(nBins);
  const binCounts = new Int32Array(nBins);

  for (let b = 0; b < nBins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;

    // Find samples in this bin
    const binSamples = sorted.filter(row => (row[featureIdx] ?? 0) >= lo && (row[featureIdx] ?? 0) < hi);
    if (binSamples.length === 0) continue;

    binCounts[b] = binSamples.length;

    // Compute effect as difference in predictions at bin edges
    const xLo = binSamples.map(row => Float64Array.from({ length: nFeatures }, (_, j) => j === featureIdx ? lo : row[j] ?? 0));
    const xHi = binSamples.map(row => Float64Array.from({ length: nFeatures }, (_, j) => j === featureIdx ? hi : row[j] ?? 0));

    const predLo = model.predict(xLo);
    const predHi = model.predict(xHi);

    let sumEffect = 0;
    for (let i = 0; i < binSamples.length; i++) sumEffect += (predHi[i] ?? 0) - (predLo[i] ?? 0);
    ale[b] = sumEffect / binSamples.length;
  }

  // Accumulate
  for (let b = 1; b < nBins; b++) ale[b] = (ale[b] ?? 0) + (ale[b - 1] ?? 0);

  // Center
  const mean = ale.reduce((s, v) => s + v, 0) / nBins;
  for (let b = 0; b < nBins; b++) ale[b] = (ale[b] ?? 0) - mean;

  const binCenters = Float64Array.from({ length: nBins }, (_, k) => ((binEdges[k] ?? 0) + (binEdges[k + 1] ?? 0)) / 2);
  return { binCenters, ale };
}

/**
 * Model complexity metrics.
 */
export function modelComplexityMetrics(
  model: {
    getParams?(): Record<string, unknown>;
    coef_?: Float64Array | Float64Array[];
    featureImportances_?: Float64Array;
  },
): {
  nParameters: number;
  sparsity: number;
  effectiveFeatures: number;
} {
  let nParams = 0;
  let nZero = 0;

  if (model.coef_) {
    const coef = model.coef_;
    if (coef instanceof Float64Array) {
      nParams = coef.length;
      for (let i = 0; i < coef.length; i++) if (Math.abs(coef[i] ?? 0) < 1e-10) nZero++;
    } else {
      for (const row of coef) {
        nParams += row.length;
        for (let j = 0; j < row.length; j++) if (Math.abs(row[j] ?? 0) < 1e-10) nZero++;
      }
    }
  }

  const sparsity = nParams > 0 ? nZero / nParams : 0;

  let effectiveFeatures = 0;
  if (model.featureImportances_) {
    const fi = model.featureImportances_;
    const total = fi.reduce((s, v) => s + v, 0);
    let cumSum = 0;
    const sorted = Float64Array.from(fi).sort((a, b) => b - a);
    for (let i = 0; i < sorted.length; i++) {
      cumSum += (sorted[i] ?? 0) / total;
      effectiveFeatures = i + 1;
      if (cumSum >= 0.99) break;
    }
  }

  return { nParameters: nParams, sparsity, effectiveFeatures };
}
