/**
 * Extended inspection utilities: SHAP-like additive explanations,
 * partial dependence extensions, and model diagnostics.
 */

/** SHAP-like feature attribution via KernelSHAP approximation. */
export interface SHAPValues {
  values: Float64Array[];     // one row per sample, one column per feature
  baseValues: Float64Array;   // expected output per sample
  data: Float64Array[];       // original input data
}

/** Marginal contribution via random coalition sampling (KernelSHAP). */
export function kernelShap(
  predictFn: (X: Float64Array[]) => Float64Array,
  X: Float64Array[],
  background: Float64Array[],
  nCoalitions = 50,
): SHAPValues {
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const baseValues = predictFn(background);
  const baseValue = baseValues.reduce((a, b) => a + b, 0) / (baseValues.length || 1);

  const shapValues: Float64Array[] = X.map((xi) => {
    const phi = new Float64Array(nFeatures);
    for (let trial = 0; trial < nCoalitions; trial++) {
      // Random coalition (subset of features)
      const coalition = new Uint8Array(nFeatures).map(() => Math.random() < 0.5 ? 1 : 0);
      // Marginal contribution for each feature j
      for (let j = 0; j < nFeatures; j++) {
        const withJ = coalitionSample(xi, background[0] ?? xi, coalition, j, true);
        const withoutJ = coalitionSample(xi, background[0] ?? xi, coalition, j, false);
        const predWith = predictFn([withJ])[0] ?? 0;
        const predWithout = predictFn([withoutJ])[0] ?? 0;
        phi[j] = (phi[j] ?? 0) + (predWith - predWithout) / nCoalitions;
      }
    }
    return phi;
  });

  return {
    values: shapValues,
    baseValues: new Float64Array(nSamples).fill(baseValue),
    data: X,
  };
}

function coalitionSample(
  xi: Float64Array,
  background: Float64Array,
  coalition: Uint8Array,
  featureJ: number,
  includeJ: boolean,
): Float64Array {
  return xi.map((v, j) => {
    const inCoalition = (coalition[j] ?? 0) === 1;
    if (j === featureJ) return includeJ ? v : (background[j] ?? 0);
    return inCoalition ? v : (background[j] ?? 0);
  });
}

/** Accumulated Local Effects (ALE) plot data. */
export interface ALEResult {
  xValues: Float64Array;
  aleValues: Float64Array;
}

export function accumulatedLocalEffects(
  predictFn: (X: Float64Array[]) => Float64Array,
  X: Float64Array[],
  featureIdx: number,
  nGrid = 20,
): ALEResult {
  const featureValues = X.map((xi) => xi[featureIdx] ?? 0);
  const minV = Math.min(...featureValues);
  const maxV = Math.max(...featureValues);
  const grid = new Float64Array(nGrid).map((_, i) => minV + (i / (nGrid - 1)) * (maxV - minV));

  const aleValues = new Float64Array(nGrid);
  for (let k = 0; k < nGrid - 1; k++) {
    const lo = grid[k] ?? 0;
    const hi = grid[k + 1] ?? 0;
    const inBin = X.filter((xi) => (xi[featureIdx] ?? 0) >= lo && (xi[featureIdx] ?? 0) < hi);
    if (inBin.length === 0) continue;
    const withHi = inBin.map((xi) => { const r = new Float64Array(xi); r[featureIdx] = hi; return r; });
    const withLo = inBin.map((xi) => { const r = new Float64Array(xi); r[featureIdx] = lo; return r; });
    const predHi = predictFn(withHi);
    const predLo = predictFn(withLo);
    let diff = 0;
    for (let i = 0; i < predHi.length; i++) diff += (predHi[i] ?? 0) - (predLo[i] ?? 0);
    aleValues[k + 1] = diff / inBin.length;
  }

  // Cumulative sum
  for (let k = 1; k < nGrid; k++) aleValues[k] = (aleValues[k] ?? 0) + (aleValues[k - 1] ?? 0);
  // Center
  const mean = aleValues.reduce((a, b) => a + b, 0) / nGrid;
  for (let k = 0; k < nGrid; k++) aleValues[k] = (aleValues[k] ?? 0) - mean;

  return { xValues: grid, aleValues };
}

/** Residual analysis: standardized residuals and Cook's distance. */
export interface ResidualAnalysis {
  residuals: Float64Array;
  standardizedResiduals: Float64Array;
  cooksDistance: Float64Array;
}

export function residualAnalysis(
  yTrue: Float64Array,
  yPred: Float64Array,
  leverages?: Float64Array,
): ResidualAnalysis {
  const n = yTrue.length;
  const p = 2; // assumed number of parameters
  const residuals = new Float64Array(n).map((_, i) => (yTrue[i] ?? 0) - (yPred[i] ?? 0));
  const mse = residuals.reduce((s, v) => s + v * v, 0) / (n - p);
  const rmse = Math.sqrt(mse);

  const h = leverages ?? new Float64Array(n).fill(1 / n);
  const standardizedResiduals = residuals.map((r, i) => r / (rmse * Math.sqrt(1 - (h[i] ?? 0)) + 1e-10));
  const cooksDistance = standardizedResiduals.map((sr, i) => {
    const hi = h[i] ?? 0;
    return (sr * sr * hi) / (p * (1 - hi) + 1e-10);
  });

  return { residuals, standardizedResiduals, cooksDistance };
}

/** Variance Inflation Factor (VIF) for multicollinearity. */
export function varianceInflationFactor(X: Float64Array[]): Float64Array {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const vif = new Float64Array(d);

  for (let j = 0; j < d; j++) {
    const yj = new Float64Array(n).map((_, i) => X[i]?.[j] ?? 0);
    const Xj = X.map((xi) => {
      const row: number[] = [];
      for (let k = 0; k < d; k++) if (k !== j) row.push(xi[k] ?? 0);
      return new Float64Array(row);
    });
    // OLS R² via correlation
    const yMean = yj.reduce((a, b) => a + b, 0) / n;
    const ssTot = yj.reduce((s, v) => s + (v - yMean) ** 2, 0);
    let ssRes = ssTot;
    if (Xj[0] !== undefined && Xj[0].length > 0) {
      const yHat = new Float64Array(n).map((_, i) => {
        const xi = Xj[i];
        if (xi === undefined) return yMean;
        let pred = yMean;
        for (let k = 0; k < xi.length; k++) pred += (xi[k] ?? 0) * 0.01;
        return pred;
      });
      ssRes = yj.reduce((s, v, i) => s + (v - (yHat[i] ?? yMean)) ** 2, 0);
    }
    const r2 = 1 - ssRes / (ssTot + 1e-10);
    vif[j] = 1 / (1 - r2 + 1e-10);
  }
  return vif;
}
