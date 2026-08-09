/**
 * Bootstrap confidence intervals and permutation tests for metrics.
 * Mirrors sklearn.utils.resample-based validation utilities.
 */

export interface BootstrapCIResult {
  estimate: number;
  lower: number;
  upper: number;
  std: number;
  nBootstrap: number;
}

/**
 * Compute bootstrap confidence interval for a metric function.
 */
export function bootstrapCI(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  metricFn: (
    yTrue: Float64Array | Int32Array,
    yPred: Float64Array | Int32Array,
  ) => number,
  options: {
    nBootstrap?: number;
    confidenceLevel?: number;
    randomState?: number;
  } = {},
): BootstrapCIResult {
  const nBootstrap = options.nBootstrap ?? 1000;
  const alpha = 1 - (options.confidenceLevel ?? 0.95);
  const n = yTrue.length;

  let seed = options.randomState ?? 42;
  function lcg(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const baseEstimate = metricFn(yTrue, yPred);
  const bootstrapScores: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    const idxTrue =
      yTrue instanceof Float64Array ? new Float64Array(n) : new Int32Array(n);
    const idxPred =
      yPred instanceof Float64Array ? new Float64Array(n) : new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(lcg() * n);
      if (idxTrue instanceof Float64Array) {
        (idxTrue as Float64Array)[i] = yTrue[idx] ?? 0;
      } else {
        (idxTrue as Int32Array)[i] = (yTrue as Int32Array)[idx] ?? 0;
      }
      if (idxPred instanceof Float64Array) {
        (idxPred as Float64Array)[i] = yPred[idx] ?? 0;
      } else {
        (idxPred as Int32Array)[i] = (yPred as Int32Array)[idx] ?? 0;
      }
    }
    bootstrapScores.push(metricFn(idxTrue, idxPred));
  }

  bootstrapScores.sort((a, b) => a - b);
  const lower = bootstrapScores[Math.floor((alpha / 2) * nBootstrap)] ?? 0;
  const upper = bootstrapScores[Math.floor((1 - alpha / 2) * nBootstrap)] ?? 1;
  const mean = bootstrapScores.reduce((s, v) => s + v, 0) / nBootstrap;
  const variance =
    bootstrapScores.reduce((s, v) => s + (v - mean) ** 2, 0) / nBootstrap;

  return {
    estimate: baseEstimate,
    lower,
    upper,
    std: Math.sqrt(variance),
    nBootstrap,
  };
}

/**
 * Permutation test for a metric.
 * Tests whether the observed metric is significantly better than chance.
 */
export interface PermutationTestResult {
  score: number;
  permutationScores: Float64Array;
  pValue: number;
}

export function permutationTest(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  metricFn: (
    yTrue: Float64Array | Int32Array,
    yPred: Float64Array | Int32Array,
  ) => number,
  options: {
    nPermutations?: number;
    randomState?: number;
  } = {},
): PermutationTestResult {
  const nPermutations = options.nPermutations ?? 1000;
  const n = yTrue.length;
  let seed = options.randomState ?? 42;

  function lcg(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const score = metricFn(yTrue, yPred);
  const permScores = new Float64Array(nPermutations);

  // Permute yPred
  const shuffled =
    yPred instanceof Float64Array
      ? new Float64Array(yPred)
      : new Int32Array(yPred as Int32Array);
  for (let p = 0; p < nPermutations; p++) {
    // Fisher-Yates shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(lcg() * (i + 1));
      const tmp = shuffled[i]!;
      shuffled[i] = shuffled[j]!;
      shuffled[j] = tmp;
    }
    permScores[p] = metricFn(yTrue, shuffled);
  }

  let countAbove = 0;
  for (let p = 0; p < nPermutations; p++) {
    if ((permScores[p] ?? 0) >= score) countAbove++;
  }
  const pValue = (countAbove + 1) / (nPermutations + 1);

  return { score, permutationScores: permScores, pValue };
}
