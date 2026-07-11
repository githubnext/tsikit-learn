/**
 * Permutation importance — ported from sklearn.inspection
 */

export interface PermutationImportanceOptions {
  /** Number of times to permute each feature */
  nRepeats?: number;
  /** Random state seed for reproducibility */
  randomState?: number | null;
  /** Sample weight array */
  sampleWeight?: Float64Array | null;
  /** Scoring function: takes (yTrue, yPred) and returns a number (higher = better) */
  scoring?:
    | ((
        yTrue: Float64Array | Int32Array,
        yPred: Float64Array | Int32Array,
      ) => number)
    | null;
}

export interface PermutationImportanceResult {
  /** Mean importance for each feature (shape: nFeatures) */
  importancesMean: Float64Array;
  /** Standard deviation of importance for each feature */
  importancesStd: Float64Array;
  /** Raw importances matrix (shape: nFeatures x nRepeats) */
  importances: Float64Array[];
  /** Feature indices sorted by decreasing mean importance */
  sortedFeatureIndices: Int32Array;
}

type AnyEstimator = {
  predict(X: Float64Array[]): Float64Array | Int32Array;
  score?: (X: Float64Array[], y: Float64Array | Int32Array) => number;
};

function defaultAccuracy(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) === (yPred[i] ?? 0)) correct++;
  }
  return correct / yTrue.length;
}

/**
 * Compute permutation importance for a fitted estimator.
 *
 * For each feature, the feature values are randomly permuted multiple times
 * and the drop in model score is recorded as the importance.
 *
 * @param estimator A fitted estimator with a predict method
 * @param X Validation data (nSamples x nFeatures)
 * @param y True labels/values for validation data
 * @param options Configuration options
 */
export function permutationImportance(
  estimator: AnyEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: PermutationImportanceOptions = {},
): PermutationImportanceResult {
  const nRepeats = options.nRepeats ?? 5;
  const scoring = options.scoring ?? defaultAccuracy;

  let rng = options.randomState ?? Math.floor(Math.random() * 2 ** 31);

  function nextRng(): number {
    rng = (1664525 * rng + 1013904223) & 0x7fffffff;
    return rng;
  }

  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;

  // Baseline score
  const baselinePred = estimator.predict(X);
  const baselineScore = scoring(y, baselinePred);

  // Compute importances for each feature
  const importances: Float64Array[] = [];

  for (let f = 0; f < nFeatures; f++) {
    const featureImportances = new Float64Array(nRepeats);

    for (let r = 0; r < nRepeats; r++) {
      // Save original values
      const original = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        original[i] = X[i]![f] ?? 0;
      }

      // Fisher-Yates shuffle of the feature column
      const permuted = original.slice();
      for (let i = nSamples - 1; i > 0; i--) {
        const j = nextRng() % (i + 1);
        const tmp = permuted[i]!;
        permuted[i] = permuted[j]!;
        permuted[j] = tmp;
      }

      // Apply permutation
      for (let i = 0; i < nSamples; i++) {
        X[i]![f] = permuted[i]!;
      }

      // Score with permuted feature
      const permutedPred = estimator.predict(X);
      const permutedScore = scoring(y, permutedPred);
      featureImportances[r] = baselineScore - permutedScore;

      // Restore original values
      for (let i = 0; i < nSamples; i++) {
        X[i]![f] = original[i]!;
      }
    }

    importances.push(featureImportances);
  }

  // Compute mean and std
  const importancesMean = new Float64Array(nFeatures);
  const importancesStd = new Float64Array(nFeatures);

  for (let f = 0; f < nFeatures; f++) {
    const fi = importances[f]!;
    let sum = 0;
    for (let r = 0; r < nRepeats; r++) sum += fi[r] ?? 0;
    const mean = sum / nRepeats;
    importancesMean[f] = mean;

    let varSum = 0;
    for (let r = 0; r < nRepeats; r++) {
      const d = (fi[r] ?? 0) - mean;
      varSum += d * d;
    }
    importancesStd[f] = Math.sqrt(varSum / nRepeats);
  }

  // Sort features by decreasing mean importance
  const sortedIndices = Array.from({ length: nFeatures }, (_, i) => i).sort(
    (a, b) => (importancesMean[b] ?? 0) - (importancesMean[a] ?? 0),
  );

  return {
    importancesMean,
    importancesStd,
    importances,
    sortedFeatureIndices: new Int32Array(sortedIndices),
  };
}
