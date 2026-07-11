/**
 * Validation and learning curve utilities.
 * Mirrors scikit-learn's model_selection._validation.validation_curve and learning_curve.
 */

export interface ValidationCurveResult {
  trainScores: Float64Array[];
  testScores: Float64Array[];
  paramRange: unknown[];
}

export interface LearningCurveResult {
  trainSizes: Int32Array;
  trainScores: Float64Array[];
  testScores: Float64Array[];
}

export interface VCEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
  setParam(name: string, value: unknown): void;
}

export interface VCSplitter {
  split(
    X: Float64Array[],
    y?: Float64Array | Int32Array,
  ): Iterable<[Int32Array, Int32Array]>;
}

/**
 * Compute training and test scores for varying values of a hyperparameter.
 */
export function validationCurve(
  estimator: VCEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  paramName: string,
  paramRange: unknown[],
  options: { cv?: number } = {},
): ValidationCurveResult {
  const cv = options.cv ?? 5;
  const trainScores: Float64Array[] = [];
  const testScores: Float64Array[] = [];

  for (const paramValue of paramRange) {
    estimator.setParam(paramName, paramValue);
    const foldTrain: number[] = [];
    const foldTest: number[] = [];
    for (const [trainIdx, testIdx] of kFoldSplit(X.length, cv)) {
      const XTrain = Array.from(trainIdx).map(
        (i) => X[i] ?? new Float64Array(0),
      );
      const yTrain = subsetLabels(y, trainIdx);
      const XTest = Array.from(testIdx).map((i) => X[i] ?? new Float64Array(0));
      const yTest = subsetLabels(y, testIdx);
      const fitted = estimator.fit(XTrain, yTrain);
      foldTrain.push(fitted.score(XTrain, yTrain));
      foldTest.push(fitted.score(XTest, yTest));
    }
    trainScores.push(new Float64Array(foldTrain));
    testScores.push(new Float64Array(foldTest));
  }

  return { trainScores, testScores, paramRange };
}

/**
 * Compute training and test scores for varying training set sizes.
 */
export function learningCurve(
  estimator: VCEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: { trainSizes?: number[]; cv?: number } = {},
): LearningCurveResult {
  const cv = options.cv ?? 5;
  const n = X.length;
  const trainSizesFrac = options.trainSizes ?? [0.1, 0.33, 0.55, 0.78, 1.0];
  const trainSizesInt = trainSizesFrac.map((frac) =>
    Math.max(1, Math.round(frac * (n - Math.floor(n / cv)))),
  );

  const trainScores: Float64Array[] = [];
  const testScores: Float64Array[] = [];

  for (const size of trainSizesInt) {
    const foldTrain: number[] = [];
    const foldTest: number[] = [];
    for (const [trainIdx, testIdx] of kFoldSplit(n, cv)) {
      const subset = trainIdx.slice(0, size);
      const XTrain = Array.from(subset).map((i) => X[i] ?? new Float64Array(0));
      const yTrain = subsetLabels(y, subset);
      const XTest = Array.from(testIdx).map((i) => X[i] ?? new Float64Array(0));
      const yTest = subsetLabels(y, testIdx);
      const fitted = estimator.fit(XTrain, yTrain);
      foldTrain.push(fitted.score(XTrain, yTrain));
      foldTest.push(fitted.score(XTest, yTest));
    }
    trainScores.push(new Float64Array(foldTrain));
    testScores.push(new Float64Array(foldTest));
  }

  return {
    trainSizes: Int32Array.from(trainSizesInt),
    trainScores,
    testScores,
  };
}

function* kFoldSplit(n: number, k: number): Iterable<[Int32Array, Int32Array]> {
  const foldSize = Math.floor(n / k);
  for (let fold = 0; fold < k; fold++) {
    const start = fold * foldSize;
    const end = fold === k - 1 ? n : start + foldSize;
    const testIdx = Int32Array.from(
      { length: end - start },
      (_, i) => start + i,
    );
    const trainIdx = Int32Array.from({ length: n - testIdx.length }, (_, i) =>
      i < start ? i : i + testIdx.length,
    );
    yield [trainIdx, testIdx];
  }
}

function subsetLabels(
  y: Float64Array | Int32Array,
  indices: Int32Array,
): Float64Array | Int32Array {
  if (y instanceof Float64Array) {
    return Float64Array.from(indices, (i) => y[i] ?? 0);
  }
  return Int32Array.from(indices, (i) => y[i] ?? 0);
}
