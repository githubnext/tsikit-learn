/**
 * Cross-validation utilities.
 * Mirrors scikit-learn's model_selection._validation.cross_validate and cross_val_predict.
 */

export interface CrossValidateResult {
  testScore: Float64Array;
  trainScore?: Float64Array;
  fitTime: Float64Array;
  scoreTime: Float64Array;
}

export interface CVEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  predict(X: Float64Array[]): Float64Array | Int32Array;
  score?(X: Float64Array[], y: Float64Array | Int32Array): number;
}

export interface CVSplitter {
  split(
    X: Float64Array[],
    y?: Float64Array | Int32Array,
  ): Iterable<[Int32Array, Int32Array]>;
}

/**
 * Evaluate an estimator by cross-validation.
 */
export function crossValidate(
  estimator: CVEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: {
    cv?: CVSplitter | number;
    returnTrainScore?: boolean;
    scoring?: (est: CVEstimator, X: Float64Array[], y: Float64Array | Int32Array) => number;
  } = {},
): CrossValidateResult {
  const { returnTrainScore = false, scoring } = options;
  const cv = options.cv ?? 5;

  const splits = typeof cv === "number"
    ? kFoldSplit(X.length, cv)
    : cv.split(X, y);

  const testScores: number[] = [];
  const trainScores: number[] = [];
  const fitTimes: number[] = [];
  const scoreTimes: number[] = [];

  for (const [trainIdx, testIdx] of splits) {
    const XTrain = Array.from(trainIdx).map((i) => X[i] ?? new Float64Array(0));
    const yTrain = subsetLabels(y, trainIdx);
    const XTest = Array.from(testIdx).map((i) => X[i] ?? new Float64Array(0));
    const yTest = subsetLabels(y, testIdx);

    const fitStart = Date.now();
    const fittedEst = estimator.fit(XTrain, yTrain);
    fitTimes.push((Date.now() - fitStart) / 1000);

    const scoreStart = Date.now();
    const testScore =
      scoring !== undefined
        ? scoring(fittedEst, XTest, yTest)
        : (fittedEst.score !== undefined ? fittedEst.score(XTest, yTest) : 0);
    scoreTimes.push((Date.now() - scoreStart) / 1000);
    testScores.push(testScore);

    if (returnTrainScore) {
      const trainScore =
        scoring !== undefined
          ? scoring(fittedEst, XTrain, yTrain)
          : (fittedEst.score !== undefined ? fittedEst.score(XTrain, yTrain) : 0);
      trainScores.push(trainScore);
    }
  }

  const result: CrossValidateResult = {
    testScore: new Float64Array(testScores),
    fitTime: new Float64Array(fitTimes),
    scoreTime: new Float64Array(scoreTimes),
  };
  if (returnTrainScore) {
    result.trainScore = new Float64Array(trainScores);
  }
  return result;
}

/**
 * Cross-validation score (mean of test scores).
 */
export function crossValScore(
  estimator: CVEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: { cv?: number; scoring?: (est: CVEstimator, X: Float64Array[], y: Float64Array | Int32Array) => number } = {},
): Float64Array {
  return crossValidate(estimator, X, y, options).testScore;
}

function* kFoldSplit(
  n: number,
  k: number,
): Iterable<[Int32Array, Int32Array]> {
  const foldSize = Math.floor(n / k);
  for (let fold = 0; fold < k; fold++) {
    const start = fold * foldSize;
    const end = fold === k - 1 ? n : start + foldSize;
    const testIdx = Int32Array.from({ length: end - start }, (_, i) => start + i);
    const trainIdx = Int32Array.from(
      { length: n - testIdx.length },
      (_, i) => {
        const idx = i < start ? i : i + testIdx.length;
        return idx;
      },
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
