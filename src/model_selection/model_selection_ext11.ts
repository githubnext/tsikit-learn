/**
 * Cross-validation utilities: cross_val_score, cross_validate, learning curves.
 */

export interface CrossValidateResult {
  testScores: Float64Array;
  trainScores?: Float64Array;
  fitTime: Float64Array;
  scoreTime: Float64Array;
}

export interface Estimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
  predict?(X: Float64Array[]): Float64Array | Int32Array;
}

export function crossValScore(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  cv = 5
): Float64Array {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  const scores = new Float64Array(cv);
  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize, end = fold === cv - 1 ? n : start + foldSize;
    const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
    const trainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
    const XTrain = trainIdx.map(i => X[i]!);
    const yTrain = y instanceof Int32Array
      ? new Int32Array(trainIdx.map(i => y[i] ?? 0))
      : new Float64Array(trainIdx.map(i => y[i] ?? 0));
    const XTest = testIdx.map(i => X[i]!);
    const yTest = y instanceof Int32Array
      ? new Int32Array(testIdx.map(i => y[i] ?? 0))
      : new Float64Array(testIdx.map(i => y[i] ?? 0));
    estimator.fit(XTrain, yTrain);
    scores[fold] = estimator.score(XTest, yTest);
  }
  return scores;
}

export function crossValidate(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  cv = 5,
  returnTrainScore = false
): CrossValidateResult {
  const n = X.length, foldSize = Math.floor(n / cv);
  const testScores = new Float64Array(cv);
  const trainScores = returnTrainScore ? new Float64Array(cv) : undefined;
  const fitTime = new Float64Array(cv);
  const scoreTime = new Float64Array(cv);

  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize, end = fold === cv - 1 ? n : start + foldSize;
    const trainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
    const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
    const XTrain = trainIdx.map(i => X[i]!);
    const yTrain = y instanceof Int32Array
      ? new Int32Array(trainIdx.map(i => y[i] ?? 0))
      : new Float64Array(trainIdx.map(i => y[i] ?? 0));
    const XTest = testIdx.map(i => X[i]!);
    const yTest = y instanceof Int32Array
      ? new Int32Array(testIdx.map(i => y[i] ?? 0))
      : new Float64Array(testIdx.map(i => y[i] ?? 0));

    const t0 = Date.now();
    estimator.fit(XTrain, yTrain);
    fitTime[fold] = (Date.now() - t0) / 1000;

    const t1 = Date.now();
    testScores[fold] = estimator.score(XTest, yTest);
    if (returnTrainScore && trainScores) trainScores[fold] = estimator.score(XTrain, yTrain);
    scoreTime[fold] = (Date.now() - t1) / 1000;
  }
  return { testScores, ...(trainScores !== undefined ? { trainScores } : {}), fitTime, scoreTime };
}

export function learningCurve(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  trainSizes: number[] = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
  cv = 5
): { trainSizes: number[]; trainScores: Float64Array[]; testScores: Float64Array[] } {
  const n = X.length;
  const trainScoresAll: Float64Array[] = [];
  const testScoresAll: Float64Array[] = [];
  const actualSizes: number[] = [];

  for (const frac of trainSizes) {
    const nTrain = Math.min(n, Math.max(2, Math.floor(frac <= 1 ? frac * n : frac)));
    actualSizes.push(nTrain);
    const cvTrainScores = new Float64Array(cv);
    const cvTestScores = new Float64Array(cv);
    const foldSize = Math.floor(n / cv);
    for (let fold = 0; fold < cv; fold++) {
      const start = fold * foldSize, end = fold === cv - 1 ? n : start + foldSize;
      const allTrainIdx = [...Array.from({ length: start }, (_, i) => i), ...Array.from({ length: n - end }, (_, i) => end + i)];
      const trainIdx = allTrainIdx.slice(0, nTrain);
      const testIdx = Array.from({ length: end - start }, (_, i) => start + i);
      const XTrain = trainIdx.map(i => X[i]!);
      const yTrain = y instanceof Int32Array ? new Int32Array(trainIdx.map(i => y[i] ?? 0)) : new Float64Array(trainIdx.map(i => y[i] ?? 0));
      const XTest = testIdx.map(i => X[i]!);
      const yTest = y instanceof Int32Array ? new Int32Array(testIdx.map(i => y[i] ?? 0)) : new Float64Array(testIdx.map(i => y[i] ?? 0));
      estimator.fit(XTrain, yTrain);
      cvTrainScores[fold] = estimator.score(XTrain, yTrain);
      cvTestScores[fold] = estimator.score(XTest, yTest);
    }
    trainScoresAll.push(cvTrainScores);
    testScoresAll.push(cvTestScores);
  }
  return { trainSizes: actualSizes, trainScores: trainScoresAll, testScores: testScoresAll };
}
