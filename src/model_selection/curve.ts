/**
 * Learning curve and validation curve utilities.
 * Mirrors sklearn.model_selection.learning_curve and validation_curve.
 */

type Estimator = {
  fit(X: Float64Array[], y: Float64Array | Int32Array): unknown;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
};

type FoldSplit = { trainIndex: Int32Array; testIndex: Int32Array };
type Splitter = {
  split(X: Float64Array[], y?: Float64Array | Int32Array): Generator<FoldSplit>;
};

function makeSplits(
  X: Float64Array[],
  y: Float64Array | Int32Array | undefined,
  cvParam: number | Splitter,
): FoldSplit[] {
  if (typeof cvParam === "number") {
    const n = X.length;
    const k = cvParam;
    const splits: FoldSplit[] = [];
    const foldSize = Math.floor(n / k);
    for (let fold = 0; fold < k; fold++) {
      const start = fold * foldSize;
      const end = fold === k - 1 ? n : start + foldSize;
      const testIdx: number[] = [];
      const trainIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i >= start && i < end) testIdx.push(i);
        else trainIdx.push(i);
      }
      splits.push({
        trainIndex: new Int32Array(trainIdx),
        testIndex: new Int32Array(testIdx),
      });
    }
    return splits;
  }
  return Array.from(cvParam.split(X, y));
}

export interface CrossValidateResult {
  testScore: Float64Array;
  trainScore: Float64Array | null;
  fitTime: Float64Array;
  scoreTime: Float64Array;
}

export interface CrossValidateOptions {
  cv?: number | Splitter;
  scoring?: (
    estimator: Estimator,
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ) => number;
  returnTrainScore?: boolean;
}

/** Run cross-validation and return detailed results including fit/score times. */
export function crossValidate(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: CrossValidateOptions = {},
): CrossValidateResult {
  const cvParam = options.cv ?? 5;
  const scoring =
    options.scoring ?? ((est, Xtest, ytest) => est.score(Xtest, ytest));
  const returnTrainScore = options.returnTrainScore ?? false;
  const splits = makeSplits(X, y, cvParam);

  const testScores: number[] = [];
  const trainScores: number[] = [];
  const fitTimes: number[] = [];
  const scoreTimes: number[] = [];

  for (const { trainIndex, testIndex } of splits) {
    const Xtrain = Array.from(trainIndex).map(
      (i) => X[i] ?? new Float64Array(0),
    );
    const Xtest = Array.from(testIndex).map((i) => X[i] ?? new Float64Array(0));
    const ytrain =
      y instanceof Int32Array
        ? new Int32Array(Array.from(trainIndex).map((i) => y[i] ?? 0))
        : new Float64Array(Array.from(trainIndex).map((i) => y[i] ?? 0));
    const ytest =
      y instanceof Int32Array
        ? new Int32Array(Array.from(testIndex).map((i) => y[i] ?? 0))
        : new Float64Array(Array.from(testIndex).map((i) => y[i] ?? 0));

    const t0 = Date.now();
    estimator.fit(Xtrain, ytrain);
    fitTimes.push(Date.now() - t0);

    const t1 = Date.now();
    testScores.push(scoring(estimator, Xtest, ytest));
    scoreTimes.push(Date.now() - t1);

    if (returnTrainScore) trainScores.push(scoring(estimator, Xtrain, ytrain));
  }

  return {
    testScore: new Float64Array(testScores),
    trainScore: returnTrainScore ? new Float64Array(trainScores) : null,
    fitTime: new Float64Array(fitTimes),
    scoreTime: new Float64Array(scoreTimes),
  };
}

export interface LearningCurveOptions {
  cv?: number | Splitter;
  trainSizes?: Float64Array;
  scoring?: (
    estimator: Estimator,
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ) => number;
}

export interface LearningCurveResult {
  trainSizes: Int32Array;
  trainScores: Float64Array[];
  testScores: Float64Array[];
}

/** Compute learning curve: train/test scores at different training set sizes. */
export function learningCurve(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: LearningCurveOptions = {},
): LearningCurveResult {
  const trainSizeFractions =
    options.trainSizes ?? new Float64Array([0.1, 0.33, 0.55, 0.78, 1.0]);
  const cvParam = options.cv ?? 5;
  const scoring =
    options.scoring ?? ((est, Xtest, ytest) => est.score(Xtest, ytest));

  const n = X.length;
  const absoluteSizes = Array.from(trainSizeFractions).map((f) =>
    Math.max(1, Math.round(f * n)),
  );
  const splits = makeSplits(X, y, cvParam);

  const trainScoresBySize: Float64Array[] = [];
  const testScoresBySize: Float64Array[] = [];

  for (const sz of absoluteSizes) {
    const tsArr: number[] = [];
    const vsArr: number[] = [];
    for (const { trainIndex, testIndex } of splits) {
      const subTrain = Array.from(trainIndex).slice(0, sz);
      const Xtrain = subTrain.map((i) => X[i] ?? new Float64Array(0));
      const Xtest = Array.from(testIndex).map(
        (i) => X[i] ?? new Float64Array(0),
      );
      const ytrain =
        y instanceof Int32Array
          ? new Int32Array(subTrain.map((i) => y[i] ?? 0))
          : new Float64Array(subTrain.map((i) => y[i] ?? 0));
      const ytest =
        y instanceof Int32Array
          ? new Int32Array(Array.from(testIndex).map((i) => y[i] ?? 0))
          : new Float64Array(Array.from(testIndex).map((i) => y[i] ?? 0));

      estimator.fit(Xtrain, ytrain);
      tsArr.push(scoring(estimator, Xtrain, ytrain));
      vsArr.push(scoring(estimator, Xtest, ytest));
    }
    trainScoresBySize.push(new Float64Array(tsArr));
    testScoresBySize.push(new Float64Array(vsArr));
  }

  return {
    trainSizes: new Int32Array(absoluteSizes),
    trainScores: trainScoresBySize,
    testScores: testScoresBySize,
  };
}

export interface ValidationCurveOptions {
  cv?: number | Splitter;
  paramName: string;
  paramRange: number[];
  scoring?: (
    estimator: Estimator,
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ) => number;
}

export interface ValidationCurveResult {
  trainScores: Float64Array[];
  testScores: Float64Array[];
}

/** Compute validation curve over a range of parameter values. */
export function validationCurve(
  estimator: Estimator & Record<string, unknown>,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: ValidationCurveOptions,
): ValidationCurveResult {
  const { paramName, paramRange } = options;
  const cvParam = options.cv ?? 5;
  const scoring =
    options.scoring ?? ((est, Xtest, ytest) => est.score(Xtest, ytest));
  const splits = makeSplits(X, y, cvParam);

  const trainScores: Float64Array[] = [];
  const testScores: Float64Array[] = [];

  for (const pval of paramRange) {
    const origVal = estimator[paramName];
    estimator[paramName] = pval;

    const tsArr: number[] = [];
    const vsArr: number[] = [];
    for (const { trainIndex, testIndex } of splits) {
      const Xtrain = Array.from(trainIndex).map(
        (i) => X[i] ?? new Float64Array(0),
      );
      const Xtest = Array.from(testIndex).map(
        (i) => X[i] ?? new Float64Array(0),
      );
      const ytrain =
        y instanceof Int32Array
          ? new Int32Array(Array.from(trainIndex).map((i) => y[i] ?? 0))
          : new Float64Array(Array.from(trainIndex).map((i) => y[i] ?? 0));
      const ytest =
        y instanceof Int32Array
          ? new Int32Array(Array.from(testIndex).map((i) => y[i] ?? 0))
          : new Float64Array(Array.from(testIndex).map((i) => y[i] ?? 0));

      estimator.fit(Xtrain, ytrain);
      tsArr.push(scoring(estimator, Xtrain, ytrain));
      vsArr.push(scoring(estimator, Xtest, ytest));
    }

    trainScores.push(new Float64Array(tsArr));
    testScores.push(new Float64Array(vsArr));
    estimator[paramName] = origVal;
  }

  return { trainScores, testScores };
}
