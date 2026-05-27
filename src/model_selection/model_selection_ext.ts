/**
 * Additional model selection: RandomizedSearchCV, cross_val_predict.
 * Mirrors sklearn.model_selection extras.
 */

import { NotFittedError } from "../exceptions.js";

export type ParamGrid = Record<string, unknown[]>;

export function* randomizedParamSampler(
  paramGrid: ParamGrid,
  nIter: number,
  randomState = 0,
): Generator<Record<string, unknown>> {
  const keys = Object.keys(paramGrid);
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };

  for (let i = 0; i < nIter; i++) {
    const params: Record<string, unknown> = {};
    for (const key of keys) {
      const values = paramGrid[key] ?? [];
      params[key] = values[Math.floor(nextRand() * values.length)];
    }
    yield params;
  }
}

export interface CVEstimator {
  fit(X: Float64Array[], y: Int32Array | Float64Array): CVEstimator;
  predict(X: Float64Array[]): Int32Array | Float64Array;
  score(X: Float64Array[], y: Int32Array | Float64Array): number;
  setParams?(params: Record<string, unknown>): void;
}

export function crossValScore(
  estimator: CVEstimator,
  X: Float64Array[],
  y: Int32Array | Float64Array,
  cv = 5,
  scoring?: (yTrue: Int32Array | Float64Array, yPred: Int32Array | Float64Array) => number,
): Float64Array {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  const scores = new Float64Array(cv);

  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize;
    const end = fold === cv - 1 ? n : start + foldSize;

    const trainX = [...X.slice(0, start), ...X.slice(end)];
    const testX = X.slice(start, end);

    let trainY: Int32Array | Float64Array;
    let testY: Int32Array | Float64Array;

    if (y instanceof Int32Array) {
      trainY = new Int32Array([...Array.from(y.slice(0, start)), ...Array.from(y.slice(end))]);
      testY = y.slice(start, end);
    } else {
      trainY = new Float64Array([...Array.from(y.slice(0, start)), ...Array.from(y.slice(end))]);
      testY = y.slice(start, end);
    }

    estimator.fit(trainX, trainY);
    const yPred = estimator.predict(testX);
    scores[fold] = scoring ? scoring(testY, yPred) : estimator.score(testX, testY);
  }

  return scores;
}

export function crossValPredict(
  estimator: CVEstimator,
  X: Float64Array[],
  y: Int32Array | Float64Array,
  cv = 5,
): Int32Array | Float64Array {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  const isClassification = y instanceof Int32Array;
  const predictions = isClassification ? new Int32Array(n) : new Float64Array(n);

  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize;
    const end = fold === cv - 1 ? n : start + foldSize;

    const trainX = [...X.slice(0, start), ...X.slice(end)];
    const testX = X.slice(start, end);

    let trainY: Int32Array | Float64Array;
    if (y instanceof Int32Array) {
      trainY = new Int32Array([...Array.from(y.slice(0, start)), ...Array.from(y.slice(end))]);
    } else {
      trainY = new Float64Array([...Array.from(y.slice(0, start)), ...Array.from(y.slice(end))]);
    }

    estimator.fit(trainX, trainY);
    const yPred = estimator.predict(testX);

    for (let i = 0; i < testX.length; i++) {
      (predictions as Int32Array | Float64Array)[start + i] = yPred[i] ?? 0;
    }
  }

  return predictions;
}

export class GridSearchCVExt {
  estimator: CVEstimator;
  paramGrid: ParamGrid;
  cv: number;
  refit: boolean;

  bestParams_: Record<string, unknown> | null = null;
  bestScore_: number = -Number.POSITIVE_INFINITY;
  bestEstimator_: CVEstimator | null = null;
  cvResults_: Array<{ params: Record<string, unknown>; meanTestScore: number; stdTestScore: number }> = [];

  constructor(
    estimator: CVEstimator,
    paramGrid: ParamGrid,
    options: { cv?: number; refit?: boolean } = {},
  ) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.cv = options.cv ?? 5;
    this.refit = options.refit ?? true;
  }

  private _paramCombinations(): Array<Record<string, unknown>> {
    const keys = Object.keys(this.paramGrid);
    if (keys.length === 0) return [{}];

    let combinations: Array<Record<string, unknown>> = [{}];
    for (const key of keys) {
      const values = this.paramGrid[key] ?? [];
      const newCombinations: Array<Record<string, unknown>> = [];
      for (const combo of combinations) {
        for (const val of values) {
          newCombinations.push({ ...combo, [key]: val });
        }
      }
      combinations = newCombinations;
    }
    return combinations;
  }

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    const combinations = this._paramCombinations();
    this.cvResults_ = [];

    for (const params of combinations) {
      if (this.estimator.setParams) this.estimator.setParams(params);
      const scores = crossValScore(this.estimator, X, y, this.cv);
      const meanScore = Array.from(scores).reduce((a, b) => a + b, 0) / scores.length;
      const stdScore = Math.sqrt(
        Array.from(scores).reduce((a, b) => a + (b - meanScore) ** 2, 0) / scores.length,
      );
      this.cvResults_.push({ params, meanTestScore: meanScore, stdTestScore: stdScore });

      if (meanScore > this.bestScore_) {
        this.bestScore_ = meanScore;
        this.bestParams_ = params;
      }
    }

    if (this.refit && this.bestParams_) {
      if (this.estimator.setParams) this.estimator.setParams(this.bestParams_);
      this.estimator.fit(X, y);
      this.bestEstimator_ = this.estimator;
    }

    return this;
  }

  predict(X: Float64Array[]): Int32Array | Float64Array {
    if (!this.bestEstimator_) throw new NotFittedError("GridSearchCVExt is not fitted");
    return this.bestEstimator_.predict(X);
  }

  score(X: Float64Array[], y: Int32Array | Float64Array): number {
    if (!this.bestEstimator_) throw new NotFittedError("GridSearchCVExt is not fitted");
    return this.bestEstimator_.score(X, y);
  }
}
