/**
 * Grid search and cross-validation utilities.
 * Mirrors sklearn.model_selection.GridSearchCV and cross_val_score.
 */

import { KFold } from "./split.js";

export interface Estimator {
  fit(X: Float64Array[], y: Float64Array): this;
  score(X: Float64Array[], y: Float64Array): number;
}

export interface GridParams {
  [key: string]: number | string | boolean;
}

function cartesianProduct(paramGrid: Record<string, (number | string | boolean)[]>): GridParams[] {
  const keys = Object.keys(paramGrid);
  if (keys.length === 0) return [{}];
  const result: GridParams[] = [{}];
  for (const key of keys) {
    const values = paramGrid[key] ?? [];
    const newResult: GridParams[] = [];
    for (const existing of result) {
      for (const val of values) {
        newResult.push({ ...existing, [key]: val });
      }
    }
    result.length = 0;
    result.push(...newResult);
  }
  return result;
}

export class GridSearchCV {
  estimator: Estimator;
  paramGrid: Record<string, (number | string | boolean)[]>;
  cv: number;
  scoring: string;

  bestParams_: GridParams | null = null;
  bestScore_: number = Number.NEGATIVE_INFINITY;
  bestEstimator_: Estimator | null = null;
  cvResults_: { params: GridParams; meanTestScore: number }[] = [];

  constructor(
    estimator: Estimator,
    paramGrid: Record<string, (number | string | boolean)[]>,
    options: { cv?: number; scoring?: string } = {},
  ) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.cv = options.cv ?? 5;
    this.scoring = options.scoring ?? "score";
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const candidates = cartesianProduct(this.paramGrid);
    const kfold = new KFold({ nSplits: this.cv });

    this.cvResults_ = [];
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestParams: GridParams = {};

    for (const params of candidates) {
      const scores: number[] = [];
      for (const fold of kfold.split(X)) {
        const trainIdx = fold.trainIndex;
        const testIdx = fold.testIndex;
        const XTrain = Array.from(trainIdx).map((i) => X[i] ?? new Float64Array(0));
        const yTrain = new Float64Array(Array.from(trainIdx).map((i) => y[i] ?? 0));
        const XTest = Array.from(testIdx).map((i) => X[i] ?? new Float64Array(0));
        const yTest = new Float64Array(Array.from(testIdx).map((i) => y[i] ?? 0));

        // Clone and set params
        const est = Object.create(
          Object.getPrototypeOf(this.estimator) as object,
        ) as Estimator & Record<string, unknown>;
        Object.assign(est, this.estimator);
        for (const [k, v] of Object.entries(params)) {
          est[k] = v;
        }
        // Reset fitted attributes
        est.fit(XTrain, yTrain);
        scores.push(est.score(XTest, yTest));
      }
      const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      this.cvResults_.push({ params, meanTestScore: meanScore });

      if (meanScore > bestScore) {
        bestScore = meanScore;
        bestParams = params;
      }
    }

    this.bestParams_ = bestParams;
    this.bestScore_ = bestScore;

    // Refit best estimator on full data
    const best = Object.create(
      Object.getPrototypeOf(this.estimator) as object,
    ) as Estimator & Record<string, unknown>;
    Object.assign(best, this.estimator);
    for (const [k, v] of Object.entries(bestParams)) {
      best[k] = v;
    }
    best.fit(X, y);
    this.bestEstimator_ = best as Estimator;

    return this;
  }

  score(X: Float64Array[], y: Float64Array): number {
    if (this.bestEstimator_ === null) throw new Error("GridSearchCV not fitted");
    return this.bestEstimator_.score(X, y);
  }
}

export function crossValScore(
  estimator: Estimator,
  X: Float64Array[],
  y: Float64Array,
  cv = 5,
): Float64Array {
  const kfold = new KFold({ nSplits: cv });
  const scores: number[] = [];

  for (const fold of kfold.split(X)) {
    const trainIdx = fold.trainIndex;
    const testIdx = fold.testIndex;
    const XTrain = Array.from(trainIdx).map((i) => X[i] ?? new Float64Array(0));
    const yTrain = new Float64Array(Array.from(trainIdx).map((i) => y[i] ?? 0));
    const XTest = Array.from(testIdx).map((i) => X[i] ?? new Float64Array(0));
    const yTest = new Float64Array(Array.from(testIdx).map((i) => y[i] ?? 0));

    const est = Object.create(
      Object.getPrototypeOf(estimator) as object,
    ) as Estimator;
    Object.assign(est, estimator);
    est.fit(XTrain, yTrain);
    scores.push(est.score(XTest, yTest));
  }

  return new Float64Array(scores);
}
