/**
 * Probability calibration.
 * Mirrors sklearn.calibration.CalibratedClassifierCV.
 * Uses Platt scaling (logistic) or isotonic regression for calibration.
 */

import { NotFittedError } from "../exceptions.js";

interface Classifier {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
  score?(X: Float64Array[], y: Float64Array): number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Platt scaling: fit a logistic function on scores to map to probabilities. */
function plattScale(scores: Float64Array, y: Float64Array): [number, number] {
  const n = scores.length;
  let A = 0;
  let B = 0;
  const lr = 0.01;

  for (let iter = 0; iter < 1000; iter++) {
    let gradA = 0;
    let gradB = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(A * (scores[i] ?? 0) + B);
      const err = p - (y[i] ?? 0);
      gradA += err * (scores[i] ?? 0);
      gradB += err;
    }
    A -= lr * gradA / n;
    B -= lr * gradB / n;
  }

  return [A, B];
}

export class CalibratedClassifierCV {
  baseEstimator: Classifier;
  method: string;
  cv: number;

  calibratedEstimators_: {
    estimator: Classifier;
    A: number;
    B: number;
  }[] | null = null;
  classes_: Float64Array | null = null;

  constructor(
    baseEstimator: Classifier,
    options: { method?: string; cv?: number } = {},
  ) {
    this.baseEstimator = baseEstimator;
    this.method = options.method ?? "sigmoid";
    this.cv = options.cv ?? 5;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const posClass = uniqueClasses[uniqueClasses.length - 1] ?? 1;

    const yBin = new Float64Array(y.map((yi) => (yi === posClass ? 1 : 0)));

    // Simple hold-out calibration
    const foldSize = Math.floor(n / this.cv);
    this.calibratedEstimators_ = [];

    for (let fold = 0; fold < this.cv; fold++) {
      const testStart = fold * foldSize;
      const testEnd = fold === this.cv - 1 ? n : testStart + foldSize;

      const trainIdx: number[] = [];
      const testIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i >= testStart && i < testEnd) testIdx.push(i);
        else trainIdx.push(i);
      }

      const XTrain = trainIdx.map((i) => X[i] ?? new Float64Array(0));
      const yTrain = new Float64Array(trainIdx.map((i) => y[i] ?? 0));
      const XTest = testIdx.map((i) => X[i] ?? new Float64Array(0));
      const yTest = new Float64Array(testIdx.map((i) => yBin[i] ?? 0));

      const est = Object.create(Object.getPrototypeOf(this.baseEstimator) as object) as Classifier;
      Object.assign(est, this.baseEstimator);
      est.fit(XTrain, yTrain);

      const testPred = est.predict(XTest);
      const [A, B] = plattScale(testPred, yTest);

      this.calibratedEstimators_.push({ estimator: est, A, B });
    }

    return this;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (this.calibratedEstimators_ === null) throw new NotFittedError("CalibratedClassifierCV");

    const n = X.length;
    const probs = new Float64Array(n);

    for (const { estimator, A, B } of this.calibratedEstimators_) {
      const scores = estimator.predict(X);
      for (let i = 0; i < n; i++) {
        probs[i] = (probs[i] ?? 0) + sigmoid(A * (scores[i] ?? 0) + B);
      }
    }

    const k = this.calibratedEstimators_.length;
    return Array.from({ length: n }, (_, i) => {
      const p = (probs[i] ?? 0) / k;
      return new Float64Array([1 - p, p]);
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("CalibratedClassifierCV");
    const classes = this.classes_;
    const proba = this.predictProba(X);
    const posClass = classes[classes.length - 1] ?? 1;
    const negClass = classes[0] ?? 0;
    return new Float64Array(proba.map((p) => ((p[1] ?? 0) >= 0.5 ? posClass : negClass)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}
