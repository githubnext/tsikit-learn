/**
 * Model selection extensions: Successive halving, Bayesian optimization, stratified shuffle split.
 * Mirrors sklearn.model_selection additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Stratified shuffle split: split data maintaining class distribution. */
export class StratifiedShuffleSplit {
  n_splits: number;
  test_size: number;
  random_state: number | null;

  constructor(params: { n_splits?: number; test_size?: number; random_state?: number | null } = {}) {
    this.n_splits = params.n_splits ?? 10;
    this.test_size = params.test_size ?? 0.1;
    this.random_state = params.random_state ?? null;
  }

  *split(X: Float64Array[], y: Int32Array): Generator<[Int32Array, Int32Array]> {
    const n = X.length;
    const classes = [...new Set(Array.from(y))];
    for (let s = 0; s < this.n_splits; s++) {
      const trainIdx: number[] = [];
      const testIdx: number[] = [];
      for (const cls of classes) {
        const clsIdx = Array.from({ length: n }, (_, i) => i).filter(i => y[i] === cls);
        const nTest = Math.round(clsIdx.length * this.test_size);
        const shuffled = [...clsIdx].sort(() => Math.random() - 0.5);
        testIdx.push(...shuffled.slice(0, nTest));
        trainIdx.push(...shuffled.slice(nTest));
      }
      yield [new Int32Array(trainIdx), new Int32Array(testIdx)];
    }
  }
}

/** Time series cross-validator (expanding window). */
export class TimeSeriesSplit {
  n_splits: number;
  gap: number;
  max_train_size: number | null;
  test_size: number | null;

  constructor(params: { n_splits?: number; gap?: number; max_train_size?: number | null; test_size?: number | null } = {}) {
    this.n_splits = params.n_splits ?? 5;
    this.gap = params.gap ?? 0;
    this.max_train_size = params.max_train_size ?? null;
    this.test_size = params.test_size ?? null;
  }

  *split(X: Float64Array[]): Generator<[Int32Array, Int32Array]> {
    const n = X.length;
    const testSize = this.test_size ?? Math.floor(n / (this.n_splits + 1));
    for (let s = 0; s < this.n_splits; s++) {
      const testEnd = n - (this.n_splits - s - 1) * testSize;
      const testStart = testEnd - testSize;
      const trainEnd = testStart - this.gap;
      const trainStart = this.max_train_size != null ? Math.max(0, trainEnd - this.max_train_size) : 0;
      if (trainEnd <= 0) continue;
      const trainIdx = new Int32Array(trainEnd - trainStart).map((_, i) => trainStart + i);
      const testIdx = new Int32Array(testEnd - testStart).map((_, i) => testStart + i);
      yield [trainIdx, testIdx];
    }
  }
}

/** Leave-P-Out cross validator. */
export class LeavePOut {
  p: number;
  constructor(p: number) {
    this.p = p;
  }

  get_n_splits(X: Float64Array[]): number {
    const n = X.length;
    return Math.floor(factorial(n) / (factorial(this.p) * factorial(n - this.p)));
  }

  *split(X: Float64Array[]): Generator<[Int32Array, Int32Array]> {
    const n = X.length;
    yield* combinations(
      Array.from({ length: n }, (_, i) => i),
      this.p,
      (testIdx) => {
        const testSet = new Set(testIdx);
        const trainIdx = new Int32Array(Array.from({ length: n }, (_, i) => i).filter(i => !testSet.has(i)));
        return [trainIdx, new Int32Array(testIdx)];
      },
    );
  }
}

function factorial(n: number): number {
  if (n <= 1) return 1;
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

function* combinations<T>(arr: T[], k: number, mapper: (combo: T[]) => [Int32Array, Int32Array]): Generator<[Int32Array, Int32Array]> {
  if (k === 0) { yield mapper([]); return; }
  for (let i = 0; i <= arr.length - k; i++) {
    const rest = arr.slice(i + 1);
    for (const sub of combinations(rest, k - 1, mapper)) {
      yield mapper([arr[i]!, ...Array.from(sub[1]).map(j => arr[j]!)]);
    }
  }
}

/** Compute permutation importance for an estimator. */
export function permutationImportance(
  estimator: BaseEstimator,
  X: Float64Array[],
  y: Int32Array,
  n_repeats = 5,
): Float64Array {
  const d = X[0]?.length ?? 0;
  const scoreFn = (est: BaseEstimator, Xp: Float64Array[], yp: Int32Array): number => {
    const preds = (est as unknown as { predict(X: Float64Array[]): Int32Array }).predict(Xp);
    let correct = 0;
    for (let i = 0; i < yp.length; i++) if ((preds[i] ?? -1) === (yp[i] ?? 0)) correct++;
    return correct / yp.length;
  };
  const baseScore = scoreFn(estimator, X, y);
  const importances = new Float64Array(d);
  for (let f = 0; f < d; f++) {
    let sum = 0;
    for (let r = 0; r < n_repeats; r++) {
      const Xp = X.map(row => new Float64Array(row));
      const vals = Xp.map(row => row[f] ?? 0);
      const shuffled = [...vals].sort(() => Math.random() - 0.5);
      for (let i = 0; i < Xp.length; i++) Xp[i]![f] = shuffled[i] ?? 0;
      sum += baseScore - scoreFn(estimator, Xp, y);
    }
    importances[f] = sum / n_repeats;
  }
  return importances;
}

/** Compute learning curve scores for varying training sizes. */
export function learningCurveScores(
  estimator: BaseEstimator,
  X: Float64Array[],
  y: Int32Array,
  trainSizes: Float64Array,
): { train_scores: Float64Array; test_scores: Float64Array } {
  const n = X.length;
  const trainS = new Float64Array(trainSizes.length);
  const testS = new Float64Array(trainSizes.length);
  const testSize = Math.floor(n * 0.2);
  const testX = X.slice(n - testSize);
  const testY = y.slice(n - testSize);
  const trainX = X.slice(0, n - testSize);
  const trainY = y.slice(0, n - testSize);

  const est = estimator as unknown as { fit(X: Float64Array[], y: Int32Array): void; predict(X: Float64Array[]): Int32Array };

  for (let i = 0; i < trainSizes.length; i++) {
    const size = Math.max(2, Math.floor((trainSizes[i] ?? 0.5) * trainX.length));
    const Xb = trainX.slice(0, size);
    const yb = trainY.slice(0, size);
    est.fit(Xb, yb);
    const trainPred = est.predict(Xb);
    const testPred = est.predict(testX);
    const scoreArr = (pred: Int32Array, yTrue: Int32Array) => {
      let c = 0;
      for (let j = 0; j < yTrue.length; j++) if ((pred[j] ?? -1) === (yTrue[j] ?? 0)) c++;
      return c / (yTrue.length || 1);
    };
    trainS[i] = scoreArr(trainPred, yb);
    testS[i] = scoreArr(testPred, testY);
  }
  return { train_scores: trainS, test_scores: testS };
}
