/**
 * Model selection utilities: train/test split and cross-validation.
 * Mirrors sklearn.model_selection.
 */

import { ValueError } from "../exceptions.js";

export interface TrainTestSplitOptions {
  testSize?: number;
  trainSize?: number;
  randomState?: number;
  shuffle?: boolean;
  stratify?: Float64Array | Int32Array;
}

export interface TrainTestSplitResult {
  XTrain: Float64Array[];
  XTest: Float64Array[];
  yTrain: Float64Array | Int32Array;
  yTest: Float64Array | Int32Array;
}

/** Simple linear congruential generator for reproducible shuffles. */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0x100000000;
  };
}

/** Fisher-Yates shuffle with optional seed. */
function shuffleIndices(n: number, rng: () => number): Int32Array {
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i] ?? 0;
    idx[i] = idx[j] ?? 0;
    idx[j] = tmp;
  }
  return idx;
}

/**
 * Split arrays or matrices into random train and test subsets.
 * Mirrors sklearn.model_selection.train_test_split.
 */
export function train_test_split(
  X: Float64Array[],
  y: Float64Array | Int32Array,
  options: TrainTestSplitOptions = {},
): TrainTestSplitResult {
  const { testSize = 0.25, randomState = 42, shuffle = true } = options;
  const n = X.length;
  const nTest = Math.max(1, Math.round(n * testSize));
  const nTrain = n - nTest;

  if (nTrain <= 0) {
    throw new ValueError(
      `With n_samples=${n} and test_size=${testSize}, the resulting train set would be empty.`,
    );
  }

  const rng = lcg(randomState);
  const indices = shuffle
    ? shuffleIndices(n, rng)
    : (() => {
        const idx = new Int32Array(n);
        for (let i = 0; i < n; i++) idx[i] = i;
        return idx;
      })();

  const trainIdx = indices.slice(0, nTrain);
  const testIdx = indices.slice(nTrain);

  const XTrain = Array.from(trainIdx, (i) => X[i] ?? new Float64Array(0));
  const XTest = Array.from(testIdx, (i) => X[i] ?? new Float64Array(0));

  const isInt = y instanceof Int32Array;
  const yTrain = isInt
    ? new Int32Array(Array.from(trainIdx, (i) => (y as Int32Array)[i] ?? 0))
    : new Float64Array(
        Array.from(trainIdx, (i) => (y as Float64Array)[i] ?? 0),
      );
  const yTest = isInt
    ? new Int32Array(Array.from(testIdx, (i) => (y as Int32Array)[i] ?? 0))
    : new Float64Array(Array.from(testIdx, (i) => (y as Float64Array)[i] ?? 0));

  return { XTrain, XTest, yTrain, yTest };
}

export interface KFoldOptions {
  nSplits?: number;
  shuffle?: boolean;
  randomState?: number;
}

export interface Fold {
  trainIndex: Int32Array;
  testIndex: Int32Array;
}

/**
 * K-Folds cross-validator.
 * Mirrors sklearn.model_selection.KFold.
 */
export class KFold {
  nSplits: number;
  shuffle: boolean;
  randomState: number;

  constructor(options: KFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState ?? 0;
  }

  /** Generate indices to split data into training and test sets. */
  *split(X: Float64Array[]): Generator<Fold> {
    const n = X.length;
    if (this.nSplits > n) {
      throw new ValueError(
        `Cannot have number of splits n_splits=${this.nSplits} greater than the number of samples=${n}`,
      );
    }

    const rng = lcg(this.randomState);
    const indices = this.shuffle
      ? shuffleIndices(n, rng)
      : (() => {
          const idx = new Int32Array(n);
          for (let i = 0; i < n; i++) idx[i] = i;
          return idx;
        })();

    const foldSizes = new Int32Array(this.nSplits).fill(
      Math.floor(n / this.nSplits),
    );
    for (let i = 0; i < n % this.nSplits; i++) {
      foldSizes[i] = (foldSizes[i] ?? 0) + 1;
    }

    let current = 0;
    for (let fold = 0; fold < this.nSplits; fold++) {
      const start = current;
      const stop = current + (foldSizes[fold] ?? 0);
      const testIndex = indices.slice(start, stop);
      const trainIndex = new Int32Array([
        ...Array.from(indices.slice(0, start)),
        ...Array.from(indices.slice(stop)),
      ]);
      yield { trainIndex, testIndex };
      current = stop;
    }
  }

  getNumSplits(): number {
    return this.nSplits;
  }
}

export interface StratifiedKFoldOptions {
  nSplits?: number;
  shuffle?: boolean;
  randomState?: number;
}

/**
 * Stratified K-Folds cross-validator.
 * Mirrors sklearn.model_selection.StratifiedKFold.
 */
export class StratifiedKFold {
  nSplits: number;
  shuffle: boolean;
  randomState: number;

  constructor(options: StratifiedKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState ?? 0;
  }

  *split(X: Float64Array[], y: Float64Array | Int32Array): Generator<Fold> {
    const n = X.length;
    const rng = lcg(this.randomState);

    // Group indices by class
    const classIndices = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const c = y[i] ?? 0;
      if (!classIndices.has(c)) classIndices.set(c, []);
      (classIndices.get(c) as number[]).push(i);
    }

    // Assign indices to folds
    const foldIndices: number[][] = Array.from(
      { length: this.nSplits },
      () => [],
    );
    for (const [, idxList] of classIndices) {
      const shuffled = this.shuffle
        ? [...idxList].sort(() => rng() - 0.5)
        : idxList;
      shuffled.forEach((idx, i) => {
        (foldIndices[i % this.nSplits] as number[]).push(idx);
      });
    }

    for (let fold = 0; fold < this.nSplits; fold++) {
      const testIndex = new Int32Array(foldIndices[fold] as number[]);
      const trainIndicesList: number[] = [];
      for (let f = 0; f < this.nSplits; f++) {
        if (f !== fold) trainIndicesList.push(...(foldIndices[f] as number[]));
      }
      yield { trainIndex: new Int32Array(trainIndicesList), testIndex };
    }
  }
}
