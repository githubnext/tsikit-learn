/**
 * Additional cross-validation splitters.
 * Mirrors sklearn.model_selection: TimeSeriesSplit, StratifiedShuffleSplit,
 * LeavePOut, RepeatedStratifiedKFold, PredefinedSplit.
 */

export interface TimeSeriesFold {
  trainIndex: Int32Array;
  testIndex: Int32Array;
}

export interface TimeSeriesSplitParams {
  nSplits?: number;
  maxTrainSize?: number | null;
  testSize?: number | null;
  gap?: number;
}

/**
 * Time Series cross-validator.
 * Provides train/test indices for time-series data where the test set
 * always comes after the train set.
 *
 * Mirrors sklearn.model_selection.TimeSeriesSplit.
 */
export class TimeSeriesSplit {
  readonly nSplits: number;
  readonly maxTrainSize: number | null;
  readonly testSize: number | null;
  readonly gap: number;

  constructor(params: TimeSeriesSplitParams = {}) {
    this.nSplits = params.nSplits ?? 5;
    this.maxTrainSize = params.maxTrainSize ?? null;
    this.testSize = params.testSize ?? null;
    this.gap = params.gap ?? 0;
  }

  *split(
    X: Float64Array[] | Int32Array | { length: number }
  ): Generator<TimeSeriesFold> {
    const n = (X as { length: number }).length;
    const nSplits = this.nSplits;
    const testSize = this.testSize ?? Math.floor((n - this.gap) / (nSplits + 1));
    const gap = this.gap;

    let testEnd = n;
    for (let i = nSplits; i > 0; i--) {
      const testStart = testEnd - testSize;
      const trainEnd = testStart - gap;
      const trainStart = this.maxTrainSize !== null
        ? Math.max(0, trainEnd - this.maxTrainSize)
        : 0;
      if (trainEnd <= 0) continue;
      const trainIdx = new Int32Array(trainEnd - trainStart);
      for (let j = 0; j < trainIdx.length; j++) trainIdx[j] = trainStart + j;
      const testIdx = new Int32Array(testSize);
      for (let j = 0; j < testSize; j++) testIdx[j] = testStart + j;
      yield { trainIndex: trainIdx, testIndex: testIdx };
      testEnd -= testSize;
    }
  }

  getN(_X: unknown): number {
    return this.nSplits;
  }
}

export interface StratifiedShuffleSplitParams {
  nSplits?: number;
  testSize?: number | null;
  trainSize?: number | null;
  randomState?: number | null;
}

/**
 * Stratified ShuffleSplit cross-validator.
 *
 * Mirrors sklearn.model_selection.StratifiedShuffleSplit.
 */
export class StratifiedShuffleSplit {
  readonly nSplits: number;
  readonly testSize: number | null;
  readonly trainSize: number | null;
  readonly randomState: number | null;

  constructor(params: StratifiedShuffleSplitParams = {}) {
    this.nSplits = params.nSplits ?? 10;
    this.testSize = params.testSize ?? null;
    this.trainSize = params.trainSize ?? null;
    this.randomState = params.randomState ?? null;
  }

  *split(
    X: Float64Array[] | { length: number },
    y: Int32Array | number[]
  ): Generator<TimeSeriesFold> {
    const n = (X as { length: number }).length;
    const yArr = y instanceof Int32Array ? y : new Int32Array(y);
    const testSizeN = this.testSize !== null
      ? (this.testSize < 1 ? Math.round(this.testSize * n) : Math.round(this.testSize))
      : Math.round(0.1 * n);
    const trainSizeN = this.trainSize !== null
      ? (this.trainSize < 1 ? Math.round(this.trainSize * n) : Math.round(this.trainSize))
      : n - testSizeN;

    // Group indices by class
    const classMap: Map<number, number[]> = new Map();
    for (let i = 0; i < n; i++) {
      const cls = yArr[i] ?? 0;
      if (!classMap.has(cls)) classMap.set(cls, []);
      classMap.get(cls)!.push(i);
    }

    let seed = this.randomState ?? 0;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 4294967296;
    };

    for (let split = 0; split < this.nSplits; split++) {
      const testIdx: number[] = [];
      const trainIdx: number[] = [];

      for (const [, indices] of classMap) {
        // Fisher-Yates shuffle
        const idx = [...indices];
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          const tmp = idx[i];
          idx[i] = idx[j]!;
          idx[j] = tmp!;
        }
        const classTestN = Math.max(1, Math.round((indices.length / n) * testSizeN));
        const classTrainN = Math.max(1, Math.round((indices.length / n) * trainSizeN));
        testIdx.push(...idx.slice(0, classTestN));
        trainIdx.push(...idx.slice(classTestN, classTestN + classTrainN));
      }

      yield {
        trainIndex: new Int32Array(trainIdx),
        testIndex: new Int32Array(testIdx),
      };
    }
  }

  getN(_X: unknown): number {
    return this.nSplits;
  }
}

export interface LeavePOutParams {
  p: number;
}

/**
 * Leave P Out cross-validator.
 * Generates all possible combinations of P samples as test set.
 * Mirrors sklearn.model_selection.LeavePOut.
 */
export class LeavePOut {
  readonly p: number;

  constructor(params: LeavePOutParams) {
    this.p = params.p;
  }

  *split(
    X: Float64Array[] | { length: number }
  ): Generator<TimeSeriesFold> {
    const n = (X as { length: number }).length;
    const p = this.p;
    // Generate combinations of p indices as test set
    const combo = new Int32Array(p);
    for (let i = 0; i < p; i++) combo[i] = i;

    while (true) {
      const testSet = new Set(Array.from(combo));
      const testIdx = new Int32Array(combo);
      const trainIdx = new Int32Array(n - p);
      let ti = 0;
      for (let i = 0; i < n; i++) {
        if (!testSet.has(i)) trainIdx[ti++] = i;
      }
      yield { trainIndex: trainIdx, testIndex: testIdx };

      // Advance combination
      let i = p - 1;
      while (i >= 0 && combo[i]! === n - p + i) i--;
      if (i < 0) break;
      combo[i]!++;
      for (let j = i + 1; j < p; j++) combo[j] = combo[j - 1]! + 1;
    }
  }

  getNCombinations(n: number): number {
    const p = this.p;
    // C(n, p)
    let result = 1;
    for (let i = 0; i < p; i++) {
      result = (result * (n - i)) / (i + 1);
    }
    return Math.round(result);
  }
}

export interface RepeatedStratifiedKFoldParams {
  nSplits?: number;
  nRepeats?: number;
  randomState?: number | null;
}

/**
 * Repeated Stratified K-Fold cross-validator.
 * Mirrors sklearn.model_selection.RepeatedStratifiedKFold.
 */
export class RepeatedStratifiedKFold {
  readonly nSplits: number;
  readonly nRepeats: number;
  readonly randomState: number | null;

  constructor(params: RepeatedStratifiedKFoldParams = {}) {
    this.nSplits = params.nSplits ?? 5;
    this.nRepeats = params.nRepeats ?? 10;
    this.randomState = params.randomState ?? null;
  }

  *split(
    X: Float64Array[] | { length: number },
    y: Int32Array | number[]
  ): Generator<TimeSeriesFold> {
    const n = (X as { length: number }).length;
    const yArr = y instanceof Int32Array ? y : new Int32Array(y);
    const k = this.nSplits;

    // Group by class
    const classMap: Map<number, number[]> = new Map();
    for (let i = 0; i < n; i++) {
      const cls = yArr[i] ?? 0;
      if (!classMap.has(cls)) classMap.set(cls, []);
      classMap.get(cls)!.push(i);
    }

    let seed = this.randomState ?? 0;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 4294967296;
    };

    for (let rep = 0; rep < this.nRepeats; rep++) {
      // Shuffle each class
      const shuffled: Map<number, number[]> = new Map();
      for (const [cls, indices] of classMap) {
        const idx = [...indices];
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          const tmp = idx[i];
          idx[i] = idx[j]!;
          idx[j] = tmp!;
        }
        shuffled.set(cls, idx);
      }

      // Assign to folds
      const folds: number[][] = Array.from({ length: k }, () => []);
      for (const [, indices] of shuffled) {
        for (let i = 0; i < indices.length; i++) {
          folds[i % k]!.push(indices[i]!);
        }
      }

      for (let fold = 0; fold < k; fold++) {
        const testIdx = new Int32Array(folds[fold] ?? []);
        const trainIdxArr: number[] = [];
        for (let f = 0; f < k; f++) {
          if (f !== fold) trainIdxArr.push(...(folds[f] ?? []));
        }
        yield { trainIndex: new Int32Array(trainIdxArr), testIndex: testIdx };
      }
    }
  }

  getN(_X: unknown): number {
    return this.nSplits * this.nRepeats;
  }
}

export interface PredefinedSplitParams {
  testFold: Int32Array | number[];
}

/**
 * Predefined split cross-validator.
 * Uses a pre-defined scheme for splits where a sample's fold is determined by the
 * `testFold` array. Samples with -1 are always put in the train set.
 *
 * Mirrors sklearn.model_selection.PredefinedSplit.
 */
export class PredefinedSplit {
  readonly testFold: Int32Array;

  constructor(params: PredefinedSplitParams) {
    this.testFold = params.testFold instanceof Int32Array
      ? params.testFold
      : new Int32Array(params.testFold);
  }

  *split(): Generator<TimeSeriesFold> {
    const folds = [...new Set(Array.from(this.testFold))].filter((f) => f >= 0).sort((a, b) => a - b);
    for (const fold of folds) {
      const testIdx = new Int32Array(
        Array.from(this.testFold)
          .map((f, i) => (f === fold ? i : -1))
          .filter((i) => i >= 0)
      );
      const trainIdx = new Int32Array(
        Array.from(this.testFold)
          .map((f, i) => (f !== fold ? i : -1))
          .filter((i) => i >= 0)
      );
      yield { trainIndex: trainIdx, testIndex: testIdx };
    }
  }

  getN(): number {
    return new Set(Array.from(this.testFold).filter((f) => f >= 0)).size;
  }
}
