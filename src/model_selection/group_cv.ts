/**
 * Group cross-validation iterators.
 * Port of sklearn.model_selection._split (GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, LeaveOneGroupOut, etc.)
 */

export interface GroupSplitResult {
  train: Int32Array;
  test: Int32Array;
}

/**
 * K-fold iterator with non-overlapping groups.
 * Port of sklearn.model_selection.GroupKFold
 */
export class GroupKFold {
  nSplits: number;

  constructor(nSplits = 5) {
    this.nSplits = nSplits;
  }

  split(
    X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): GroupSplitResult[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups))).sort(
      (a, b) => a - b,
    );
    const nGroups = uniqueGroups.length;
    if (nGroups < this.nSplits)
      throw new Error(
        `Cannot have n_splits=${this.nSplits} > n_groups=${nGroups}`,
      );

    // Distribute groups into folds greedily (balanced by group size)
    const groupSizes = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const g = groups[i]!;
      groupSizes.set(g, (groupSizes.get(g) ?? 0) + 1);
    }

    const folds: number[][] = Array.from({ length: this.nSplits }, () => []);
    const foldSizes = new Int32Array(this.nSplits);
    const sortedGroups = [...uniqueGroups].sort(
      (a, b) => (groupSizes.get(b) ?? 0) - (groupSizes.get(a) ?? 0),
    );
    for (const g of sortedGroups) {
      const minFold = foldSizes.indexOf(Math.min(...foldSizes));
      folds[minFold]!.push(g);
      foldSizes[minFold] += groupSizes.get(g) ?? 0;
    }

    return folds.map((testGroups) => {
      const testSet = new Set(testGroups);
      const train: number[] = [];
      const test: number[] = [];
      for (let i = 0; i < n; i++) {
        if (testSet.has(groups[i]!)) test.push(i);
        else train.push(i);
      }
      return { train: new Int32Array(train), test: new Int32Array(test) };
    });
  }

  getNumSplits(
    _X?: Float64Array[],
    _y?: Int32Array,
    _groups?: Int32Array,
  ): number {
    return this.nSplits;
  }
}

/**
 * Stratified GroupKFold.
 * Port of sklearn.model_selection.StratifiedGroupKFold
 */
export class StratifiedGroupKFold {
  nSplits: number;
  shuffle: boolean;
  randomState: number | null;

  constructor(nSplits = 5, shuffle = false, randomState: number | null = null) {
    this.nSplits = nSplits;
    this.shuffle = shuffle;
    this.randomState = randomState;
  }

  split(
    X: Float64Array[],
    y: Int32Array,
    groups: Int32Array,
  ): GroupSplitResult[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups))).sort(
      (a, b) => a - b,
    );
    const nGroups = uniqueGroups.length;

    // Per-group class distribution
    const groupClassDist = new Map<number, Map<number, number>>();
    for (let i = 0; i < n; i++) {
      const g = groups[i]!;
      const c = y[i]!;
      if (!groupClassDist.has(g)) groupClassDist.set(g, new Map());
      const dist = groupClassDist.get(g)!;
      dist.set(c, (dist.get(c) ?? 0) + 1);
    }

    const folds: number[][] = Array.from({ length: this.nSplits }, () => []);
    const foldClassDist: Map<number, number>[] = Array.from(
      { length: this.nSplits },
      () => new Map(),
    );

    for (const g of uniqueGroups) {
      const classDist = groupClassDist.get(g)!;
      // Find fold with best class balance
      let bestFold = 0;
      let bestScore = Number.POSITIVE_INFINITY;
      for (let f = 0; f < this.nSplits; f++) {
        let score = 0;
        for (const [c, cnt] of classDist) {
          const foldCount = foldClassDist[f]!.get(c) ?? 0;
          score += (foldCount + cnt) ** 2;
        }
        if (score < bestScore) {
          bestScore = score;
          bestFold = f;
        }
      }
      folds[bestFold]!.push(g);
      for (const [c, cnt] of classDist) {
        foldClassDist[bestFold]!.set(
          c,
          (foldClassDist[bestFold]!.get(c) ?? 0) + cnt,
        );
      }
    }

    return folds.map((testGroups) => {
      const testSet = new Set(testGroups);
      const train: number[] = [];
      const test: number[] = [];
      for (let i = 0; i < n; i++) {
        if (testSet.has(groups[i]!)) test.push(i);
        else train.push(i);
      }
      return { train: new Int32Array(train), test: new Int32Array(test) };
    });
  }

  getNumSplits(): number {
    return this.nSplits;
  }
}

/**
 * Leave One Group Out cross-validation.
 * Port of sklearn.model_selection.LeaveOneGroupOut
 */
export class LeaveOneGroupOut {
  split(
    X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): GroupSplitResult[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups))).sort(
      (a, b) => a - b,
    );
    return uniqueGroups.map((g) => {
      const train: number[] = [];
      const test: number[] = [];
      for (let i = 0; i < n; i++) {
        if (groups[i] === g) test.push(i);
        else train.push(i);
      }
      return { train: new Int32Array(train), test: new Int32Array(test) };
    });
  }

  getNumSplits(
    _X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): number {
    return new Set(Array.from(groups)).size;
  }
}

/**
 * Leave P Groups Out.
 * Port of sklearn.model_selection.LeavePGroupsOut
 */
export class LeavePGroupsOut {
  nGroups: number;

  constructor(nGroups: number) {
    this.nGroups = nGroups;
  }

  split(
    X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): GroupSplitResult[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups))).sort(
      (a, b) => a - b,
    );
    const results: GroupSplitResult[] = [];

    // Generate all combinations of nGroups groups to leave out
    const combinations = this._combinations(uniqueGroups, this.nGroups);
    for (const combo of combinations) {
      const testSet = new Set(combo);
      const train: number[] = [];
      const test: number[] = [];
      for (let i = 0; i < n; i++) {
        if (testSet.has(groups[i]!)) test.push(i);
        else train.push(i);
      }
      results.push({
        train: new Int32Array(train),
        test: new Int32Array(test),
      });
    }
    return results;
  }

  private _combinations<T>(arr: T[], k: number): T[][] {
    if (k === 0) return [[]];
    if (arr.length < k) return [];
    const [first, ...rest] = arr;
    return [
      ...this._combinations(rest, k - 1).map((c) => [first!, ...c]),
      ...this._combinations(rest, k),
    ];
  }

  getNumSplits(
    _X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): number {
    const n = new Set(Array.from(groups)).size;
    const k = this.nGroups;
    let result = 1;
    for (let i = 0; i < k; i++) result *= (n - i) / (i + 1);
    return Math.round(result);
  }
}

/**
 * Group Shuffle Split.
 * Port of sklearn.model_selection.GroupShuffleSplit
 */
export class GroupShuffleSplit {
  nSplits: number;
  testSize: number | null;
  trainSize: number | null;
  randomState: number | null;

  constructor(
    nSplits = 5,
    testSize: number | null = null,
    trainSize: number | null = null,
    randomState: number | null = null,
  ) {
    this.nSplits = nSplits;
    this.testSize = testSize;
    this.trainSize = trainSize;
    this.randomState = randomState;
  }

  split(
    X: Float64Array[],
    _y: Int32Array | null,
    groups: Int32Array,
  ): GroupSplitResult[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups))).sort(
      (a, b) => a - b,
    );
    const nGroups = uniqueGroups.length;
    const testFrac = this.testSize ?? 0.1;
    const nTest = Math.max(1, Math.round(nGroups * testFrac));

    let seed = this.randomState ?? 42;
    const rand = (): number => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0x100000000;
    };

    return Array.from({ length: this.nSplits }, () => {
      // Shuffle groups
      const shuffled = [...uniqueGroups];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = shuffled[i]!;
        shuffled[i] = shuffled[j]!;
        shuffled[j] = tmp;
      }
      const testGroups = new Set(shuffled.slice(0, nTest));
      const train: number[] = [];
      const test: number[] = [];
      for (let i = 0; i < n; i++) {
        if (testGroups.has(groups[i]!)) test.push(i);
        else train.push(i);
      }
      return { train: new Int32Array(train), test: new Int32Array(test) };
    });
  }
}
