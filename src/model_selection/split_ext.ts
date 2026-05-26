/**
 * Extended cross-validation splitters: LeaveOneOut, LeavePOut, LeaveOneGroupOut, StratifiedGroupKFold, PredefinedSplit
 */

export interface SplitResult {
  trainIndices: Int32Array;
  testIndices: Int32Array;
}

export class LeaveOneOut {
  getNumSplits(nSamples: number): number {
    return nSamples;
  }

  *split(nSamples: number): Generator<SplitResult> {
    for (let i = 0; i < nSamples; i++) {
      const trainIndices = new Int32Array(nSamples - 1);
      let ti = 0;
      for (let j = 0; j < nSamples; j++) {
        if (j !== i) trainIndices[ti++] = j;
      }
      yield { trainIndices, testIndices: new Int32Array([i]) };
    }
  }
}

export class LeavePOut {
  private p: number;

  constructor(p: number) {
    this.p = p;
  }

  getNumSplits(nSamples: number): number {
    if (this.p > nSamples) return 0;
    let result = 1;
    for (let i = 0; i < this.p; i++) {
      result = (result * (nSamples - i)) / (i + 1);
    }
    return Math.round(result);
  }

  *split(nSamples: number): Generator<SplitResult> {
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    const combinations = this.getCombinations(indices, this.p);
    for (const testSet of combinations) {
      const testSet32 = new Int32Array(testSet);
      const trainSet = new Int32Array(nSamples - this.p);
      let ti = 0;
      for (let i = 0; i < nSamples; i++) {
        if (!testSet.includes(i)) trainSet[ti++] = i;
      }
      yield { trainIndices: trainSet, testIndices: testSet32 };
    }
  }

  private *getCombinations(arr: number[], k: number): Generator<number[]> {
    if (k === 0) { yield []; return; }
    if (arr.length < k) return;
    const [first, ...rest] = arr;
    for (const combo of this.getCombinations(rest, k - 1)) {
      yield [first!, ...combo];
    }
    yield* this.getCombinations(rest, k);
  }
}

export class LeaveOneGroupOut {
  getNumSplits(groups: Int32Array): number {
    return new Set(Array.from(groups)).size;
  }

  *split(nSamples: number, groups: Int32Array): Generator<SplitResult> {
    const uniqueGroups = [...new Set(Array.from(groups))];
    for (const g of uniqueGroups) {
      const testIdx: number[] = [];
      const trainIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if ((groups[i] ?? -1) === g) testIdx.push(i);
        else trainIdx.push(i);
      }
      yield {
        trainIndices: new Int32Array(trainIdx),
        testIndices: new Int32Array(testIdx),
      };
    }
  }
}

export class StratifiedGroupKFold {
  private nSplits: number;

  constructor(nSplits = 5) {
    this.nSplits = nSplits;
  }

  *split(
    nSamples: number,
    y: Int32Array,
    groups: Int32Array
  ): Generator<SplitResult> {
    const foldAssignments = new Int32Array(nSamples);
    const groupCounts = new Map<number, number>();
    for (let i = 0; i < nSamples; i++) {
      const g = groups[i] ?? 0;
      groupCounts.set(g, (groupCounts.get(g) ?? 0) + 1);
    }
    let foldIdx = 0;
    for (const [g] of groupCounts) {
      for (let i = 0; i < nSamples; i++) {
        if ((groups[i] ?? -1) === g) foldAssignments[i] = foldIdx % this.nSplits;
      }
      foldIdx++;
    }
    for (let fold = 0; fold < this.nSplits; fold++) {
      const trainIdx: number[] = [];
      const testIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if ((foldAssignments[i] ?? 0) === fold) testIdx.push(i);
        else trainIdx.push(i);
      }
      yield {
        trainIndices: new Int32Array(trainIdx),
        testIndices: new Int32Array(testIdx),
      };
    }
  }
}

export class PredefinedSplit {
  private testFold: Int32Array;

  constructor(testFold: Int32Array) {
    this.testFold = testFold;
  }

  getNumSplits(): number {
    return new Set(Array.from(this.testFold).filter((v) => v >= 0)).size;
  }

  *split(): Generator<SplitResult> {
    const uniqueFolds = [...new Set(Array.from(this.testFold).filter((v) => v >= 0))].sort(
      (a, b) => a - b
    );
    for (const fold of uniqueFolds) {
      const trainIdx: number[] = [];
      const testIdx: number[] = [];
      for (let i = 0; i < this.testFold.length; i++) {
        if ((this.testFold[i] ?? -1) === fold) testIdx.push(i);
        else if ((this.testFold[i] ?? -1) !== -1) trainIdx.push(i);
      }
      yield {
        trainIndices: new Int32Array(trainIdx),
        testIndices: new Int32Array(testIdx),
      };
    }
  }
}
