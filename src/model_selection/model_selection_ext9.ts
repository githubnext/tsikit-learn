/**
 * RepeatedKFold, RepeatedStratifiedKFold, and GroupKFold cross-validation.
 */

export interface CVSplit {
  trainIndices: number[];
  testIndices: number[];
}

export class RepeatedKFold {
  constructor(private nSplits = 5, private nRepeats = 10, private randomState = 42) {}

  split(X: Float64Array[]): CVSplit[][] {
    const n = X.length;
    const allReps: CVSplit[][] = [];
    let seed = this.randomState;
    for (let rep = 0; rep < this.nRepeats; rep++) {
      // Shuffle indices
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        seed = (seed * 6364136223846793005 + 1442695040888963407) >>> 0;
        const j = seed % (i + 1);
        [perm[i], perm[j]] = [perm[j]!, perm[i]!];
      }
      const foldSize = Math.floor(n / this.nSplits);
      const repSplits: CVSplit[] = [];
      for (let fold = 0; fold < this.nSplits; fold++) {
        const start = fold * foldSize;
        const end = fold === this.nSplits - 1 ? n : start + foldSize;
        const testIndices = perm.slice(start, end);
        const trainIndices = [...perm.slice(0, start), ...perm.slice(end)];
        repSplits.push({ trainIndices, testIndices });
      }
      allReps.push(repSplits);
      seed = (seed + rep) >>> 0;
    }
    return allReps;
  }

  getNCVSplits(): number { return this.nSplits * this.nRepeats; }
}

export class GroupKFold {
  constructor(private nSplits = 5) {}

  split(X: Float64Array[], y: Float64Array, groups: Int32Array): CVSplit[] {
    const n = X.length;
    const uniqueGroups = Array.from(new Set(Array.from(groups)));
    if (uniqueGroups.length < this.nSplits) throw new Error(`Not enough groups for ${this.nSplits} folds`);
    const foldSize = Math.floor(uniqueGroups.length / this.nSplits);
    const splits: CVSplit[] = [];
    for (let fold = 0; fold < this.nSplits; fold++) {
      const start = fold * foldSize;
      const end = fold === this.nSplits - 1 ? uniqueGroups.length : start + foldSize;
      const testGroups = new Set(uniqueGroups.slice(start, end));
      const testIndices = Array.from({ length: n }, (_, i) => i).filter(i => testGroups.has(groups[i]!));
      const trainIndices = Array.from({ length: n }, (_, i) => i).filter(i => !testGroups.has(groups[i]!));
      splits.push({ trainIndices, testIndices });
    }
    void y;
    return splits;
  }
}

export class TimeSeriesSplit {
  constructor(private nSplits = 5, private maxTrainSize: number | null = null, private gap = 0) {}

  split(X: Float64Array[]): CVSplit[] {
    const n = X.length;
    const testSize = Math.floor(n / (this.nSplits + 1));
    const splits: CVSplit[] = [];
    for (let fold = 0; fold < this.nSplits; fold++) {
      const testEnd = (fold + 2) * testSize;
      const testStart = testEnd - testSize;
      const trainEnd = testStart - this.gap;
      const trainStart = this.maxTrainSize !== null ? Math.max(0, trainEnd - this.maxTrainSize) : 0;
      splits.push({
        trainIndices: Array.from({ length: trainEnd - trainStart }, (_, i) => trainStart + i),
        testIndices: Array.from({ length: testEnd - testStart }, (_, i) => testStart + i),
      });
    }
    return splits;
  }
}
