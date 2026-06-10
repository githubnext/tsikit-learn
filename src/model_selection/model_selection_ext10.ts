/**
 * ShuffleSplit, LeaveOneOut, and LeavePOut cross-validation splitters.
 */

export interface SplitResult {
  train: number[];
  test: number[];
}

export class ShuffleSplit {
  constructor(private nSplits = 10, private testSize = 0.1, private randomState = 42) {}

  split(X: Float64Array[]): SplitResult[] {
    const n = X.length;
    const nTest = typeof this.testSize === 'number' && this.testSize < 1
      ? Math.floor(n * this.testSize)
      : Math.floor(this.testSize as number);
    const splits: SplitResult[] = [];
    let seed = this.randomState;
    for (let s = 0; s < this.nSplits; s++) {
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        seed = (seed * 6364136223846793005 + 1442695040888963407) >>> 0;
        const j = seed % (i + 1);
        [perm[i], perm[j]] = [perm[j]!, perm[i]!];
      }
      splits.push({ test: perm.slice(0, nTest), train: perm.slice(nTest) });
    }
    return splits;
  }
}

export class LeaveOneOut {
  split(X: Float64Array[]): SplitResult[] {
    const n = X.length;
    return Array.from({ length: n }, (_, i) => ({
      test: [i],
      train: Array.from({ length: n - 1 }, (__, j) => j < i ? j : j + 1),
    }));
  }

  getNCVSplits(X: Float64Array[]): number { return X.length; }
}

export class LeavePOut {
  constructor(private p = 2) {}

  split(X: Float64Array[]): SplitResult[] {
    const n = X.length;
    const splits: SplitResult[] = [];
    // Generate all combinations of p indices for test
    const combine = (start: number, combo: number[]) => {
      if (combo.length === this.p) {
        const testSet = new Set(combo);
        splits.push({ test: combo, train: Array.from({ length: n }, (_, i) => i).filter(i => !testSet.has(i)) });
        return;
      }
      for (let i = start; i <= n - (this.p - combo.length); i++) combine(i + 1, [...combo, i]);
    };
    combine(0, []);
    return splits;
  }

  getNCVSplits(n: number): number {
    let result = 1;
    for (let i = 0; i < this.p; i++) { result *= (n - i); result /= (i + 1); }
    return Math.round(result);
  }
}

export class StratifiedShuffleSplit {
  constructor(private nSplits = 10, private testSize = 0.1, private randomState = 42) {}

  split(X: Float64Array[], y: Int32Array): SplitResult[] {
    const n = X.length;
    const nTest = typeof this.testSize === 'number' && this.testSize < 1
      ? Math.floor(n * this.testSize) : Math.floor(this.testSize as number);
    const classes = Array.from(new Set(Array.from(y)));
    const splits: SplitResult[] = [];
    let seed = this.randomState;
    for (let s = 0; s < this.nSplits; s++) {
      const test: number[] = [], train: number[] = [];
      for (const cls of classes) {
        const clsIndices = Array.from({ length: n }, (_, i) => i).filter(i => y[i] === cls);
        const nTestCls = Math.max(1, Math.round(nTest * clsIndices.length / n));
        const perm = [...clsIndices];
        for (let i = perm.length - 1; i > 0; i--) {
          seed = (seed * 6364136223846793005 + 1442695040888963407) >>> 0;
          const j = seed % (i + 1);
          [perm[i], perm[j]] = [perm[j]!, perm[i]!];
        }
        test.push(...perm.slice(0, nTestCls));
        train.push(...perm.slice(nTestCls));
      }
      splits.push({ test, train });
    }
    return splits;
  }
}
