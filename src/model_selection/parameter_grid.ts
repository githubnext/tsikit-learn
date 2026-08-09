/**
 * ParameterGrid, ParameterSampler, ShuffleSplit, GroupKFold: additional model selection utilities.
 * Mirrors sklearn.model_selection parameter grid/sampler and additional CV splitters.
 */

import { BaseEstimator } from "../base.js";

export type ParamGrid = Record<string, unknown[]>;

/**
 * Grid of parameters with a discrete number of values for each.
 * Exhaustive parameter grid for use with GridSearchCV.
 */
export class ParameterGrid {
  paramGrid: ParamGrid | ParamGrid[];

  constructor(paramGrid: ParamGrid | ParamGrid[]) {
    this.paramGrid = paramGrid;
  }

  *[Symbol.iterator](): Generator<Record<string, unknown>> {
    const grids = Array.isArray(this.paramGrid)
      ? this.paramGrid
      : [this.paramGrid];
    for (const grid of grids) {
      const keys = Object.keys(grid);
      if (keys.length === 0) {
        yield {};
        continue;
      }
      const values = keys.map((k) => grid[k]!);
      const counts = values.map((v) => v.length);
      const total = counts.reduce((a, b) => a * b, 1);
      for (let i = 0; i < total; i++) {
        const params: Record<string, unknown> = {};
        let idx = i;
        for (let ki = 0; ki < keys.length; ki++) {
          const n = counts[ki] ?? 1;
          params[keys[ki]!] = values[ki]![idx % n];
          idx = Math.floor(idx / n);
        }
        yield params;
      }
    }
  }

  toArray(): Record<string, unknown>[] {
    return [...this];
  }

  get length(): number {
    const grids = Array.isArray(this.paramGrid)
      ? this.paramGrid
      : [this.paramGrid];
    let total = 0;
    for (const grid of grids) {
      const keys = Object.keys(grid);
      let prod = 1;
      for (const k of keys) prod *= grid[k]!.length;
      total += prod;
    }
    return total;
  }
}

export interface ParameterSamplerOptions {
  nIter: number;
  randomState?: number;
}

/**
 * Generator of parameter settings sampled from a parameter grid.
 * Supports distributions (objects with rvs method) or lists of values.
 */
export class ParameterSampler {
  paramDistributions: Record<
    string,
    unknown[] | { rvs(seed: number): unknown }
  >;
  nIter: number;
  randomState: number;

  constructor(
    paramDistributions: Record<
      string,
      unknown[] | { rvs(seed: number): unknown }
    >,
    opts: ParameterSamplerOptions,
  ) {
    this.paramDistributions = paramDistributions;
    this.nIter = opts.nIter;
    this.randomState = opts.randomState ?? 0;
  }

  *[Symbol.iterator](): Generator<Record<string, unknown>> {
    let seed = this.randomState;
    for (let i = 0; i < this.nIter; i++) {
      const params: Record<string, unknown> = {};
      for (const [key, dist] of Object.entries(this.paramDistributions)) {
        seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
        if (Array.isArray(dist)) {
          params[key] = dist[Math.abs(seed) % dist.length];
        } else {
          params[key] = dist.rvs(seed);
        }
      }
      yield params;
    }
  }

  toArray(): Record<string, unknown>[] {
    return [...this];
  }
}

export interface ShuffleSplitOptions {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
  randomState?: number;
}

export interface ShuffleSplitFold {
  trainIndex: Int32Array;
  testIndex: Int32Array;
}

/**
 * Random permutation cross-validator.
 * Randomly shuffles and splits into train/test sets.
 */
export class ShuffleSplit {
  nSplits: number;
  testSize: number;
  trainSize: number | null;
  randomState: number;

  constructor(opts: ShuffleSplitOptions = {}) {
    this.nSplits = opts.nSplits ?? 10;
    this.testSize = opts.testSize ?? 0.1;
    this.trainSize = opts.trainSize ?? null;
    this.randomState = opts.randomState ?? 0;
  }

  *split(X: unknown[]): Generator<ShuffleSplitFold> {
    const n = X.length;
    const nTest = Math.floor(
      this.testSize < 1 ? n * this.testSize : this.testSize,
    );
    const nTrain =
      this.trainSize !== null
        ? this.trainSize < 1
          ? Math.floor(n * this.trainSize)
          : this.trainSize
        : n - nTest;
    let seed = this.randomState;

    for (let split = 0; split < this.nSplits; split++) {
      // Fisher-Yates shuffle
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
        const j = Math.abs(seed) % (i + 1);
        const tmp = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = tmp;
      }
      yield {
        testIndex: new Int32Array(perm.slice(0, nTest)),
        trainIndex: new Int32Array(perm.slice(nTest, nTest + nTrain)),
      };
    }
  }
}

export interface GroupKFoldOptions {
  nSplits?: number;
}

/**
 * K-fold iterator variant with non-overlapping groups.
 */
export class GroupKFold {
  nSplits: number;

  constructor(opts: GroupKFoldOptions = {}) {
    this.nSplits = opts.nSplits ?? 5;
  }

  *split(
    X: unknown[],
    _y?: unknown[],
    groups?: number[],
  ): Generator<ShuffleSplitFold> {
    const n = X.length;
    const grps = groups ?? Array.from({ length: n }, (_, i) => i);
    const uniqueGroups = [...new Set(grps)].sort((a, b) => a - b);
    const k = Math.min(this.nSplits, uniqueGroups.length);
    const foldSize = Math.floor(uniqueGroups.length / k);

    for (let fold = 0; fold < k; fold++) {
      const start = fold * foldSize;
      const end = fold === k - 1 ? uniqueGroups.length : start + foldSize;
      const testGroups = new Set(uniqueGroups.slice(start, end));

      const testIdx: number[] = [];
      const trainIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (testGroups.has(grps[i]!)) testIdx.push(i);
        else trainIdx.push(i);
      }
      yield {
        trainIndex: new Int32Array(trainIdx),
        testIndex: new Int32Array(testIdx),
      };
    }
  }
}

export interface RepeatedKFoldOptions {
  nSplits?: number;
  nRepeats?: number;
  randomState?: number;
}

/**
 * Repeated K-Fold cross validator.
 */
export class RepeatedKFold {
  nSplits: number;
  nRepeats: number;
  randomState: number;

  constructor(opts: RepeatedKFoldOptions = {}) {
    this.nSplits = opts.nSplits ?? 5;
    this.nRepeats = opts.nRepeats ?? 10;
    this.randomState = opts.randomState ?? 0;
  }

  *split(X: unknown[]): Generator<ShuffleSplitFold> {
    const n = X.length;
    let seed = this.randomState;
    const foldSize = Math.floor(n / this.nSplits);

    for (let rep = 0; rep < this.nRepeats; rep++) {
      // Shuffle indices
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
        const j = Math.abs(seed) % (i + 1);
        const tmp = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = tmp;
      }

      for (let fold = 0; fold < this.nSplits; fold++) {
        const start = fold * foldSize;
        const end = fold === this.nSplits - 1 ? n : start + foldSize;
        const testIdx = perm.slice(start, end);
        const trainIdx = [...perm.slice(0, start), ...perm.slice(end)];
        yield {
          trainIndex: new Int32Array(trainIdx),
          testIndex: new Int32Array(testIdx),
        };
      }
    }
  }
}

export interface LeaveOneOutFold {
  trainIndex: Int32Array;
  testIndex: Int32Array;
}

/**
 * Leave-One-Out cross-validator.
 */
export class LeaveOneOut {
  *split(X: unknown[]): Generator<LeaveOneOutFold> {
    const n = X.length;
    for (let i = 0; i < n; i++) {
      const trainIdx = Array.from({ length: n - 1 }, (_, k) =>
        k >= i ? k + 1 : k,
      );
      yield {
        trainIndex: new Int32Array(trainIdx),
        testIndex: new Int32Array([i]),
      };
    }
  }

  getNSplits(X: unknown[]): number {
    return X.length;
  }
}
