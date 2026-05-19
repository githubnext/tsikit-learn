/**
 * Bootstrap confidence intervals and permutation tests for metrics.
 * Mirrors sklearn.utils.resample-based validation utilities.
 */

export interface BootstrapCIResult {
  estimate: number;
  lower: number;
  upper: number;
  std: number;
  nBootstrap: number;
}

/**
 * Compute bootstrap confidence interval for a metric function.
 */
export function bootstrapCI(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  metricFn: (yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number,
  options: {
    nBootstrap?: number;
    confidenceLevel?: number;
    randomState?: number;
  } = {}
): BootstrapCIResult {
  const nBootstrap = options.nBootstrap ?? 1000;
  const alpha = 1 - (options.confidenceLevel ?? 0.95);
  const n = yTrue.length;

  let seed = options.randomState ?? 42;
  function lcg(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const baseEstimate = metricFn(yTrue, yPred);
  const bootstrapScores: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    const idxTrue = yTrue instanceof Float64Array ? new Float64Array(n) : new Int32Array(n);
    const idxPred = yPred instanceof Float64Array ? new Float64Array(n) : new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(lcg() * n);
      if (idxTrue instanceof Float64Array) {
        (idxTrue as Float64Array)[i] = yTrue[idx] ?? 0;
      } else {
        (idxTrue as Int32Array)[i] = (yTrue as Int32Array)[idx] ?? 0;
      }
      if (idxPred instanceof Float64Array) {
        (idxPred as Float64Array)[i] = yPred[idx] ?? 0;
      } else {
        (idxPred as Int32Array)[i] = (yPred as Int32Array)[idx] ?? 0;
      }
    }
    bootstrapScores.push(metricFn(idxTrue, idxPred));
  }

  bootstrapScores.sort((a, b) => a - b);
  const lower = bootstrapScores[Math.floor(alpha / 2 * nBootstrap)] ?? 0;
  const upper = bootstrapScores[Math.floor((1 - alpha / 2) * nBootstrap)] ?? 1;
  const mean = bootstrapScores.reduce((s, v) => s + v, 0) / nBootstrap;
  const variance = bootstrapScores.reduce((s, v) => s + (v - mean) ** 2, 0) / nBootstrap;

  return { estimate: baseEstimate, lower, upper, std: Math.sqrt(variance), nBootstrap };
}

/**
 * Permutation test for a metric.
 * Tests whether the observed metric is significantly better than chance.
 */
export interface PermutationTestResult {
  score: number;
  permutationScores: Float64Array;
  pValue: number;
}

export function permutationTest(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  metricFn: (yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number,
  options: {
    nPermutations?: number;
    randomState?: number;
  } = {}
): PermutationTestResult {
  const nPermutations = options.nPermutations ?? 1000;
  const n = yTrue.length;
  let seed = options.randomState ?? 42;

  function lcg(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const score = metricFn(yTrue, yPred);
  const permScores = new Float64Array(nPermutations);

  // Permute yPred
  const shuffled = yPred instanceof Float64Array ? new Float64Array(yPred) : new Int32Array(yPred as Int32Array);
  for (let p = 0; p < nPermutations; p++) {
    // Fisher-Yates shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(lcg() * (i + 1));
      const tmp = shuffled[i]!;
      shuffled[i] = shuffled[j]!;
      shuffled[j] = tmp;
    }
    permScores[p] = metricFn(yTrue, shuffled);
  }

  let countAbove = 0;
  for (let p = 0; p < nPermutations; p++) {
    if ((permScores[p] ?? 0) >= score) countAbove++;
  }
  const pValue = (countAbove + 1) / (nPermutations + 1);

  return { score, permutationScores: permScores, pValue };
}

/**
 * Repeated K-Fold cross-validation.
 */
export interface RepeatedKFoldOptions {
  nSplits?: number;
  nRepeats?: number;
  randomState?: number;
}

export class RepeatedKFold {
  nSplits: number;
  nRepeats: number;
  randomState: number;

  constructor(options: RepeatedKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.nRepeats = options.nRepeats ?? 10;
    this.randomState = options.randomState ?? 0;
  }

  *split(X: Float64Array[] | Int32Array[]): Generator<[Int32Array, Int32Array]> {
    const n = X.length;
    let seed = this.randomState;

    for (let r = 0; r < this.nRepeats; r++) {
      // Create shuffled indices
      const indices = new Int32Array(n);
      for (let i = 0; i < n; i++) indices[i] = i;

      // Fisher-Yates shuffle
      for (let i = n - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) & 0xffffffff;
        const j = ((seed >>> 0) / 0xffffffff * (i + 1)) | 0;
        const tmp = indices[i]!;
        indices[i] = indices[j]!;
        indices[j] = tmp;
      }

      const foldSize = Math.floor(n / this.nSplits);
      for (let f = 0; f < this.nSplits; f++) {
        const testStart = f * foldSize;
        const testEnd = f === this.nSplits - 1 ? n : testStart + foldSize;
        const testIdx = indices.slice(testStart, testEnd);
        const trainIdx = new Int32Array([
          ...Array.from(indices.slice(0, testStart)),
          ...Array.from(indices.slice(testEnd))
        ]);
        yield [trainIdx, testIdx];
      }
    }
  }

  getNSplits(): number {
    return this.nSplits * this.nRepeats;
  }
}

/**
 * Repeated Stratified K-Fold.
 */
export class RepeatedStratifiedKFold {
  nSplits: number;
  nRepeats: number;
  randomState: number;

  constructor(options: RepeatedKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.nRepeats = options.nRepeats ?? 10;
    this.randomState = options.randomState ?? 0;
  }

  *split(X: Float64Array[], y: Int32Array): Generator<[Int32Array, Int32Array]> {
    const n = X.length;
    let seed = this.randomState;

    // Group indices by class
    const classIndices = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const cls = y[i] ?? 0;
      if (!classIndices.has(cls)) classIndices.set(cls, []);
      classIndices.get(cls)!.push(i);
    }

    for (let r = 0; r < this.nRepeats; r++) {
      // Shuffle within each class
      const shuffledByClass = new Map<number, number[]>();
      for (const [cls, idxs] of classIndices) {
        const arr = [...idxs];
        for (let i = arr.length - 1; i > 0; i--) {
          seed = (seed * 1664525 + 1013904223) & 0xffffffff;
          const j = ((seed >>> 0) / 0xffffffff * (i + 1)) | 0;
          const tmp = arr[i]!;
          arr[i] = arr[j]!;
          arr[j] = tmp;
        }
        shuffledByClass.set(cls, arr);
      }

      // Create fold assignments
      const foldAssign = new Int32Array(n);
      for (const [, idxs] of shuffledByClass) {
        for (let i = 0; i < idxs.length; i++) {
          foldAssign[idxs[i]!] = i % this.nSplits;
        }
      }

      for (let f = 0; f < this.nSplits; f++) {
        const trainIdxs: number[] = [];
        const testIdxs: number[] = [];
        for (let i = 0; i < n; i++) {
          if (foldAssign[i] === f) testIdxs.push(i);
          else trainIdxs.push(i);
        }
        yield [new Int32Array(trainIdxs), new Int32Array(testIdxs)];
      }
    }
  }

  getNSplits(): number {
    return this.nSplits * this.nRepeats;
  }
}
