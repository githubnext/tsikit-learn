/**
 * Additional dataset generators — ported from sklearn.datasets
 * make_low_rank_matrix, make_sparse_coded_signal, make_biclusters, make_checkerboard
 */

export interface LowRankMatrixOptions {
  nSamples?: number;
  nFeatures?: number;
  effectiveRank?: number;
  tailStrength?: number;
  randomState?: number | null;
}

export interface LowRankMatrixResult {
  X: Float64Array[];
}

/**
 * Generate a mostly low-rank matrix with bell-shaped singular values.
 * Useful for testing matrix decomposition algorithms.
 */
export function makeLowRankMatrix(options: LowRankMatrixOptions = {}): LowRankMatrixResult {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 100;
  const effectiveRank = options.effectiveRank ?? 10;
  const tailStrength = options.tailStrength ?? 0.5;

  let seed = options.randomState ?? 42;
  function randn(): number {
    seed = (1664525 * seed + 1013904223) & 0x7fffffff;
    const u1 = seed / 0x7fffffff;
    seed = (1664525 * seed + 1013904223) & 0x7fffffff;
    const u2 = seed / 0x7fffffff;
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  const n = Math.min(nSamples, nFeatures);

  // Singular values: bell-shaped around effectiveRank
  const singularValues = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i - effectiveRank) / (effectiveRank / 2);
    singularValues[i] = Math.exp(-0.5 * x * x) * (1 - tailStrength) + tailStrength / n;
  }

  // Random orthonormal U (nSamples x n) and V (nFeatures x n)
  // Simplified: just use random Gaussian matrices (not fully orthogonal)
  const U: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) row[j] = randn();
    return row;
  });

  const V: Float64Array[] = Array.from({ length: nFeatures }, () => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) row[j] = randn();
    return row;
  });

  // X = U @ diag(singularValues) @ V.T
  const X: Float64Array[] = Array.from({ length: nSamples }, (_, i) => {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let val = 0;
      for (let k = 0; k < n; k++) {
        val += (U[i]![k] ?? 0) * (singularValues[k] ?? 0) * (V[j]![k] ?? 0);
      }
      row[j] = val;
    }
    return row;
  });

  return { X };
}

export interface SparseCodingOptions {
  nSamples?: number;
  nComponents?: number;
  nFeatures?: number;
  nNonzeroCoefs?: number;
  randomState?: number | null;
}

export interface SparseCodingResult {
  X: Float64Array[];
  dictionary: Float64Array[];
  code: Float64Array[];
}

/**
 * Generate a sparse signal using a fixed dictionary.
 * Useful for testing dictionary learning algorithms.
 */
export function makeSparseCodedSignal(options: SparseCodingOptions = {}): SparseCodingResult {
  const nSamples = options.nSamples ?? 100;
  const nComponents = options.nComponents ?? 40;
  const nFeatures = options.nFeatures ?? 64;
  const nNonzeroCoefs = options.nNonzeroCoefs ?? 3;

  let seed = options.randomState ?? 0;
  function rand(): number {
    seed = (1664525 * seed + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  }
  function randn(): number {
    const u1 = rand() + 1e-10;
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Random dictionary (nComponents x nFeatures), normalized atoms
  const dictionary: Float64Array[] = Array.from({ length: nComponents }, () => {
    const atom = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) atom[j] = randn();
    let norm = 0;
    for (let j = 0; j < nFeatures; j++) norm += (atom[j] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let j = 0; j < nFeatures; j++) atom[j]! /= norm;
    return atom;
  });

  // Sparse codes (nSamples x nComponents)
  const code: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(nComponents);
    // Pick nNonzeroCoefs random non-zero positions
    const positions: number[] = [];
    const available = Array.from({ length: nComponents }, (_, i) => i);
    for (let k = 0; k < nNonzeroCoefs && available.length > 0; k++) {
      const idx = Math.floor(rand() * available.length);
      positions.push(available[idx]!);
      available.splice(idx, 1);
    }
    for (const pos of positions) {
      row[pos] = randn();
    }
    return row;
  });

  // X = code @ dictionary
  const X: Float64Array[] = Array.from({ length: nSamples }, (_, i) => {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let val = 0;
      for (let k = 0; k < nComponents; k++) {
        val += (code[i]![k] ?? 0) * (dictionary[k]![j] ?? 0);
      }
      row[j] = val;
    }
    return row;
  });

  return { X, dictionary, code };
}

export interface BiclustersOptions {
  shape?: [number, number];
  nClusters?: number;
  noise?: number;
  minsize?: number;
  randomState?: number | null;
}

export interface BiclustersResult {
  X: Float64Array[];
  rows: boolean[][];
  columns: boolean[][];
}

/**
 * Generate a 2D array with planted biclusters.
 * Useful for testing biclustering algorithms.
 */
export function makeBiclusters(options: BiclustersOptions = {}): BiclustersResult {
  const [nRows, nCols] = options.shape ?? [100, 100];
  const nClusters = options.nClusters ?? 5;
  const noise = options.noise ?? 0.0;

  let seed = options.randomState ?? 0;
  function rand(): number {
    seed = (1664525 * seed + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  // Assign rows and columns to clusters
  const rowAssignments = new Int32Array(nRows);
  const colAssignments = new Int32Array(nCols);
  for (let i = 0; i < nRows; i++) rowAssignments[i] = Math.floor(rand() * nClusters);
  for (let j = 0; j < nCols; j++) colAssignments[j] = Math.floor(rand() * nClusters);

  const X: Float64Array[] = Array.from({ length: nRows }, (_, i) => {
    const row = new Float64Array(nCols);
    for (let j = 0; j < nCols; j++) {
      const sameBicluster = (rowAssignments[i] ?? 0) === (colAssignments[j] ?? 0) ? 1 : 0;
      const noiseVal = noise > 0 ? (rand() - 0.5) * noise : 0;
      row[j] = sameBicluster + noiseVal;
    }
    return row;
  });

  // Build membership arrays
  const rows: boolean[][] = Array.from({ length: nClusters }, (_, c) =>
    Array.from({ length: nRows }, (__, i) => (rowAssignments[i] ?? 0) === c)
  );
  const columns: boolean[][] = Array.from({ length: nClusters }, (_, c) =>
    Array.from({ length: nCols }, (__, j) => (colAssignments[j] ?? 0) === c)
  );

  return { X, rows, columns };
}

export interface CheckerboardOptions {
  shape?: [number, number];
  nClusters?: [number, number];
  noise?: number;
  randomState?: number | null;
}

export interface CheckerboardResult {
  X: Float64Array[];
  rows: boolean[][];
  columns: boolean[][];
}

/**
 * Generate a checkerboard pattern dataset for testing biclustering.
 */
export function makeCheckerboard(options: CheckerboardOptions = {}): CheckerboardResult {
  const [nRows, nCols] = options.shape ?? [100, 100];
  const [nRowClusters, nColClusters] = options.nClusters ?? [4, 4];
  const noise = options.noise ?? 0.0;

  let seed = options.randomState ?? 0;
  function rand(): number {
    seed = (1664525 * seed + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  const X: Float64Array[] = Array.from({ length: nRows }, (_, i) => {
    const row = new Float64Array(nCols);
    const rowCluster = Math.floor(i / Math.ceil(nRows / nRowClusters));
    for (let j = 0; j < nCols; j++) {
      const colCluster = Math.floor(j / Math.ceil(nCols / nColClusters));
      const val = ((rowCluster + colCluster) % 2 === 0) ? 1 : 0;
      const noiseVal = noise > 0 ? (rand() - 0.5) * noise : 0;
      row[j] = val + noiseVal;
    }
    return row;
  });

  const rows: boolean[][] = Array.from({ length: nRowClusters }, (_, rc) =>
    Array.from({ length: nRows }, (__, i) =>
      Math.floor(i / Math.ceil(nRows / nRowClusters)) === rc
    )
  );
  const columns: boolean[][] = Array.from({ length: nColClusters }, (_, cc) =>
    Array.from({ length: nCols }, (__, j) =>
      Math.floor(j / Math.ceil(nCols / nColClusters)) === cc
    )
  );

  return { X, rows, columns };
}
