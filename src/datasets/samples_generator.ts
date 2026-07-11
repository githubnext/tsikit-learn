/**
 * Additional synthetic dataset generators.
 * Mirrors sklearn.datasets: make_hastie_10_2, make_friedman1/2/3,
 * make_sparse_uncorrelated, make_checkerboard, make_multilabel_classification.
 */

/** Result type for generated datasets. */
export interface SamplesDatasetResult {
  X: Float64Array[];
  y: Float64Array | Int32Array;
}

/** Simple seeded Mulberry32 RNG for reproducibility. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-14);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * make_hastie_10_2 — 10-feature binary classification problem.
 * y = sign(sum(X_i^2) - 9.34) where X ~ N(0,1).
 */
export function makeHastie10_2(
  nSamples = 12000,
  randomState = 0,
): { X: Float64Array[]; y: Int32Array } {
  const rng = makeRng(randomState);
  const X: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(10);
    for (let j = 0; j < 10; j++) row[j]! = randn(rng);
    return row;
  });
  const y = Int32Array.from(X, (row) => {
    let s = 0;
    for (const v of row) s += v * v;
    return s > 9.34 ? 1 : -1;
  });
  return { X, y };
}

/**
 * make_friedman1 — regression dataset from Friedman (1991).
 * y = 10*sin(π*X0*X1) + 20*(X2-0.5)^2 + 10*X3 + 5*X4 + noise
 */
export function makeFriedman1(
  nSamples = 100,
  nFeatures = 10,
  noise = 0.0,
  randomState = 0,
): SamplesDatasetResult {
  if (nFeatures < 5)
    throw new Error("makeFriedman1 requires at least 5 features");
  const rng = makeRng(randomState);
  const X: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j]! = rng();
    return row;
  });
  const y = Float64Array.from(X, (row) => {
    const x0 = row[0]! ?? 0;
    const x1 = row[1]! ?? 0;
    const x2 = row[2]! ?? 0;
    const x3 = row[3]! ?? 0;
    const x4 = row[4]! ?? 0;
    return (
      10 * Math.sin(Math.PI * x0 * x1) +
      20 * (x2 - 0.5) ** 2 +
      10 * x3 +
      5 * x4 +
      (noise > 0 ? noise * randn(rng) : 0)
    );
  });
  return { X, y };
}

/**
 * make_friedman2 — regression with nonlinear interactions.
 * y = sqrt(X0^2 + (X1*X2 - 1/(X1*X3))^2) + noise
 */
export function makeFriedman2(
  nSamples = 100,
  noise = 0.0,
  randomState = 0,
): SamplesDatasetResult {
  const rng = makeRng(randomState);
  const bounds: [number, number][] = [
    [0, 100],
    [40 * Math.PI, 560 * Math.PI],
    [0, 1],
    [1, 11],
  ];
  const X: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(4);
    for (let j = 0; j < 4; j++) {
      const [lo, hi] = bounds[j]!;
      row[j]! = lo + rng() * (hi - lo);
    }
    return row;
  });
  const y = Float64Array.from(X, (row) => {
    const x0 = row[0]! ?? 0;
    const x1 = row[1]! ?? 0;
    const x2 = row[2]! ?? 0;
    const x3 = Math.max(row[3]! ?? 1, 1e-6);
    const inner = x1 * x2 - 1 / (x1 * x3);
    return (
      Math.sqrt(x0 ** 2 + inner ** 2) + (noise > 0 ? noise * randn(rng) : 0)
    );
  });
  return { X, y };
}

/**
 * make_friedman3 — regression with arctan transformation.
 * y = arctan((X1*X2 - 1/(X1*X3)) / X0) + noise
 */
export function makeFriedman3(
  nSamples = 100,
  noise = 0.0,
  randomState = 0,
): SamplesDatasetResult {
  const rng = makeRng(randomState);
  const bounds: [number, number][] = [
    [0, 100],
    [40 * Math.PI, 560 * Math.PI],
    [0, 1],
    [1, 11],
  ];
  const X: Float64Array[] = Array.from({ length: nSamples }, () => {
    const row = new Float64Array(4);
    for (let j = 0; j < 4; j++) {
      const [lo, hi] = bounds[j]!;
      row[j]! = lo + rng() * (hi - lo);
    }
    return row;
  });
  const y = Float64Array.from(X, (row) => {
    const x0 = Math.max(Math.abs(row[0]! ?? 0), 1e-6);
    const x1 = row[1]! ?? 0;
    const x2 = row[2]! ?? 0;
    const x3 = Math.max(row[3]! ?? 1, 1e-6);
    const inner = x1 * x2 - 1 / (x1 * x3);
    return Math.atan(inner / x0) + (noise > 0 ? noise * randn(rng) : 0);
  });
  return { X, y };
}

/**
 * make_sparse_uncorrelated — regression dataset with 4 informative features
 * and `nFeatures - 4` noise features.
 */
export function makeSparseUncorrelated(
  nSamples = 100,
  nFeatures = 10,
  randomState = 0,
): SamplesDatasetResult {
  const rng = makeRng(randomState);
  const X: Float64Array[] = Array.from({ length: nSamples }, () =>
    Float64Array.from({ length: nFeatures }, () => randn(rng)),
  );
  const coef = [1, 2, 0.5, -0.5]; // informative coefficients
  const y = Float64Array.from(X, (row) => {
    let s = 0;
    for (let j = 0; j < coef.length; j++) s += (coef[j]! ?? 0) * (row[j]! ?? 0);
    s += randn(rng);
    return s;
  });
  return { X, y };
}

/**
 * make_multilabel_classification — random multilabel dataset.
 *
 * @param nSamples - Number of samples.
 * @param nFeatures - Number of features.
 * @param nClasses - Number of classes (labels).
 * @param nLabels - Average number of labels per sample.
 * @param randomState - Random seed.
 */
export function makeMultilabelClassification(
  nSamples = 100,
  nFeatures = 20,
  nClasses = 5,
  nLabels = 2,
  randomState = 0,
): { X: Float64Array[]; y: Int32Array[] } {
  const rng = makeRng(randomState);
  const X: Float64Array[] = Array.from({ length: nSamples }, () =>
    Float64Array.from({ length: nFeatures }, () => (rng() > 0.5 ? 1 : 0)),
  );
  const y: Int32Array[] = Array.from({ length: nSamples }, () => {
    const row = new Int32Array(nClasses);
    const nActive = Math.max(1, Math.round(nLabels + (rng() - 0.5) * 2));
    for (let k = 0; k < nActive && k < nClasses; k++) {
      row[Math.floor(rng() * nClasses)]! = 1;
    }
    return row;
  });
  return { X, y };
}

/**
 * make_checkerboard — checkerboard pattern for biclustering.
 *
 * @param shape - [n_rows, n_cols].
 * @param nClusters - [n_row_clusters, n_col_clusters].
 * @param noise - Noise standard deviation.
 * @param randomState - Random seed.
 */
export function makeCheckerboard(
  shape: [number, number] = [300, 300],
  nClusters: [number, number] = [4, 3],
  noise = 0.5,
  randomState = 0,
): { data: Float64Array[]; rowLabels: Int32Array; colLabels: Int32Array } {
  const rng = makeRng(randomState);
  const [nRows, nCols] = shape;
  const [nRowC, nColC] = nClusters;
  const rowLabels = Int32Array.from({ length: nRows }, (_, i) => i % nRowC);
  const colLabels = Int32Array.from({ length: nCols }, (_, j) => j % nColC);
  const data: Float64Array[] = Array.from({ length: nRows }, (_, i) => {
    const row = new Float64Array(nCols);
    for (let j = 0; j < nCols; j++) {
      const match = rowLabels[i]! % 2 === colLabels[j]! % 2;
      row[j]! = (match ? 1 : 0) + noise * randn(rng);
    }
    return row;
  });
  return { data, rowLabels, colLabels };
}
