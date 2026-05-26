/**
 * Extended datasets: makeMultilabelClassification, makeMultivariateNormal, makeCheckerboard, makeS_curve
 */

export interface MultilabelDataset {
  X: Float64Array[];
  Y: Int32Array[];
  nClasses: number;
}

export function makeMultilabelClassification(
  nSamples = 100,
  nFeatures = 20,
  nClasses = 5,
  nLabels = 2,
  randomState?: number
): MultilabelDataset {
  const rng = randomState !== undefined ? seededRng(randomState) : Math.random;
  const X: Float64Array[] = [];
  const Y: Int32Array[] = [];
  for (let i = 0; i < nSamples; i++) {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j] = rng() * 2 - 1;
    X.push(row);
    const labels = new Int32Array(nClasses);
    const selected = new Set<number>();
    while (selected.size < nLabels) selected.add(Math.floor(rng() * nClasses));
    for (const l of selected) labels[l] = 1;
    Y.push(labels);
  }
  return { X, Y, nClasses };
}

function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

export interface MultivariateNormalDataset {
  X: Float64Array[];
  mean: Float64Array;
  cov: Float64Array[];
}

export function makeMultivariateNormal(
  nSamples = 100,
  mean: Float64Array,
  cov: Float64Array[]
): MultivariateNormalDataset {
  const nFeatures = mean.length;
  // Cholesky decomposition of cov
  const L: Float64Array[] = Array.from({ length: nFeatures }, () => new Float64Array(nFeatures));
  for (let i = 0; i < nFeatures; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = cov[i]![j] ?? 0;
      for (let k = 0; k < j; k++) sum -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
      L[i]![j] = i === j ? Math.sqrt(Math.max(0, sum)) : (L[j]![j] ?? 1) < 1e-10 ? 0 : sum / (L[j]![j] ?? 1);
    }
  }
  // Sample z ~ N(0, I) then x = L*z + mean
  const X: Float64Array[] = [];
  for (let s = 0; s < nSamples; s++) {
    const z = new Float64Array(nFeatures);
    for (let i = 0; i < nFeatures; i++) {
      const u1 = Math.random(), u2 = Math.random();
      z[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    }
    const x = new Float64Array(nFeatures);
    for (let i = 0; i < nFeatures; i++) {
      x[i] = mean[i] ?? 0;
      for (let j = 0; j <= i; j++) x[i] += (L[i]![j] ?? 0) * (z[j] ?? 0);
    }
    X.push(x);
  }
  return { X, mean, cov };
}

export interface CheckerboardDataset {
  X: Float64Array[];
  y: Int32Array;
  nSquares: number;
}

export function makeCheckerboard(
  nSamples = 200,
  nSquares = 4
): CheckerboardDataset {
  const X: Float64Array[] = [];
  const y = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const x0 = Math.random();
    const x1 = Math.random();
    X.push(new Float64Array([x0, x1]));
    const sq0 = Math.floor(x0 * nSquares);
    const sq1 = Math.floor(x1 * nSquares);
    y[i] = (sq0 + sq1) % 2;
  }
  return { X, y, nSquares };
}

export interface SCurveDataset {
  X: Float64Array[];
  t: Float64Array;
}

export function makeS_curve(nSamples = 100, noise = 0.0): SCurveDataset {
  const t = new Float64Array(nSamples);
  const X: Float64Array[] = [];
  for (let i = 0; i < nSamples; i++) {
    t[i] = 1.5 * Math.PI * (1 + 2 * Math.random());
    const ti = t[i] ?? 0;
    const x = Math.sin(ti) + (noise > 0 ? (Math.random() - 0.5) * noise : 0);
    const y = Math.sign(ti - Math.PI) * (Math.cos(ti) - 1) + (noise > 0 ? (Math.random() - 0.5) * noise : 0);
    const z = 2 * Math.random() + (noise > 0 ? (Math.random() - 0.5) * noise : 0);
    X.push(new Float64Array([x, y, z]));
  }
  return { X, t };
}

export function makeLowRankMatrix(
  nSamples = 100,
  nFeatures = 50,
  effectiveRank = 10,
  tailStrength = 0.5
): Float64Array[] {
  const n = Math.min(nSamples, nFeatures);
  const singularVals = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const low = Math.exp(-i / effectiveRank);
    const high = tailStrength * Math.exp(-i / (n * tailStrength + 1e-10));
    singularVals[i] = (1 - tailStrength) * low + high;
  }
  // Random orthogonal matrices via Gram-Schmidt
  const makeOrthogonal = (rows: number, cols: number): Float64Array[] => {
    const mat: Float64Array[] = Array.from({ length: rows }, () => {
      const row = new Float64Array(cols);
      for (let j = 0; j < cols; j++) row[j] = Math.random() - 0.5;
      return row;
    });
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < j; k++) {
        let dot = 0;
        for (let i = 0; i < rows; i++) dot += (mat[i]![j] ?? 0) * (mat[i]![k] ?? 0);
        for (let i = 0; i < rows; i++) mat[i]![j] = (mat[i]![j] ?? 0) - dot * (mat[i]![k] ?? 0);
      }
      let norm = 0;
      for (let i = 0; i < rows; i++) norm += (mat[i]![j] ?? 0) ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < rows; i++) mat[i]![j] = (mat[i]![j] ?? 0) / norm;
    }
    return mat;
  };
  const U = makeOrthogonal(nSamples, n);
  const V = makeOrthogonal(nFeatures, n);
  return Array.from({ length: nSamples }, (_, i) => {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      for (let k = 0; k < n; k++) row[j] += (U[i]![k] ?? 0) * (singularVals[k] ?? 0) * (V[j]![k] ?? 0);
    }
    return row;
  });
}
