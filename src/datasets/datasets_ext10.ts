/**
 * Synthetic datasets: make_checkerboard, make_classification extensions, make_low_rank_matrix.
 */

export function makeCheckerboard(shape: [number, number] = [100, 100], nClusters = [4, 3], noise = 0.5, shuffle = true): { data: Float64Array[]; rows: Int32Array; cols: Int32Array } {
  const [nRows, nCols] = shape;
  const [nRowClusters, nColClusters] = nClusters;
  const data: Float64Array[] = [];
  const rowLabels = new Int32Array(nRows);
  const colLabels = new Int32Array(nCols);

  for (let i = 0; i < nRows; i++) rowLabels[i] = i % nRowClusters;
  for (let j = 0; j < nCols; j++) colLabels[j] = j % nColClusters;

  for (let i = 0; i < nRows; i++) {
    const row = new Float64Array(nCols).map((_, j) => {
      const baseVal = (rowLabels[i]! + colLabels[j]!) % 2 === 0 ? 1 : 0;
      return baseVal + noise * (Math.random() - 0.5);
    });
    data.push(row);
  }

  if (shuffle) {
    const perm = Array.from({ length: nRows }, (_, i) => i).sort(() => Math.random() - 0.5);
    const shuffled = perm.map(i => data[i]!);
    const shuffledRows = new Int32Array(perm.map(i => rowLabels[i]!));
    return { data: shuffled, rows: shuffledRows, cols: colLabels };
  }
  return { data, rows: rowLabels, cols: colLabels };
}

export function makeLowRankMatrix(nSamples = 100, nFeatures = 50, effectiveRank = 10, tailStrength = 0.5): Float64Array[] {
  // Generate random matrix with specified effective rank
  const nV = Math.min(nSamples, nFeatures);
  // Create U matrix (nSamples x nV)
  const U: Float64Array[] = [];
  for (let k = 0; k < nV; k++) {
    const col = new Float64Array(nSamples).map(() => Math.random() - 0.5);
    // Orthogonalize against previous columns
    for (let prev = 0; prev < k; prev++) {
      const dot = col.reduce((s, v, i) => s + v * (U[prev]![i] ?? 0), 0);
      for (let i = 0; i < nSamples; i++) col[i] = (col[i] ?? 0) - dot * (U[prev]![i] ?? 0);
    }
    const norm = Math.sqrt(col.reduce((s, v) => s + v * v, 0));
    U.push(new Float64Array(col.map(v => v / (norm + 1e-10))));
  }
  // Singular values
  const sigma = new Float64Array(nV).map((_, k) => k < effectiveRank
    ? (1 - tailStrength) * Math.exp(-k / effectiveRank) + tailStrength
    : tailStrength * Math.exp(-(k - effectiveRank) / nV)
  );
  // V matrix (nFeatures x nV)
  const V: Float64Array[] = [];
  for (let k = 0; k < nV; k++) {
    const col = new Float64Array(nFeatures).map(() => Math.random() - 0.5);
    for (let prev = 0; prev < k; prev++) {
      const dot = col.reduce((s, v, i) => s + v * (V[prev]![i] ?? 0), 0);
      for (let i = 0; i < nFeatures; i++) col[i] = (col[i] ?? 0) - dot * (V[prev]![i] ?? 0);
    }
    const norm = Math.sqrt(col.reduce((s, v) => s + v * v, 0));
    V.push(new Float64Array(col.map(v => v / (norm + 1e-10))));
  }
  // X = U * diag(sigma) * V^T
  return Array.from({ length: nSamples }, (_, i) =>
    new Float64Array(nFeatures).map((_, j) =>
      U.reduce((s, u, k) => s + (u[i] ?? 0) * (sigma[k] ?? 0) * (V[k]![j] ?? 0), 0)
    )
  );
}

export function makeSpd(nDim = 3): Float64Array[] {
  const A = Array.from({ length: nDim }, () => new Float64Array(nDim).map(() => Math.random()));
  return Array.from({ length: nDim }, (_, i) =>
    new Float64Array(nDim).map((_, j) => A[i]!.reduce((s, v, k) => s + v * (A[j]![k] ?? 0), 0) / nDim + (i === j ? nDim : 0))
  );
}

export function makeDense(nSamples = 100, nFeatures = 10, density = 0.5): Float64Array[] {
  return Array.from({ length: nSamples }, () =>
    new Float64Array(nFeatures).map(() => Math.random() < density ? Math.random() - 0.5 : 0)
  );
}

export function makeSparse(nSamples = 100, nFeatures = 100, density = 0.1): Float64Array[] {
  return makeDense(nSamples, nFeatures, density);
}
