/**
 * Cluster utility functions.
 * Mirrors sklearn.cluster._mean_shift and related utilities.
 */

/**
 * Estimate the bandwidth for Mean Shift algorithm.
 * Uses a ball-tree-like approach: for each sample, counts how many
 * samples are within the estimated bandwidth.
 *
 * @param X - Input data (n_samples x n_features)
 * @param quantile - Quantile of pairwise distances to use as bandwidth (default 0.3)
 * @param nSamples - Number of samples to use for estimation (default: all)
 * @param seed - Random seed for subsampling
 */
export function estimateBandwidth(
  X: Float64Array[],
  options: {
    quantile?: number;
    nSamples?: number;
    seed?: number;
  } = {},
): number {
  const { quantile = 0.3, seed = 0 } = options;
  const n = X.length;
  let nSamples = options.nSamples ?? n;
  nSamples = Math.min(nSamples, n);

  // Subsample if needed
  let indices: number[];
  if (nSamples < n) {
    let rng = seed;
    const rand = () => {
      rng = (rng * 1664525 + 1013904223) & 0xffffffff;
      return (rng >>> 0) / 0xffffffff;
    };
    indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = indices[i]!; indices[i] = indices[j]!; indices[j] = tmp;
    }
    indices = indices.slice(0, nSamples);
  } else {
    indices = Array.from({ length: n }, (_, i) => i);
  }

  // Compute pairwise distances between sampled points and all points
  // Then take the quantile
  const allDists: number[] = [];
  for (const idx of indices) {
    const xi = X[idx]!;
    for (let j = 0; j < n; j++) {
      const xj = X[j]!;
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) {
        d2 += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
      }
      allDists.push(Math.sqrt(d2));
    }
  }

  allDists.sort((a, b) => a - b);
  const qIdx = Math.floor(quantile * (allDists.length - 1));
  return allDists[qIdx] ?? 1.0;
}

/**
 * Find initial seed points for Mean Shift.
 * Seeds are bin centers of a uniform grid at bandwidth resolution.
 *
 * @param X - Input data
 * @param bandwidth - Bin size
 * @param minBinFreq - Minimum number of points per bin to be included
 */
export function getBinSeeds(
  X: Float64Array[],
  bandwidth: number,
  minBinFreq = 1,
): Float64Array[] {
  if (bandwidth <= 0) throw new Error("bandwidth must be positive");
  const n = X.length;
  const d = X[0]?.length ?? 0;

  // Discretize X into bins
  const binMap = new Map<string, { sum: Float64Array; count: number }>();

  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    const binCoords: number[] = [];
    for (let k = 0; k < d; k++) {
      binCoords.push(Math.round((xi[k] ?? 0) / bandwidth));
    }
    const key = binCoords.join(",");
    const existing = binMap.get(key);
    if (existing) {
      for (let k = 0; k < d; k++) {
        existing.sum[k]! += xi[k] ?? 0;
      }
      existing.count++;
    } else {
      const sum = new Float64Array(d);
      for (let k = 0; k < d; k++) sum[k] = xi[k] ?? 0;
      binMap.set(key, { sum, count: 1 });
    }
  }

  // Return bin centers with sufficient frequency
  const seeds: Float64Array[] = [];
  for (const { sum, count } of binMap.values()) {
    if (count >= minBinFreq) {
      const center = new Float64Array(d);
      for (let k = 0; k < d; k++) center[k] = (sum[k] ?? 0) / count;
      seeds.push(center);
    }
  }

  return seeds;
}

/**
 * Find which bin each point belongs to.
 * @returns Int32Array of bin indices (one per sample)
 */
export function assignBins(
  X: Float64Array[],
  seeds: Float64Array[],
): Int32Array {
  const n = X.length;
  const result = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    let bestDist = Number.POSITIVE_INFINITY;
    let bestJ = -1;
    for (let j = 0; j < seeds.length; j++) {
      const seed = seeds[j]!;
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) {
        d2 += ((xi[k] ?? 0) - (seed[k] ?? 0)) ** 2;
      }
      if (d2 < bestDist) { bestDist = d2; bestJ = j; }
    }
    result[i] = bestJ;
  }
  return result;
}

/**
 * Single iteration of mean-shift update for a set of seeds.
 * Updates each seed to the mean of all points within bandwidth distance.
 *
 * @returns New seed positions and whether any seed moved more than tol
 */
export function meanShiftStep(
  X: Float64Array[],
  seeds: Float64Array[],
  bandwidth: number,
): { newSeeds: Float64Array[]; converged: boolean } {
  const d = X[0]?.length ?? 0;
  const bw2 = bandwidth * bandwidth;
  const newSeeds: Float64Array[] = [];
  let maxShift = 0;

  for (const seed of seeds) {
    const newSeed = new Float64Array(d);
    let weight = 0;
    for (const xi of X) {
      let d2 = 0;
      for (let k = 0; k < d; k++) {
        d2 += ((xi[k] ?? 0) - (seed[k] ?? 0)) ** 2;
      }
      if (d2 <= bw2) {
        weight++;
        for (let k = 0; k < d; k++) newSeed[k]! += xi[k] ?? 0;
      }
    }
    if (weight > 0) {
      for (let k = 0; k < d; k++) newSeed[k]! /= weight;
    } else {
      newSeed.set(seed);
    }

    // Track max shift
    let shift2 = 0;
    for (let k = 0; k < d; k++) {
      shift2 += ((newSeed[k] ?? 0) - (seed[k] ?? 0)) ** 2;
    }
    maxShift = Math.max(maxShift, Math.sqrt(shift2));
    newSeeds.push(newSeed);
  }

  return { newSeeds, converged: maxShift < 1e-3 * bandwidth };
}

/**
 * Merge nearby seeds by deduplication within bandwidth distance.
 * Returns unique cluster centers.
 */
export function mergeSeeds(
  seeds: Float64Array[],
  bandwidth: number,
): Float64Array[] {
  const bw2 = bandwidth * bandwidth;
  const merged: Float64Array[] = [];

  for (const seed of seeds) {
    let isNew = true;
    for (const center of merged) {
      let d2 = 0;
      for (let k = 0; k < seed.length; k++) {
        d2 += ((seed[k] ?? 0) - (center[k] ?? 0)) ** 2;
      }
      if (d2 <= bw2) { isNew = false; break; }
    }
    if (isNew) merged.push(seed);
  }

  return merged;
}

/**
 * Compute cluster labels for X given cluster centers.
 * Each point is assigned to its nearest center.
 */
export function clusterLabels(
  X: Float64Array[],
  centers: Float64Array[],
): Int32Array {
  const labels = new Int32Array(X.length);
  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    let best = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    for (let j = 0; j < centers.length; j++) {
      const c = centers[j]!;
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) {
        d2 += ((xi[k] ?? 0) - (c[k] ?? 0)) ** 2;
      }
      if (d2 < bestDist) { bestDist = d2; best = j; }
    }
    labels[i] = best;
  }
  return labels;
}

/**
 * Compute inertia (within-cluster sum of squared distances to centers).
 */
export function computeInertia(
  X: Float64Array[],
  centers: Float64Array[],
  labels: Int32Array,
): number {
  let inertia = 0;
  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    const c = centers[labels[i]!]!;
    let d2 = 0;
    for (let k = 0; k < xi.length; k++) {
      d2 += ((xi[k] ?? 0) - (c[k] ?? 0)) ** 2;
    }
    inertia += d2;
  }
  return inertia;
}

/**
 * Compute cluster centers from assignments.
 */
export function computeCenters(
  X: Float64Array[],
  labels: Int32Array,
  nClusters: number,
): Float64Array[] {
  const d = X[0]?.length ?? 0;
  const sums: Float64Array[] = Array.from({ length: nClusters }, () => new Float64Array(d));
  const counts = new Int32Array(nClusters);

  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    const lbl = labels[i] ?? 0;
    if (lbl >= 0 && lbl < nClusters) {
      counts[lbl]!++;
      for (let k = 0; k < d; k++) sums[lbl]![k]! += xi[k] ?? 0;
    }
  }

  return sums.map((s, j) => {
    const cnt = counts[j] ?? 1;
    if (cnt === 0) return s;
    const c = new Float64Array(d);
    for (let k = 0; k < d; k++) c[k] = (s[k] ?? 0) / cnt;
    return c;
  });
}
