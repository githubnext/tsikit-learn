/**
 * Mean Shift clustering extensions.
 * Mirrors scikit-learn's cluster.MeanShift with bandwidth estimation.
 */

export interface MeanShiftExtOptions {
  bandwidth?: number;
  seeds?: Float64Array[];
  binSeeding?: boolean;
  minBinFreq?: number;
  clusterAll?: boolean;
  maxIter?: number;
}

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

/**
 * Estimate bandwidth for Mean Shift using median heuristic.
 */
export function estimateBandwidth(
  X: Float64Array[],
  options: { quantile?: number; nSamples?: number } = {},
): number {
  const { quantile = 0.3, nSamples } = options;
  const n = X.length;
  const sample = nSamples !== undefined ? X.slice(0, nSamples) : X;
  const nS = sample.length;
  const dists: number[] = [];

  for (let i = 0; i < nS; i++) {
    for (let j = i + 1; j < n; j++) {
      dists.push(euclidean(sample[i]!, X[j]!));
    }
  }
  dists.sort((a, b) => a - b);
  const idx = Math.floor(quantile * dists.length);
  return dists[idx] ?? 1;
}

export class MeanShiftExt {
  readonly bandwidth: number | null;
  readonly clusterAll: boolean;
  readonly maxIter: number;

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;

  constructor(options: MeanShiftExtOptions = {}) {
    this.bandwidth = options.bandwidth ?? null;
    this.clusterAll = options.clusterAll ?? true;
    this.maxIter = options.maxIter ?? 300;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const bw = this.bandwidth ?? estimateBandwidth(X);

    // Initialize seeds at data points
    let seeds = X.map((row) => row.slice() as Float64Array);

    // Iterate mean shift
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxShift = 0;
      const newSeeds = seeds.map((seed) => {
        const weights: number[] = X.map((xi) => {
          const d = euclidean(xi, seed);
          return Math.exp(-0.5 * (d / bw) ** 2);
        });
        const totalW = weights.reduce((s, w) => s + w, 0);
        if (totalW < 1e-10) return seed;
        const newSeed = new Float64Array(nFeatures);
        for (let j = 0; j < nFeatures; j++) {
          newSeed[j] = X.reduce((s, xi, i) => s + (weights[i] ?? 0) * (xi[j] ?? 0), 0) / totalW;
        }
        maxShift = Math.max(maxShift, euclidean(newSeed, seed));
        return newSeed;
      });
      seeds = newSeeds;
      if (maxShift < 1e-5) break;
    }

    // Merge nearby seeds into cluster centers
    const centers: Float64Array[] = [];
    for (const seed of seeds) {
      let merged = false;
      for (const center of centers) {
        if (euclidean(seed, center) < bw / 2) {
          merged = true;
          // Update center as mean
          for (let j = 0; j < nFeatures; j++) {
            center[j] = ((center[j] ?? 0) + (seed[j] ?? 0)) / 2;
          }
          break;
        }
      }
      if (!merged) centers.push(seed.slice() as Float64Array);
    }

    this.clusterCenters_ = centers;
    this.labels_ = Int32Array.from({ length: n }, (_, i) => {
      let bestC = -1;
      let bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < centers.length; c++) {
        const d = euclidean(X[i]!, centers[c]!);
        if (d < bestD) { bestD = d; bestC = c; }
      }
      if (!this.clusterAll && bestD > bw) return -1;
      return bestC;
    });

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this.clusterCenters_ === null) throw new Error("MeanShiftExt must be fitted first");
    const centers = this.clusterCenters_;
    return Int32Array.from(X, (xi) => {
      let best = 0;
      let bestD = euclidean(xi, centers[0]!);
      for (let c = 1; c < centers.length; c++) {
        const d = euclidean(xi, centers[c]!);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    });
  }
}
