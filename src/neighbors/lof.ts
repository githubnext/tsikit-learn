/**
 * Local Outlier Factor (LOF): density-based outlier detection.
 * Mirrors sklearn.neighbors.LocalOutlierFactor.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

/** k nearest neighbours indices and distances for a single query. */
function knnQuery(
  query: Float64Array,
  points: Float64Array[],
  k: number,
  excludeSelf = false,
): { indices: number[]; distances: number[] } {
  const dists = points.map((p, i) => ({ i, d: euclidean(query, p) }));
  dists.sort((a, b) => a.d - b.d);
  const start = excludeSelf ? 1 : 0;
  const nbrs = dists.slice(start, start + k);
  return {
    indices: nbrs.map((x) => x.i),
    distances: nbrs.map((x) => x.d),
  };
}

/**
 * Local Outlier Factor.
 * Mirrors sklearn.neighbors.LocalOutlierFactor.
 */
export class LocalOutlierFactor {
  nNeighbors: number;
  algorithm: "auto";
  contamination: number | "auto";
  novelty: boolean;
  metric: "euclidean";

  fitX_: Float64Array[] | null = null;
  negativeLofScores_: Float64Array | null = null;
  threshold_: number = -1.5;
  offset_: number = -1.5;

  constructor(
    options: {
      nNeighbors?: number;
      contamination?: number | "auto";
      novelty?: boolean;
    } = {},
  ) {
    this.nNeighbors = options.nNeighbors ?? 20;
    this.algorithm = "auto";
    this.contamination = options.contamination ?? "auto";
    this.novelty = options.novelty ?? false;
    this.metric = "euclidean";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const k = Math.min(this.nNeighbors, n - 1);
    this.fitX_ = X;

    // Compute k-distance and k-neighbors for all training points
    const kDistances = new Float64Array(n);
    const kNbrIndices: number[][] = [];

    for (let i = 0; i < n; i++) {
      const { indices, distances } = knnQuery(
        X[i] ?? new Float64Array(0),
        X,
        k + 1,
        true,
      );
      kNbrIndices.push(indices);
      kDistances[i] = distances[k - 1] ?? 0;
    }

    // Compute local reachability density (lrd)
    const lrd = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const nbrs = kNbrIndices[i] ?? [];
      let reachSum = 0;
      for (const j of nbrs) {
        const dist = euclidean(
          X[i] ?? new Float64Array(0),
          X[j] ?? new Float64Array(0),
        );
        reachSum += Math.max(kDistances[j] ?? 0, dist);
      }
      lrd[i] = nbrs.length > 0 ? nbrs.length / Math.max(reachSum, 1e-10) : 1;
    }

    // Compute LOF scores
    const lof = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const nbrs = kNbrIndices[i] ?? [];
      let lrdRatioSum = 0;
      for (const j of nbrs) {
        lrdRatioSum += (lrd[j] ?? 1) / Math.max(lrd[i] ?? 1, 1e-10);
      }
      lof[i] = nbrs.length > 0 ? lrdRatioSum / nbrs.length : 1;
    }

    this.negativeLofScores_ = new Float64Array(lof.map((v) => -v));

    if (this.contamination === "auto") {
      this.offset_ = -1.5;
    } else {
      const sorted = Array.from(this.negativeLofScores_).sort((a, b) => a - b);
      const idx = Math.floor((this.contamination as number) * n);
      this.offset_ = sorted[Math.min(idx, n - 1)] ?? -1.5;
    }
    this.threshold_ = this.offset_;
    return this;
  }

  /** Score samples: negative LOF (higher = more normal). */
  scoresSamples(X: Float64Array[]): Float64Array {
    if (this.fitX_ === null) throw new NotFittedError("LocalOutlierFactor");
    const trainX = this.fitX_;
    const n = trainX.length;
    const k = Math.min(this.nNeighbors, n - 1);

    // Pre-compute training k-distances
    const kDistancesTrain = new Float64Array(n);
    const kNbrIndicesTrain: number[][] = [];
    const lrdTrain = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const { indices, distances } = knnQuery(
        trainX[i] ?? new Float64Array(0),
        trainX,
        k + 1,
        true,
      );
      kNbrIndicesTrain.push(indices);
      kDistancesTrain[i] = distances[k - 1] ?? 0;
    }
    for (let i = 0; i < n; i++) {
      const nbrs = kNbrIndicesTrain[i] ?? [];
      let reachSum = 0;
      for (const j of nbrs) {
        const dist = euclidean(
          trainX[i] ?? new Float64Array(0),
          trainX[j] ?? new Float64Array(0),
        );
        reachSum += Math.max(kDistancesTrain[j] ?? 0, dist);
      }
      lrdTrain[i] =
        nbrs.length > 0 ? nbrs.length / Math.max(reachSum, 1e-10) : 1;
    }

    const scores = new Float64Array(X.length);
    for (let qi = 0; qi < X.length; qi++) {
      const { indices, distances } = knnQuery(
        X[qi] ?? new Float64Array(0),
        trainX,
        k,
        false,
      );
      let reachSum = 0;
      for (let ni = 0; ni < indices.length; ni++) {
        const j = indices[ni] ?? 0;
        reachSum += Math.max(kDistancesTrain[j] ?? 0, distances[ni] ?? 0);
      }
      const lrdQuery =
        indices.length > 0 ? indices.length / Math.max(reachSum, 1e-10) : 1;
      let lrdRatioSum = 0;
      for (const j of indices)
        lrdRatioSum += (lrdTrain[j] ?? 1) / Math.max(lrdQuery, 1e-10);
      const lof = indices.length > 0 ? lrdRatioSum / indices.length : 1;
      scores[qi] = -lof;
    }
    return scores;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    const scores = this.scoresSamples(X);
    return new Float64Array(scores.map((s) => s - this.offset_));
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.novelty) {
      // In non-novelty mode, return training scores
      if (this.negativeLofScores_ === null)
        throw new NotFittedError("LocalOutlierFactor");
      return new Int32Array(
        this.negativeLofScores_.map((s) => (s >= this.offset_ ? 1 : -1)),
      );
    }
    const scores = this.decisionFunction(X);
    return new Int32Array(scores.map((s) => (s >= 0 ? 1 : -1)));
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    if (this.negativeLofScores_ === null)
      throw new NotFittedError("LocalOutlierFactor");
    return new Int32Array(
      this.negativeLofScores_.map((s) => (s >= this.offset_ ? 1 : -1)),
    );
  }
}
