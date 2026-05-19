/**
 * Cluster validation utilities: elbow method, gap statistic, Davies-Bouldin.
 * Extends sklearn.cluster with additional validation tools.
 */

import type { KMeans } from "./kmeans.js";

/**
 * Elbow method: run KMeans for multiple k values and find the elbow.
 */
export interface ElbowResult {
  kValues: number[];
  inertias: number[];
  optimalK: number;
}

export function elbowMethod(
  X: Float64Array[],
  kRange: number[] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
  KMeansClass: new (opts: { nClusters: number; randomState?: number }) => {
    fit(X: Float64Array[]): unknown;
    inertia_: number;
  },
  randomState?: number
): ElbowResult {
  const inertias: number[] = [];
  for (const k of kRange) {
    const km = new KMeansClass({ nClusters: k, randomState });
    km.fit(X);
    inertias.push(km.inertia_);
  }

  // Find elbow using maximum curvature (second derivative)
  let optimalK = kRange[0] ?? 2;
  if (inertias.length >= 3) {
    let maxCurvature = -Infinity;
    for (let i = 1; i < inertias.length - 1; i++) {
      const d1 = (inertias[i - 1] ?? 0) - (inertias[i] ?? 0);
      const d2 = (inertias[i] ?? 0) - (inertias[i + 1] ?? 0);
      const curvature = d1 - d2;
      if (curvature > maxCurvature) {
        maxCurvature = curvature;
        optimalK = kRange[i] ?? 2;
      }
    }
  }

  return { kValues: kRange, inertias, optimalK };
}

/**
 * Gap statistic: compare inertia to reference (uniform) distribution.
 */
export interface GapStatisticResult {
  kValues: number[];
  gaps: number[];
  sks: number[];
  optimalK: number;
}

export function gapStatistic(
  X: Float64Array[],
  kRange: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  KMeansClass: new (opts: { nClusters: number; randomState?: number }) => {
    fit(X: Float64Array[]): unknown;
    inertia_: number;
  },
  nRefs = 10,
  randomState = 42
): GapStatisticResult {
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;

  // Compute bounding box of data
  const mins = new Float64Array(nFeatures);
  const maxs = new Float64Array(nFeatures);
  mins.fill(Infinity);
  maxs.fill(-Infinity);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) {
      const v = row[j] ?? 0;
      if (v < (mins[j] ?? Infinity)) mins[j] = v;
      if (v > (maxs[j] ?? -Infinity)) maxs[j] = v;
    }
  }

  // Seeded simple LCG RNG
  let seed = randomState;
  function randFloat(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const gaps: number[] = [];
  const sks: number[] = [];

  for (const k of kRange) {
    const km = new KMeansClass({ nClusters: k, randomState });
    km.fit(X);
    const logW = Math.log(km.inertia_ + 1e-10);

    // Reference distribution
    const refLogWs: number[] = [];
    for (let r = 0; r < nRefs; r++) {
      const Xref: Float64Array[] = [];
      for (let i = 0; i < nSamples; i++) {
        const row = new Float64Array(nFeatures);
        for (let j = 0; j < nFeatures; j++) {
          row[j] = (mins[j] ?? 0) + randFloat() * ((maxs[j] ?? 1) - (mins[j] ?? 0));
        }
        Xref.push(row);
      }
      const kmRef = new KMeansClass({ nClusters: k, randomState: r });
      kmRef.fit(Xref);
      refLogWs.push(Math.log(kmRef.inertia_ + 1e-10));
    }

    const meanRefLogW = refLogWs.reduce((s, v) => s + v, 0) / nRefs;
    const variance = refLogWs.reduce((s, v) => s + (v - meanRefLogW) ** 2, 0) / nRefs;
    const sd = Math.sqrt(variance);
    const sk = sd * Math.sqrt(1 + 1 / nRefs);

    gaps.push(meanRefLogW - logW);
    sks.push(sk);
  }

  // Optimal k: smallest k such that gap(k) >= gap(k+1) - sk+1
  let optimalK = kRange[0] ?? 1;
  for (let i = 0; i < kRange.length - 1; i++) {
    if ((gaps[i] ?? 0) >= (gaps[i + 1] ?? 0) - (sks[i + 1] ?? 0)) {
      optimalK = kRange[i] ?? 1;
      break;
    }
  }

  return { kValues: kRange, gaps, sks, optimalK };
}

/**
 * Davies-Bouldin Index (lower is better).
 * Complements silhouette score for cluster validation.
 */
export function daviesBouldinScore(X: Float64Array[], labels: Int32Array): number {
  const uniqueLabels = Array.from(new Set(Array.from(labels))).sort((a, b) => a - b);
  const k = uniqueLabels.length;
  if (k < 2) return 0;

  const nFeatures = X[0]?.length ?? 0;

  // Compute centroids
  const centroids: Float64Array[] = [];
  const counts: number[] = [];
  const labelToIdx = new Map<number, number>();
  uniqueLabels.forEach((l, i) => labelToIdx.set(l, i));

  for (let ci = 0; ci < k; ci++) {
    centroids.push(new Float64Array(nFeatures));
    counts.push(0);
  }

  for (let i = 0; i < X.length; i++) {
    const ci = labelToIdx.get(labels[i] ?? 0) ?? 0;
    counts[ci] = (counts[ci] ?? 0) + 1;
    for (let j = 0; j < nFeatures; j++) {
      centroids[ci]![j] = (centroids[ci]![j] ?? 0) + (X[i]?.[j] ?? 0);
    }
  }
  for (let ci = 0; ci < k; ci++) {
    for (let j = 0; j < nFeatures; j++) {
      centroids[ci]![j] = (centroids[ci]![j] ?? 0) / (counts[ci] ?? 1);
    }
  }

  // Compute scatter (avg distance of cluster points to centroid)
  const scatter: number[] = new Array(k).fill(0);
  const memberCounts = new Array(k).fill(0);
  for (let i = 0; i < X.length; i++) {
    const ci = labelToIdx.get(labels[i] ?? 0) ?? 0;
    let dist = 0;
    for (let j = 0; j < nFeatures; j++) {
      dist += ((X[i]?.[j] ?? 0) - (centroids[ci]?.[j] ?? 0)) ** 2;
    }
    scatter[ci] = (scatter[ci] ?? 0) + Math.sqrt(dist);
    memberCounts[ci] = (memberCounts[ci] ?? 0) + 1;
  }
  for (let ci = 0; ci < k; ci++) {
    scatter[ci] = (scatter[ci] ?? 0) / (memberCounts[ci] || 1);
  }

  // Compute Davies-Bouldin index
  let dbSum = 0;
  for (let i = 0; i < k; i++) {
    let maxR = -Infinity;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      let distCentroids = 0;
      for (let f = 0; f < nFeatures; f++) {
        distCentroids += ((centroids[i]?.[f] ?? 0) - (centroids[j]?.[f] ?? 0)) ** 2;
      }
      distCentroids = Math.sqrt(distCentroids);
      const R = ((scatter[i] ?? 0) + (scatter[j] ?? 0)) / (distCentroids || 1e-10);
      if (R > maxR) maxR = R;
    }
    dbSum += maxR;
  }

  return dbSum / k;
}

/**
 * Calinski-Harabasz Index (higher is better).
 */
export function calinskiHarabaszScore(X: Float64Array[], labels: Int32Array): number {
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const uniqueLabels = Array.from(new Set(Array.from(labels))).sort((a, b) => a - b);
  const k = uniqueLabels.length;
  if (k < 2 || nSamples <= k) return 0;

  const labelToIdx = new Map<number, number>();
  uniqueLabels.forEach((l, i) => labelToIdx.set(l, i));

  // Global centroid
  const globalCentroid = new Float64Array(nFeatures);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) globalCentroid[j] = (globalCentroid[j] ?? 0) + (row[j] ?? 0);
  }
  for (let j = 0; j < nFeatures; j++) globalCentroid[j] = (globalCentroid[j] ?? 0) / nSamples;

  // Cluster centroids and counts
  const centroids = Array.from({ length: k }, () => new Float64Array(nFeatures));
  const counts = new Array(k).fill(0);
  for (let i = 0; i < nSamples; i++) {
    const ci = labelToIdx.get(labels[i] ?? 0) ?? 0;
    counts[ci] = (counts[ci] ?? 0) + 1;
    for (let j = 0; j < nFeatures; j++) {
      centroids[ci]![j] = (centroids[ci]![j] ?? 0) + (X[i]?.[j] ?? 0);
    }
  }
  for (let ci = 0; ci < k; ci++) {
    for (let j = 0; j < nFeatures; j++) {
      centroids[ci]![j] = (centroids[ci]![j] ?? 0) / (counts[ci] ?? 1);
    }
  }

  // Between-cluster scatter (BGSS)
  let bgss = 0;
  for (let ci = 0; ci < k; ci++) {
    let d = 0;
    for (let j = 0; j < nFeatures; j++) {
      d += ((centroids[ci]?.[j] ?? 0) - (globalCentroid[j] ?? 0)) ** 2;
    }
    bgss += (counts[ci] ?? 0) * d;
  }

  // Within-cluster scatter (WGSS)
  let wgss = 0;
  for (let i = 0; i < nSamples; i++) {
    const ci = labelToIdx.get(labels[i] ?? 0) ?? 0;
    let d = 0;
    for (let j = 0; j < nFeatures; j++) {
      d += ((X[i]?.[j] ?? 0) - (centroids[ci]?.[j] ?? 0)) ** 2;
    }
    wgss += d;
  }

  return (bgss / (k - 1)) / ((wgss / (nSamples - k)) || 1e-10);
}
