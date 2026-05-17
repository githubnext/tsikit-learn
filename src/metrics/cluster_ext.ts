/**
 * Additional clustering evaluation metrics.
 * Mirrors sklearn.metrics: davies_bouldin_score, calinski_harabasz_score,
 * v_measure_score, mutual_info_score, normalized_mutual_info_score,
 * adjusted_mutual_info_score, fowlkes_mallows_score, completeness_score.
 */

/**
 * Davies-Bouldin index clustering evaluation.
 * Mirrors sklearn.metrics.davies_bouldin_score.
 * Lower is better.
 */
export function daviesBouldinScore(
  X: Float64Array[],
  labels: Int32Array
): number {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const uniqueLabels = [...new Set(Array.from(labels))].sort((a, b) => a - b);
  const k = uniqueLabels.length;
  if (k < 2) return 0;

  // Compute centroids
  const centroids: Map<number, Float64Array> = new Map();
  const counts: Map<number, number> = new Map();
  for (const lbl of uniqueLabels) {
    centroids.set(lbl, new Float64Array(p));
    counts.set(lbl, 0);
  }
  for (let i = 0; i < n; i++) {
    const lbl = labels[i] ?? 0;
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const xi = X[i] ?? new Float64Array(p);
    for (let j = 0; j < p; j++) c[j]! += xi[j] ?? 0;
    counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
  }
  for (const lbl of uniqueLabels) {
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const cnt = counts.get(lbl) ?? 1;
    for (let j = 0; j < p; j++) c[j]! /= cnt;
  }

  // Compute average intra-cluster distances
  const scatter: Map<number, number> = new Map();
  for (const lbl of uniqueLabels) scatter.set(lbl, 0);
  for (let i = 0; i < n; i++) {
    const lbl = labels[i] ?? 0;
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const xi = X[i] ?? new Float64Array(p);
    let dist = 0;
    for (let j = 0; j < p; j++) dist += ((xi[j] ?? 0) - (c[j] ?? 0)) ** 2;
    scatter.set(lbl, (scatter.get(lbl) ?? 0) + Math.sqrt(dist));
  }
  for (const lbl of uniqueLabels) {
    scatter.set(lbl, (scatter.get(lbl) ?? 0) / (counts.get(lbl) ?? 1));
  }

  // Compute DB index
  let dbSum = 0;
  for (const li of uniqueLabels) {
    let maxR = 0;
    const ci = centroids.get(li) ?? new Float64Array(p);
    const si = scatter.get(li) ?? 0;
    for (const lj of uniqueLabels) {
      if (li === lj) continue;
      const cj = centroids.get(lj) ?? new Float64Array(p);
      let dist = 0;
      for (let j = 0; j < p; j++) dist += ((ci[j] ?? 0) - (cj[j] ?? 0)) ** 2;
      const dij = Math.sqrt(dist);
      const r = dij > 0 ? ((si + (scatter.get(lj) ?? 0)) / dij) : 0;
      if (r > maxR) maxR = r;
    }
    dbSum += maxR;
  }
  return dbSum / k;
}

/**
 * Calinski-Harabasz index (Variance Ratio Criterion).
 * Mirrors sklearn.metrics.calinski_harabasz_score.
 * Higher is better.
 */
export function calinskiHarabaszScore(
  X: Float64Array[],
  labels: Int32Array
): number {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const uniqueLabels = [...new Set(Array.from(labels))].sort((a, b) => a - b);
  const k = uniqueLabels.length;
  if (k < 2) return 0;

  // Global centroid
  const globalCentroid = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(p);
    for (let j = 0; j < p; j++) globalCentroid[j]! += xi[j] ?? 0;
  }
  for (let j = 0; j < p; j++) globalCentroid[j]! /= n;

  // Cluster centroids
  const centroids: Map<number, Float64Array> = new Map();
  const counts: Map<number, number> = new Map();
  for (const lbl of uniqueLabels) {
    centroids.set(lbl, new Float64Array(p));
    counts.set(lbl, 0);
  }
  for (let i = 0; i < n; i++) {
    const lbl = labels[i] ?? 0;
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const xi = X[i] ?? new Float64Array(p);
    for (let j = 0; j < p; j++) c[j]! += xi[j] ?? 0;
    counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
  }
  for (const lbl of uniqueLabels) {
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const cnt = counts.get(lbl) ?? 1;
    for (let j = 0; j < p; j++) c[j]! /= cnt;
  }

  // Between-cluster dispersion (BGSS)
  let bgss = 0;
  for (const lbl of uniqueLabels) {
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const cnt = counts.get(lbl) ?? 0;
    let d = 0;
    for (let j = 0; j < p; j++) d += ((c[j] ?? 0) - (globalCentroid[j] ?? 0)) ** 2;
    bgss += cnt * d;
  }

  // Within-cluster dispersion (WGSS)
  let wgss = 0;
  for (let i = 0; i < n; i++) {
    const lbl = labels[i] ?? 0;
    const c = centroids.get(lbl) ?? new Float64Array(p);
    const xi = X[i] ?? new Float64Array(p);
    for (let j = 0; j < p; j++) wgss += ((xi[j] ?? 0) - (c[j] ?? 0)) ** 2;
  }

  if (wgss === 0) return 1;
  return (bgss / (k - 1)) / (wgss / (n - k));
}

/**
 * Mutual information between two label arrays.
 * Mirrors sklearn.metrics.mutual_info_score.
 */
export function mutualInfoScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array
): number {
  const n = labelsTrue.length;
  const counts: Map<string, number> = new Map();
  const trueCount: Map<number, number> = new Map();
  const predCount: Map<number, number> = new Map();

  for (let i = 0; i < n; i++) {
    const t = labelsTrue[i] ?? 0;
    const p = labelsPred[i] ?? 0;
    const key = `${t},${p}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
    trueCount.set(t, (trueCount.get(t) ?? 0) + 1);
    predCount.set(p, (predCount.get(p) ?? 0) + 1);
  }

  let mi = 0;
  for (const [key, nij] of counts) {
    const [t, p] = key.split(",").map(Number);
    const ni = trueCount.get(t ?? 0) ?? 0;
    const nj = predCount.get(p ?? 0) ?? 0;
    if (ni > 0 && nj > 0 && nij > 0) {
      mi += (nij / n) * Math.log((n * nij) / (ni * nj));
    }
  }
  return Math.max(mi, 0);
}

/**
 * Entropy of a label array.
 */
function entropy(labels: Int32Array): number {
  const n = labels.length;
  const counts: Map<number, number> = new Map();
  for (let i = 0; i < n; i++) counts.set(labels[i] ?? 0, (counts.get(labels[i] ?? 0) ?? 0) + 1);
  let h = 0;
  for (const cnt of counts.values()) {
    const p = cnt / n;
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/**
 * Normalized mutual information.
 * Mirrors sklearn.metrics.normalized_mutual_info_score.
 */
export function normalizedMutualInfoScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array,
  average: "arithmetic" | "geometric" | "min" | "max" = "arithmetic"
): number {
  const mi = mutualInfoScore(labelsTrue, labelsPred);
  const ht = entropy(labelsTrue);
  const hp = entropy(labelsPred);
  let denom: number;
  switch (average) {
    case "arithmetic":
      denom = (ht + hp) / 2;
      break;
    case "geometric":
      denom = Math.sqrt(ht * hp);
      break;
    case "min":
      denom = Math.min(ht, hp);
      break;
    case "max":
      denom = Math.max(ht, hp);
      break;
  }
  return denom === 0 ? 0 : mi / denom;
}

/**
 * V-measure: harmonic mean of homogeneity and completeness.
 * Mirrors sklearn.metrics.v_measure_score.
 */
export function vMeasureScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array,
  beta = 1.0
): number {
  const mi = mutualInfoScore(labelsTrue, labelsPred);
  const ht = entropy(labelsTrue);
  const hp = entropy(labelsPred);

  const h = ht > 0 ? mi / ht : 1;
  const c = hp > 0 ? mi / hp : 1;

  if (h + c === 0) return 0;
  return (1 + beta * beta) * (h * c) / (beta * beta * h + c);
}

/**
 * Completeness score.
 * Mirrors sklearn.metrics.completeness_score.
 */
export function completenessScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array
): number {
  const mi = mutualInfoScore(labelsTrue, labelsPred);
  const hp = entropy(labelsPred);
  return hp > 0 ? mi / hp : 1;
}

/**
 * Fowlkes-Mallows index.
 * Mirrors sklearn.metrics.fowlkes_mallows_score.
 */
export function fowlkesMallowsScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array
): number {
  const n = labelsTrue.length;
  let tp = 0;
  let fpPlusTp = 0;
  let fnPlusTp = 0;

  // Count pairs
  const pairCounts: Map<string, number> = new Map();
  const trueCounts: Map<number, number> = new Map();
  const predCounts: Map<number, number> = new Map();

  for (let i = 0; i < n; i++) {
    const t = labelsTrue[i] ?? 0;
    const p = labelsPred[i] ?? 0;
    const key = `${t},${p}`;
    pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
    trueCounts.set(t, (trueCounts.get(t) ?? 0) + 1);
    predCounts.set(p, (predCounts.get(p) ?? 0) + 1);
  }

  for (const nij of pairCounts.values()) {
    tp += nij * (nij - 1) / 2;
  }
  for (const ni of trueCounts.values()) {
    fnPlusTp += ni * (ni - 1) / 2;
  }
  for (const nj of predCounts.values()) {
    fpPlusTp += nj * (nj - 1) / 2;
  }

  return fnPlusTp === 0 || fpPlusTp === 0 ? 0 : tp / Math.sqrt(fnPlusTp * fpPlusTp);
}
