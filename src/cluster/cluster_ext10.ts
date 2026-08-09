/**
 * Cluster extensions: HDBSCAN extensions, cluster statistics, gap statistic.
 * Mirrors sklearn.cluster extensions.
 */

import { BaseEstimator } from "../base.js";

/** Compute silhouette score for clustering. */
export function silhouetteScoreExt(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const n = X.length;
  const scores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ci = labels[i] ?? -1;
    if (ci === -1) { scores[i] = 0; continue; }
    let aSum = 0, aCnt = 0;
    const bMap = new Map<number, { sum: number; cnt: number }>();
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const cj = labels[j] ?? -1;
      let dist = 0;
      const xi = X[i]!, xj = X[j]!;
      for (let k = 0; k < xi.length; k++) dist += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
      dist = Math.sqrt(dist);
      if (cj === ci) { aSum += dist; aCnt++; }
      else {
        if (!bMap.has(cj)) bMap.set(cj, { sum: 0, cnt: 0 });
        const e = bMap.get(cj)!;
        e.sum += dist; e.cnt++;
      }
    }
    const a = aCnt > 0 ? aSum / aCnt : 0;
    let b = Number.POSITIVE_INFINITY;
    for (const { sum, cnt } of bMap.values()) if (cnt > 0) b = Math.min(b, sum / cnt);
    if (!Number.isFinite(b)) b = 0;
    const denom = Math.max(a, b);
    scores[i] = denom === 0 ? 0 : (b - a) / denom;
  }
  let s = 0;
  for (let i = 0; i < n; i++) s += scores[i] ?? 0;
  return s / n;
}

/** Calinski-Harabasz index (variance ratio criterion). */
export function calinskiHarabaszScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const n = X.length;
  const nf = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(labels).filter((c) => c !== -1))];
  const k = classes.length;
  if (k <= 1) return 0;
  const overall = new Float64Array(nf);
  for (const xi of X) for (let j = 0; j < nf; j++) overall[j] = (overall[j] ?? 0) + (xi[j] ?? 0);
  for (let j = 0; j < nf; j++) overall[j] = (overall[j] ?? 0) / n;
  let bss = 0, wss = 0;
  for (const c of classes) {
    const members = X.filter((_, i) => (labels[i] ?? -1) === c);
    const nc = members.length;
    const cm = new Float64Array(nf);
    for (const xi of members) for (let j = 0; j < nf; j++) cm[j] = (cm[j] ?? 0) + (xi[j] ?? 0);
    for (let j = 0; j < nf; j++) {
      cm[j] = (cm[j] ?? 0) / nc;
      bss += nc * ((cm[j] ?? 0) - (overall[j] ?? 0)) ** 2;
    }
    for (const xi of members) for (let j = 0; j < nf; j++) wss += ((xi[j] ?? 0) - (cm[j] ?? 0)) ** 2;
  }
  return wss === 0 ? 0 : (bss / (k - 1)) / (wss / (n - k));
}

/** Davies-Bouldin index. */
export function daviesBouldinScore(
  X: Float64Array[],
  labels: Int32Array,
): number {
  const nf = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(labels).filter((c) => c !== -1))];
  const k = classes.length;
  if (k <= 1) return 0;
  const centroids = classes.map((c) => {
    const members = X.filter((_, i) => (labels[i] ?? -1) === c);
    const cm = new Float64Array(nf);
    for (const xi of members) for (let j = 0; j < nf; j++) cm[j] = (cm[j] ?? 0) + (xi[j] ?? 0);
    for (let j = 0; j < nf; j++) cm[j] = (cm[j] ?? 0) / members.length;
    return cm;
  });
  const si = classes.map((c, ci) => {
    const members = X.filter((_, i) => (labels[i] ?? -1) === c);
    let s = 0;
    const centroid = centroids[ci]!;
    for (const xi of members) {
      let d = 0;
      for (let j = 0; j < nf; j++) d += ((xi[j] ?? 0) - (centroid[j] ?? 0)) ** 2;
      s += Math.sqrt(d);
    }
    return members.length > 0 ? s / members.length : 0;
  });
  const dist = (a: Float64Array, b: Float64Array): number => {
    let d = 0;
    for (let j = 0; j < a.length; j++) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(d);
  };
  let db = 0;
  for (let i = 0; i < k; i++) {
    let maxR = 0;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const d = dist(centroids[i]!, centroids[j]!);
      const r = d > 0 ? ((si[i] ?? 0) + (si[j] ?? 0)) / d : 0;
      if (r > maxR) maxR = r;
    }
    db += maxR;
  }
  return db / k;
}

/** GapStatistic: estimate optimal number of clusters. */
export class GapStatistic extends BaseEstimator {
  n_clusters_: number = 0;
  gap_values_: Float64Array = new Float64Array(0);
  sk_: Float64Array = new Float64Array(0);

  fit(X: Float64Array[], maxK = 10, nRef = 10): this {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    const gaps = new Float64Array(maxK);
    const sks = new Float64Array(maxK);
    const mins = new Float64Array(nf), maxs = new Float64Array(nf);
    for (let j = 0; j < nf; j++) {
      let mn = Number.POSITIVE_INFINITY, mx = Number.NEGATIVE_INFINITY;
      for (const xi of X) { const v = xi[j] ?? 0; if (v < mn) mn = v; if (v > mx) mx = v; }
      mins[j] = mn; maxs[j] = mx;
    }
    for (let k = 1; k <= maxK; k++) {
      const Wk = this._kmeansWk(X, k);
      let refWkSum = 0, refWkSumSq = 0;
      for (let r = 0; r < nRef; r++) {
        const ref = Array.from({ length: n }, () => {
          const xi = new Float64Array(nf);
          for (let j = 0; j < nf; j++) xi[j] = (mins[j] ?? 0) + Math.random() * ((maxs[j] ?? 1) - (mins[j] ?? 0));
          return xi;
        });
        const w = Math.log(Math.max(this._kmeansWk(ref, k), 1e-10));
        refWkSum += w; refWkSumSq += w * w;
      }
      const logWk = Math.log(Math.max(Wk, 1e-10));
      const expLogWk = refWkSum / nRef;
      gaps[k - 1] = expLogWk - logWk;
      sks[k - 1] = Math.sqrt(Math.max(refWkSumSq / nRef - expLogWk ** 2, 0)) * Math.sqrt(1 + 1 / nRef);
    }
    this.gap_values_ = gaps;
    this.sk_ = sks;
    for (let k = 0; k < maxK - 1; k++) {
      if ((gaps[k] ?? 0) >= (gaps[k + 1] ?? 0) - (sks[k + 1] ?? 0)) { this.n_clusters_ = k + 1; return this; }
    }
    this.n_clusters_ = maxK;
    return this;
  }

  private _kmeansWk(X: Float64Array[], k: number): number {
    const n = X.length;
    const nf = X[0]?.length ?? 0;
    const centroids = X.slice(0, k).map((xi) => new Float64Array(xi));
    const labels = new Int32Array(n);
    for (let iter = 0; iter < 10; iter++) {
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Number.POSITIVE_INFINITY;
        for (let c = 0; c < k; c++) {
          let d = 0;
          for (let j = 0; j < nf; j++) d += ((X[i]?.[j] ?? 0) - (centroids[c]?.[j] ?? 0)) ** 2;
          if (d < bestD) { bestD = d; best = c; }
        }
        labels[i] = best;
      }
      for (let c = 0; c < k; c++) {
        const cm = new Float64Array(nf);
        let cnt = 0;
        for (let i = 0; i < n; i++) if (labels[i] === c) { for (let j = 0; j < nf; j++) cm[j] = (cm[j] ?? 0) + (X[i]?.[j] ?? 0); cnt++; }
        if (cnt > 0) { for (let j = 0; j < nf; j++) cm[j] = (cm[j] ?? 0) / cnt; centroids[c] = cm; }
      }
    }
    let w = 0;
    for (let c = 0; c < k; c++) {
      const members = X.filter((_, i) => labels[i] === c);
      for (const xi of members) for (let j = 0; j < nf; j++) w += ((xi[j] ?? 0) - (centroids[c]?.[j] ?? 0)) ** 2;
    }
    return w;
  }
}
