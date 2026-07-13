/**
 * SubspaceClustering and ProjectedClustering — subspace clustering methods.
 */

export class PROCLUS {
  nClusters: number;
  l: number;
  maxIter: number;
  labels_: Int32Array | null = null;
  subspaces_: number[][] | null = null;
  clusterCenters_: Float64Array[] | null = null;

  constructor(nClusters = 5, l = 3, maxIter = 30) {
    this.nClusters = nClusters;
    this.l = l;
    this.maxIter = maxIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = this.nClusters;
    const l = Math.min(this.l, p);

    // Initialize medoids
    let medoidIdx = Array.from({ length: k }, (_, i) => i % n);
    let subspaces = medoidIdx.map(() =>
      Array.from({ length: l }, (_, j) => j % p)
    );
    this.subspaces_ = subspaces;
    let labels = new Int32Array(n);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Assignment step: assign each point to nearest medoid in its subspace
      const newLabels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        let bestCluster = 0, bestDist = Number.POSITIVE_INFINITY;
        for (let ci = 0; ci < k; ci++) {
          const sub = subspaces[ci] as number[];
          const med = X[medoidIdx[ci] as number] as Float64Array;
          const d = sub.reduce((s, f) => s + ((X[i]?.[f] ?? 0) - (med[f] ?? 0)) ** 2, 0);
          if (d < bestDist) { bestDist = d; bestCluster = ci; }
        }
        newLabels[i] = bestCluster;
      }

      // Update medoids and subspaces
      for (let ci = 0; ci < k; ci++) {
        const members = Array.from({ length: n }, (_, i) => i).filter((i) => newLabels[i] === ci);
        if (members.length === 0) continue;

        // Find best medoid
        let bestMed = members[0] as number, bestIntra = Number.POSITIVE_INFINITY;
        for (const m of members) {
          const intra = members.reduce((s, j) => {
            let d = 0;
            for (let f = 0; f < p; f++) d += ((X[m]?.[f] ?? 0) - (X[j]?.[f] ?? 0)) ** 2;
            return s + d;
          }, 0);
          if (intra < bestIntra) { bestIntra = intra; bestMed = m; }
        }
        medoidIdx[ci] = bestMed;

        // Find best subspace dimensions (lowest variance within cluster)
        const variances = new Float64Array(p);
        const mean = new Float64Array(p);
        for (const m of members) for (let f = 0; f < p; f++) mean[f]! += (X[m]?.[f] ?? 0) / members.length;
        for (const m of members) for (let f = 0; f < p; f++) variances[f]! += ((X[m]?.[f] ?? 0) - (mean[f] ?? 0)) ** 2 / members.length;

        // Select l features with lowest variance
        const sorted = Array.from({ length: p }, (_, f) => f).sort((a, b) => (variances[a] ?? 0) - (variances[b] ?? 0));
        subspaces[ci] = sorted.slice(0, l);
      }

      let changed = false;
      for (let i = 0; i < n; i++) if (newLabels[i] !== labels[i]) { changed = true; break; }
      labels = newLabels;
      if (!changed) break;
    }

    this.labels_ = labels;
    this.subspaces_ = subspaces;

    // Compute centers
    this.clusterCenters_ = Array.from({ length: k }, () => new Float64Array(p));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const ci = labels[i] ?? 0;
      counts[ci]!++;
      for (let f = 0; f < p; f++) (this.clusterCenters_[ci]! as Float64Array)[f]! += ((X[i] as Float64Array)[f] ?? 0);
    }
    for (let ci = 0; ci < k; ci++) {
      const cnt = counts[ci] ?? 1;
      for (let f = 0; f < p; f++) (this.clusterCenters_[ci]! as Float64Array)[f]! /= cnt;
    }
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ ?? new Int32Array(X.length);
  }
}

export class ProjectedKMeans {
  nClusters: number;
  nComponents: number;
  maxIter: number;
  labels_: Int32Array | null = null;
  projectionMatrix_: Float64Array[] | null = null;
  clusterCenters_: Float64Array[] | null = null;
  inertia_: number = 0;

  constructor(nClusters = 5, nComponents = 2, maxIter = 100) {
    this.nClusters = nClusters;
    this.nComponents = nComponents;
    this.maxIter = maxIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const d = Math.min(this.nComponents, p);

    // Random projection
    const scale = 1 / Math.sqrt(d);
    this.projectionMatrix_ = Array.from({ length: p }, () =>
      Float64Array.from({ length: d }, () => (Math.random() < 0.5 ? 1 : -1) * scale)
    );

    // Project data
    const Xproj = X.map((row) => {
      const out = new Float64Array(d);
      for (let j = 0; j < d; j++) for (let i = 0; i < p; i++) out[j]! += (row[i] ?? 0) * ((this.projectionMatrix_![i] as Float64Array)[j] ?? 0);
      return out;
    });

    // K-means in projected space
    let centers = Xproj.slice(0, this.nClusters).map((r) => new Float64Array(r));
    let labels = new Int32Array(n);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const newLabels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Number.POSITIVE_INFINITY;
        for (let ki = 0; ki < this.nClusters; ki++) {
          const d2 = (Xproj[i] as Float64Array).reduce((s, v, j) => s + (v - ((centers[ki] as Float64Array)[j] ?? 0)) ** 2, 0);
          if (d2 < bestD) { bestD = d2; best = ki; }
        }
        newLabels[i] = best;
      }
      const newCenters = Array.from({ length: this.nClusters }, () => new Float64Array(d));
      const counts = new Int32Array(this.nClusters);
      for (let i = 0; i < n; i++) {
        const ci = newLabels[i] ?? 0;
        counts[ci]!++;
        for (let j = 0; j < d; j++) (newCenters[ci]! as Float64Array)[j]! += ((Xproj[i] as Float64Array)[j] ?? 0);
      }
      for (let ki = 0; ki < this.nClusters; ki++) {
        const cnt = counts[ki] ?? 1;
        for (let j = 0; j < d; j++) (newCenters[ki]! as Float64Array)[j]! /= cnt;
      }
      let changed = false;
      for (let i = 0; i < n; i++) if (newLabels[i] !== labels[i]) { changed = true; break; }
      labels = newLabels;
      centers = newCenters;
      if (!changed) break;
    }
    this.labels_ = labels;

    // Compute original-space centers
    this.clusterCenters_ = Array.from({ length: this.nClusters }, () => new Float64Array(p));
    const finalCounts = new Int32Array(this.nClusters);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const ci = labels[i] ?? 0;
      finalCounts[ci]!++;
      for (let f = 0; f < p; f++) (this.clusterCenters_[ci]! as Float64Array)[f]! += ((X[i] as Float64Array)[f] ?? 0);
    }
    for (let ki = 0; ki < this.nClusters; ki++) {
      const cnt = finalCounts[ki] ?? 1;
      for (let f = 0; f < p; f++) (this.clusterCenters_[ki]! as Float64Array)[f]! /= cnt;
    }
    for (let i = 0; i < n; i++) {
      const ci = labels[i] ?? 0;
      inertia += (X[i] as Float64Array).reduce((s, v, f) => s + (v - ((this.clusterCenters_![ci] as Float64Array)[f] ?? 0)) ** 2, 0);
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new Error("Not fitted");
    const centers = this.clusterCenters_;
    return Int32Array.from(X.map((x) => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let ki = 0; ki < centers.length; ki++) {
        const d = x.reduce((s, v, j) => s + (v - ((centers[ki] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = ki; }
      }
      return best;
    }));
  }
}

export class ClusteringWithOutliers {
  innerClusterer: { fit: (X: Float64Array[]) => unknown; labels_: Int32Array | null };
  contamination: number;
  labels_: Int32Array | null = null;
  outlierMask_: Uint8Array | null = null;

  constructor(innerClusterer: { fit: (X: Float64Array[]) => unknown; labels_: Int32Array | null }, contamination = 0.05) {
    this.innerClusterer = innerClusterer;
    this.contamination = contamination;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this.innerClusterer.fit(X);
    const innerLabels = this.innerClusterer.labels_ ?? new Int32Array(n);

    // Detect outliers via LOF-like logic: compute local density
    const nClasses = new Set(Array.from(innerLabels)).size;
    const k = Math.max(3, Math.ceil(Math.sqrt(n / nClasses)));
    const scores = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      // Find k nearest neighbors
      const dists = Array.from({ length: n }, (_, j) => ({
        j,
        d: (X[i] as Float64Array).reduce((s, v, f) => s + (v - ((X[j] as Float64Array)[f] ?? 0)) ** 2, 0),
      }));
      dists.sort((a, b) => a.d - b.d);
      const knn = dists.slice(1, k + 1);
      const kDist = knn[knn.length - 1]?.d ?? 1;
      const lrd = knn.reduce((s, nb) => s + Math.max(kDist, nb.d), 0) / k;
      scores[i] = lrd > 0 ? 1 / lrd : 0;
    }

    const threshold = (() => {
      const sorted = Array.from(scores).sort((a, b) => a - b);
      const idx = Math.floor(this.contamination * n);
      return sorted[idx] ?? 0;
    })();

    this.outlierMask_ = new Uint8Array(n);
    this.labels_ = new Int32Array(innerLabels);
    for (let i = 0; i < n; i++) {
      if ((scores[i] ?? 0) <= threshold) {
        this.outlierMask_[i] = 1;
        this.labels_[i] = -1;
      }
    }
    return this;
  }
}
