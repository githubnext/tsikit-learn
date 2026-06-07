/**
 * TwoLevelClustering and EnsembleClustering — hierarchical cluster ensemble methods.
 */

function euclidean(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(d);
}

function kMeansOneStep(X: Float64Array[], k: number, maxIter = 100): Int32Array {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  let centers = X.slice(0, k).map((r) => new Float64Array(r));
  let labels = new Int32Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assignment
    const newLabels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let ki = 0; ki < k; ki++) {
        const d = (X[i] as Float64Array).reduce((s, v, j) => s + (v - ((centers[ki] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = ki; }
      }
      newLabels[i] = best;
    }
    // Update
    const newCenters = Array.from({ length: k }, () => new Float64Array(p));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const ci = newLabels[i] ?? 0;
      counts[ci]++;
      for (let j = 0; j < p; j++) (newCenters[ci] as Float64Array)[j] += ((X[i] as Float64Array)[j] ?? 0);
    }
    for (let ki = 0; ki < k; ki++) {
      const cnt = counts[ki] ?? 1;
      for (let j = 0; j < p; j++) (newCenters[ki] as Float64Array)[j] /= cnt;
    }

    let changed = false;
    for (let i = 0; i < n; i++) if (newLabels[i] !== labels[i]) { changed = true; break; }
    labels = newLabels;
    centers = newCenters;
    if (!changed) break;
  }
  return labels;
}

export class TwoLevelClustering {
  nClusters: number;
  nSubClusters: number;
  maxIter: number;
  labels_: Int32Array | null = null;
  clusterCenters_: Float64Array[] | null = null;
  inertia_: number = 0;

  constructor(nClusters = 5, nSubClusters = 20, maxIter = 300) {
    this.nClusters = nClusters;
    this.nSubClusters = nSubClusters;
    this.maxIter = maxIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    // Level 1: over-cluster
    const subLabels = kMeansOneStep(X, Math.min(this.nSubClusters, n), this.maxIter);

    // Compute sub-cluster centers
    const subCenters: Float64Array[] = Array.from({ length: this.nSubClusters }, () => new Float64Array(p));
    const subCounts = new Int32Array(this.nSubClusters);
    for (let i = 0; i < n; i++) {
      const ci = subLabels[i] ?? 0;
      if (ci < this.nSubClusters) {
        subCounts[ci]++;
        for (let j = 0; j < p; j++) (subCenters[ci] as Float64Array)[j] += ((X[i] as Float64Array)[j] ?? 0);
      }
    }
    const activeSubs: Float64Array[] = [];
    for (let ki = 0; ki < this.nSubClusters; ki++) {
      if ((subCounts[ki] ?? 0) > 0) {
        const cnt = subCounts[ki] ?? 1;
        activeSubs.push(new Float64Array(Array.from((subCenters[ki] as Float64Array)).map((v) => v / cnt)));
      }
    }

    // Level 2: cluster the sub-cluster centers
    const topLabels = kMeansOneStep(activeSubs, Math.min(this.nClusters, activeSubs.length), this.maxIter);

    // Map original points to top-level labels
    const finalLabels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const sub = subLabels[i] ?? 0;
      finalLabels[i] = topLabels[sub] ?? 0;
    }

    // Compute centers
    const centers = Array.from({ length: this.nClusters }, () => new Float64Array(p));
    const counts = new Int32Array(this.nClusters);
    for (let i = 0; i < n; i++) {
      const ci = finalLabels[i] ?? 0;
      counts[ci]++;
      for (let j = 0; j < p; j++) (centers[ci] as Float64Array)[j] += ((X[i] as Float64Array)[j] ?? 0);
    }
    for (let ki = 0; ki < this.nClusters; ki++) {
      const cnt = counts[ki] ?? 1;
      for (let j = 0; j < p; j++) (centers[ki] as Float64Array)[j] /= cnt;
    }

    this.labels_ = finalLabels;
    this.clusterCenters_ = centers;
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const ci = finalLabels[i] ?? 0;
      inertia += (X[i] as Float64Array).reduce((s, v, j) => s + (v - ((centers[ci] as Float64Array)[j] ?? 0)) ** 2, 0);
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new Error("Not fitted");
    return Int32Array.from(X.map((x) => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let ki = 0; ki < (this.clusterCenters_?.length ?? 0); ki++) {
        const d = euclidean(x, this.clusterCenters_![ki] as Float64Array);
        if (d < bestD) { bestD = d; best = ki; }
      }
      return best;
    }));
  }
}

export class EnsembleClustering {
  baseClusterers: Array<{ fit: (X: Float64Array[]) => unknown; labels_: Int32Array | null }>;
  nClusters: number;
  consensusMethod: "co-occurrence" | "voting";
  labels_: Int32Array | null = null;

  constructor(
    baseClusterers: Array<{ fit: (X: Float64Array[]) => unknown; labels_: Int32Array | null }>,
    nClusters = 5,
    consensusMethod: "co-occurrence" | "voting" = "co-occurrence",
  ) {
    this.baseClusterers = baseClusterers;
    this.nClusters = nClusters;
    this.consensusMethod = consensusMethod;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    // Run all clusterers
    const allLabels: Int32Array[] = [];
    for (const clf of this.baseClusterers) {
      clf.fit(X);
      if (clf.labels_) allLabels.push(clf.labels_);
    }
    if (allLabels.length === 0) {
      this.labels_ = new Int32Array(n);
      return this;
    }

    if (this.consensusMethod === "co-occurrence") {
      // Build co-occurrence matrix
      const coOcc: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
      for (const lbls of allLabels) {
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            if ((lbls[i] ?? -1) === (lbls[j] ?? -2)) {
              (coOcc[i] as Float64Array)[j]++;
              (coOcc[j] as Float64Array)[i]++;
            }
          }
        }
      }
      // Normalize
      const nEstimators = allLabels.length;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) (coOcc[i] as Float64Array)[j] /= nEstimators;
      // Spectral clustering on co-occurrence (simplified: greedy assignment)
      const labels = new Int32Array(n).fill(-1);
      let cluster = 0;
      const assigned = new Uint8Array(n);
      for (let i = 0; i < n && cluster < this.nClusters; i++) {
        if (assigned[i]) continue;
        labels[i] = cluster;
        assigned[i] = 1;
        for (let j = i + 1; j < n; j++) {
          if (!assigned[j] && ((coOcc[i] as Float64Array)[j] ?? 0) > 0.5) {
            labels[j] = cluster;
            assigned[j] = 1;
          }
        }
        cluster++;
      }
      // Assign remaining
      for (let i = 0; i < n; i++) {
        if (labels[i] === -1) labels[i] = Math.floor(Math.random() * this.nClusters);
      }
      this.labels_ = labels;
    } else {
      // Simple majority vote (take mode of cluster assignments)
      this.labels_ = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const votes: Map<number, number> = new Map();
        for (const lbls of allLabels) {
          const l = lbls[i] ?? 0;
          votes.set(l, (votes.get(l) ?? 0) + 1);
        }
        let bestL = 0, bestV = -1;
        for (const [l, v] of votes) if (v > bestV) { bestV = v; bestL = l; }
        this.labels_[i] = bestL % this.nClusters;
      }
    }
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ ?? new Int32Array(X.length);
  }
}

export class RandomSubspaceClustering {
  nClusters: number;
  nSubspaces: number;
  subsampleRatio: number;
  maxIter: number;
  labels_: Int32Array | null = null;
  selectedFeatures_: number[][] | null = null;

  constructor(nClusters = 5, nSubspaces = 10, subsampleRatio = 0.7, maxIter = 100) {
    this.nClusters = nClusters;
    this.nSubspaces = nSubspaces;
    this.subsampleRatio = subsampleRatio;
    this.maxIter = maxIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const nFeats = Math.max(1, Math.ceil(p * this.subsampleRatio));
    this.selectedFeatures_ = [];
    const allLabels: Int32Array[] = [];

    for (let s = 0; s < this.nSubspaces; s++) {
      // Random feature subset
      const feats = Array.from({ length: p }, (_, i) => i);
      for (let i = p - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = feats[i]; feats[i] = feats[j] as number; feats[j] = t as number;
      }
      const selected = feats.slice(0, nFeats);
      this.selectedFeatures_.push(selected);

      // Project data
      const Xsub = X.map((row) => Float64Array.from(selected, (f) => row[f] ?? 0));
      allLabels.push(kMeansOneStep(Xsub, this.nClusters, this.maxIter));
    }

    // Consensus via majority voting (modular cluster IDs)
    const finalLabels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const votes: Map<number, number> = new Map();
      for (const lbls of allLabels) {
        const l = (lbls[i] ?? 0) % this.nClusters;
        votes.set(l, (votes.get(l) ?? 0) + 1);
      }
      let bestL = 0, bestV = -1;
      for (const [l, v] of votes) if (v > bestV) { bestV = v; bestL = l; }
      finalLabels[i] = bestL;
    }
    this.labels_ = finalLabels;
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ ?? new Int32Array(X.length);
  }
}
