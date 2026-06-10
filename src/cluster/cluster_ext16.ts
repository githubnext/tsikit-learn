/**
 * Constrained K-Means and K-Medoids clustering algorithms.
 */

export class KMedoids {
  private medoidIndices_!: number[];
  private labels_!: Int32Array;
  private fitted_ = false;

  constructor(private nClusters = 8, private maxIter = 100) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    // Initialize medoids randomly
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [perm[i], perm[j]] = [perm[j]!, perm[i]!];
    }
    this.medoidIndices_ = perm.slice(0, this.nClusters);
    this.labels_ = new Int32Array(n);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Assign
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Number.POSITIVE_INFINITY;
        for (let k = 0; k < this.nClusters; k++) {
          const m = this.medoidIndices_[k]!;
          const d = X[i]!.reduce((s, v, j) => s + (v - (X[m]![j] ?? 0)) ** 2, 0);
          if (d < bestD) { bestD = d; best = k; }
        }
        this.labels_[i] = best;
      }
      // Update medoids
      let changed = false;
      for (let k = 0; k < this.nClusters; k++) {
        const members = Array.from({ length: n }, (_, i) => i).filter(i => this.labels_[i] === k);
        if (members.length === 0) continue;
        let bestMed = this.medoidIndices_[k]!, bestCost = Number.POSITIVE_INFINITY;
        for (const cand of members) {
          const cost = members.reduce((s, i) => s + X[i]!.reduce((ss, v, j) => ss + (v - (X[cand]![j] ?? 0)) ** 2, 0), 0);
          if (cost < bestCost) { bestCost = cost; bestMed = cand; }
        }
        if (bestMed !== this.medoidIndices_[k]) { this.medoidIndices_[k] = bestMed; changed = true; }
      }
      if (!changed) break;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[], XTrain?: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const base = XTrain ?? X;
    return new Int32Array(X.map(x => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const m = this.medoidIndices_[k]!;
        const d = x.reduce((s, v, j) => s + (v - (base[m]![j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = k; }
      }
      return best;
    }));
  }

  get medoidIndices(): number[] { return this.medoidIndices_; }
  get labels(): Int32Array { return this.labels_; }
}

export class FuzzyKMeans {
  private centers_!: Float64Array[];
  private membership_!: Float64Array[];
  private fitted_ = false;

  constructor(private nClusters = 3, private m = 2.0, private maxIter = 150, private tol = 1e-4) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    this.centers_ = Array.from({ length: this.nClusters }, () =>
      new Float64Array(X[Math.floor(Math.random() * n)]!)
    );
    this.membership_ = Array.from({ length: n }, () => {
      const u = new Float64Array(this.nClusters).map(() => Math.random());
      const s = u.reduce((a, b) => a + b, 0);
      return new Float64Array(u.map(v => v / s));
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      const prevCenters = this.centers_.map(c => new Float64Array(c));
      // Update centers
      for (let k = 0; k < this.nClusters; k++) {
        this.centers_[k] = new Float64Array(p);
        let denom = 0;
        for (let i = 0; i < n; i++) {
          const uk = this.membership_[i]![k] ?? 0;
          const ukm = uk ** this.m;
          denom += ukm;
          for (let j = 0; j < p; j++) {
            this.centers_[k]![j] = (this.centers_[k]![j] ?? 0) + ukm * (X[i]![j] ?? 0);
          }
        }
        if (denom > 0) for (let j = 0; j < p; j++) this.centers_[k]![j] = (this.centers_[k]![j] ?? 0) / denom;
      }
      // Update membership
      for (let i = 0; i < n; i++) {
        const dists = this.centers_.map(c => Math.sqrt(X[i]!.reduce((s, v, j) => s + (v - (c[j] ?? 0)) ** 2, 0)) + 1e-10);
        for (let k = 0; k < this.nClusters; k++) {
          const sum = dists.reduce((s, dj) => s + ((dists[k] ?? 1) / dj) ** (2 / (this.m - 1)), 0);
          this.membership_[i]![k] = 1 / sum;
        }
      }
      const diff = this.centers_.reduce((s, c, k) => s + c.reduce((ss, v, j) => ss + (v - (prevCenters[k]![j] ?? 0)) ** 2, 0), 0);
      if (diff < this.tol) break;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const d = x.reduce((s, v, j) => s + (v - (this.centers_[k]![j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = k; }
      }
      return best;
    }));
  }

  get membership(): Float64Array[] { return this.membership_; }
  get clusterCenters(): Float64Array[] { return this.centers_; }
}
