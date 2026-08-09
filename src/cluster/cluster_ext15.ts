/**
 * Online Mini-Batch K-Means extension and Affinity Propagation variant.
 */

export class IncrementalKMeans {
  private centers_!: Float64Array[];
  private counts_!: Int32Array;
  private fitted_ = false;

  constructor(private nClusters = 8, private batchSize = 100, private maxIter = 100) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 1;
    this.centers_ = Array.from({ length: this.nClusters }, (_, k) =>
      new Float64Array(X[k % X.length]!)
    );
    this.counts_ = new Int32Array(this.nClusters);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const batch = Array.from({ length: this.batchSize }, () => X[Math.floor(Math.random() * X.length)]!);
      const assigns = batch.map(x => this._nearest(x));
      const eta = 1 / (iter + 1);
      for (let i = 0; i < batch.length; i++) {
        const c = assigns[i]!;
        this.counts_[c] = (this.counts_[c] ?? 0) + 1;
        const lr = 1 / this.counts_[c]!;
        for (let j = 0; j < p; j++) {
          this.centers_[c]![j] = (this.centers_[c]![j] ?? 0) * (1 - lr) + (batch[i]![j] ?? 0) * lr;
        }
        void eta;
      }
    }
    this.fitted_ = true;
    return this;
  }

  private _nearest(x: Float64Array): number {
    let best = 0, bestD = Number.POSITIVE_INFINITY;
    for (let k = 0; k < this.nClusters; k++) {
      const d = x.reduce((s, v, j) => s + (v - (this.centers_[k]![j] ?? 0)) ** 2, 0);
      if (d < bestD) { bestD = d; best = k; }
    }
    return best;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => this._nearest(x)));
  }

  get clusterCenters(): Float64Array[] { return this.centers_; }
}

export class BisectingKMeans {
  private labels_!: Int32Array;
  private centers_!: Float64Array[];
  private fitted_ = false;

  constructor(private nClusters = 8) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    let clusters: number[][] = [Array.from({ length: n }, (_, i) => i)];
    const centers: Float64Array[] = [];

    while (clusters.length < this.nClusters) {
      // Pick largest cluster to split
      const largest = clusters.reduce((best, c, i) => c.length > (clusters[best]?.length ?? 0) ? i : best, 0);
      const clusterIndices = clusters[largest]!;
      const clusterX = clusterIndices.map(i => X[i]!);
      const p = clusterX[0]?.length ?? 1;

      // Simple k-means 2
      let c0 = new Float64Array(clusterX[0] ?? []);
      let c1 = new Float64Array(clusterX[clusterX.length - 1] ?? []);
      const assigns = new Int32Array(clusterX.length);
      for (let iter = 0; iter < 10; iter++) {
        for (let i = 0; i < clusterX.length; i++) {
          const d0 = clusterX[i]!.reduce((s, v, j) => s + (v - (c0[j] ?? 0)) ** 2, 0);
          const d1 = clusterX[i]!.reduce((s, v, j) => s + (v - (c1[j] ?? 0)) ** 2, 0);
          assigns[i] = d0 <= d1 ? 0 : 1;
        }
        c0 = new Float64Array(p);
        c1 = new Float64Array(p);
        let n0 = 0, n1 = 0;
        for (let i = 0; i < clusterX.length; i++) {
          if (assigns[i] === 0) { n0++; for (let j = 0; j < p; j++) c0[j] = (c0[j] ?? 0) + (clusterX[i]![j] ?? 0); }
          else { n1++; for (let j = 0; j < p; j++) c1[j] = (c1[j] ?? 0) + (clusterX[i]![j] ?? 0); }
        }
        if (n0 > 0) for (let j = 0; j < p; j++) c0[j] = (c0[j] ?? 0) / n0;
        if (n1 > 0) for (let j = 0; j < p; j++) c1[j] = (c1[j] ?? 0) / n1;
      }

      const g0 = clusterIndices.filter((_, i) => assigns[i] === 0);
      const g1 = clusterIndices.filter((_, i) => assigns[i] === 1);
      clusters.splice(largest, 1, g0, g1);
      centers.push(c0, c1);
    }

    this.centers_ = centers.slice(0, this.nClusters);
    this.labels_ = new Int32Array(n);
    for (let c = 0; c < clusters.length; c++) {
      for (const idx of clusters[c]!) this.labels_[idx] = c;
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => {
      let best = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.centers_.length; k++) {
        const d = x.reduce((s, v, j) => s + (v - (this.centers_[k]![j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; best = k; }
      }
      return best;
    }));
  }

  get labels(): Int32Array { return this.labels_; }
}
