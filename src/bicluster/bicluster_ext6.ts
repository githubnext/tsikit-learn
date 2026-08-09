/**
 * Co-clustering with alternating optimization (Bregman divergence-based).
 */

export class CoclusteringExt {
  private rowLabels_!: Int32Array;
  private colLabels_!: Int32Array;
  private fitted_ = false;

  constructor(
    private nRowClusters = 3,
    private nColClusters = 3,
    private maxIter = 100,
    private tol = 1e-4
  ) {}

  fit(X: Float64Array[]): this {
    const nRows = X.length, nCols = X[0]?.length ?? 0;
    // Initialize row and col labels randomly
    this.rowLabels_ = new Int32Array(nRows).map(() => Math.floor(Math.random() * this.nRowClusters));
    this.colLabels_ = new Int32Array(nCols).map(() => Math.floor(Math.random() * this.nColClusters));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const oldRowLabels = new Int32Array(this.rowLabels_);
      const oldColLabels = new Int32Array(this.colLabels_);

      // Update row assignments
      this.rowLabels_ = this._updateRowLabels(X, nRows, nCols);
      // Update col assignments
      this.colLabels_ = this._updateColLabels(X, nRows, nCols);

      const changed = Array.from(this.rowLabels_).some((v, i) => v !== oldRowLabels[i])
        || Array.from(this.colLabels_).some((v, j) => v !== oldColLabels[j]);
      if (!changed) break;
    }
    this.fitted_ = true;
    void this.tol;
    return this;
  }

  private _updateRowLabels(X: Float64Array[], nRows: number, nCols: number): Int32Array {
    // Compute co-cluster centroids
    const centroids = Array.from({ length: this.nRowClusters }, () =>
      Array.from({ length: this.nColClusters }, () => ({ sum: 0, count: 0 }))
    );
    for (let i = 0; i < nRows; i++) {
      const rc = this.rowLabels_[i] ?? 0;
      for (let j = 0; j < nCols; j++) {
        const cc = this.colLabels_[j] ?? 0;
        centroids[rc]![cc]!.sum += X[i]![j] ?? 0;
        centroids[rc]![cc]!.count++;
      }
    }
    // Assign each row to nearest row cluster
    return new Int32Array(nRows).map((_, i) => {
      let best = 0, bestDist = Number.POSITIVE_INFINITY;
      for (let rc = 0; rc < this.nRowClusters; rc++) {
        let dist = 0;
        for (let j = 0; j < nCols; j++) {
          const cc = this.colLabels_[j] ?? 0;
          const c = centroids[rc]![cc]!;
          const mean = c.count > 0 ? c.sum / c.count : 0;
          dist += ((X[i]![j] ?? 0) - mean) ** 2;
        }
        if (dist < bestDist) { bestDist = dist; best = rc; }
      }
      return best;
    });
  }

  private _updateColLabels(X: Float64Array[], nRows: number, nCols: number): Int32Array {
    const centroids = Array.from({ length: this.nRowClusters }, () =>
      Array.from({ length: this.nColClusters }, () => ({ sum: 0, count: 0 }))
    );
    for (let i = 0; i < nRows; i++) {
      const rc = this.rowLabels_[i] ?? 0;
      for (let j = 0; j < nCols; j++) {
        const cc = this.colLabels_[j] ?? 0;
        centroids[rc]![cc]!.sum += X[i]![j] ?? 0;
        centroids[rc]![cc]!.count++;
      }
    }
    return new Int32Array(nCols).map((_, j) => {
      let best = 0, bestDist = Number.POSITIVE_INFINITY;
      for (let cc = 0; cc < this.nColClusters; cc++) {
        let dist = 0;
        for (let i = 0; i < nRows; i++) {
          const rc = this.rowLabels_[i] ?? 0;
          const c = centroids[rc]![cc]!;
          const mean = c.count > 0 ? c.sum / c.count : 0;
          dist += ((X[i]![j] ?? 0) - mean) ** 2;
        }
        if (dist < bestDist) { bestDist = dist; best = cc; }
      }
      return best;
    });
  }

  get rowLabels(): Int32Array { return this.rowLabels_; }
  get columnLabels(): Int32Array { return this.colLabels_; }

  getBiclusterRows(rcIdx: number): Int32Array {
    return new Int32Array(Array.from(this.rowLabels_).map((v, i) => v === rcIdx ? i : -1).filter(i => i >= 0));
  }
  getBiclusterColumns(ccIdx: number): Int32Array {
    return new Int32Array(Array.from(this.colLabels_).map((v, j) => v === ccIdx ? j : -1).filter(j => j >= 0));
  }
}

export class DiagonalBlockClustering extends CoclusteringExt {
  constructor(nClusters = 3, maxIter = 100) {
    super(nClusters, nClusters, maxIter);
  }
}
