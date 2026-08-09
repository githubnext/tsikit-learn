/**
 * Layered Biclustering: finds biclusters in layers of a matrix.
 */

export interface Bicluster {
  rowIndices: Int32Array;
  colIndices: Int32Array;
  score: number;
}

export class LayeredBiclustering {
  private biclusters_: Bicluster[] = [];
  private fitted_ = false;

  constructor(
    private nClusters = 3,
    private nLayers = 2,
    private maxIter = 100,
    private tol = 1e-4
  ) {}

  fit(X: Float64Array[]): this {
    const nRows = X.length, nCols = X[0]?.length ?? 0;
    this.biclusters_ = [];

    let residual = X.map(row => new Float64Array(row));

    for (let layer = 0; layer < this.nLayers; layer++) {
      for (let c = 0; c < this.nClusters; c++) {
        // Random initialization
        let rowMask = new Int32Array(nRows).map(() => Math.random() < 0.5 ? 1 : 0);
        let colMask = new Int32Array(nCols).map(() => Math.random() < 0.5 ? 1 : 0);
        if (!Array.from(rowMask).some(v => v)) rowMask[0] = 1;
        if (!Array.from(colMask).some(v => v)) colMask[0] = 1;

        for (let iter = 0; iter < this.maxIter; iter++) {
          const oldRowMask = new Int32Array(rowMask);
          const oldColMask = new Int32Array(colMask);

          // Update col mask: include cols that improve bicluster coherence
          const colIndices = Array.from(colMask).map((v, j) => v ? j : -1).filter(j => j >= 0);
          for (let j = 0; j < nCols; j++) {
            const rowIndices = Array.from(rowMask).map((v, i) => v ? i : -1).filter(i => i >= 0);
            if (rowIndices.length === 0) continue;
            const mean = rowIndices.reduce((s, i) => s + (residual[i]![j] ?? 0), 0) / rowIndices.length;
            const variance = rowIndices.reduce((s, i) => s + ((residual[i]![j] ?? 0) - mean) ** 2, 0) / rowIndices.length;
            void colIndices;
            colMask[j] = variance < 1 ? 1 : 0;
          }
          if (!Array.from(colMask).some(v => v)) colMask[0] = 1;

          // Update row mask
          const newColIndices = Array.from(colMask).map((v, j) => v ? j : -1).filter(j => j >= 0);
          for (let i = 0; i < nRows; i++) {
            if (newColIndices.length === 0) continue;
            const mean = newColIndices.reduce((s, j) => s + (residual[i]![j] ?? 0), 0) / newColIndices.length;
            const variance = newColIndices.reduce((s, j) => s + ((residual[i]![j] ?? 0) - mean) ** 2, 0) / newColIndices.length;
            rowMask[i] = variance < 1 ? 1 : 0;
          }
          if (!Array.from(rowMask).some(v => v)) rowMask[0] = 1;

          const changed = Array.from(rowMask).some((v, i) => v !== oldRowMask[i])
            || Array.from(colMask).some((v, j) => v !== oldColMask[j]);
          if (!changed) break;
        }

        const rowIndices = new Int32Array(Array.from(rowMask).map((v, i) => v ? i : -1).filter(i => i >= 0));
        const colIndices = new Int32Array(Array.from(colMask).map((v, j) => v ? j : -1).filter(j => j >= 0));
        if (rowIndices.length > 0 && colIndices.length > 0) {
          // Score: mean squared coherence
          const score = this._biclusterScore(residual, rowIndices, colIndices);
          this.biclusters_.push({ rowIndices, colIndices, score });
          // Subtract bicluster mean from residual
          const biMean = Array.from(rowIndices).reduce((s, i) =>
            s + Array.from(colIndices).reduce((cs, j) => cs + (residual[i]![j] ?? 0), 0), 0)
            / (rowIndices.length * colIndices.length);
          for (const i of rowIndices) for (const j of colIndices) residual[i]![j] = (residual[i]![j] ?? 0) - biMean;
        }
      }
    }
    this.fitted_ = true;
    void this.tol;
    return this;
  }

  private _biclusterScore(X: Float64Array[], rows: Int32Array, cols: Int32Array): number {
    const vals = Array.from(rows).flatMap(i => Array.from(cols).map(j => X[i]![j] ?? 0));
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    return -vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
  }

  get biclusters(): Bicluster[] { return this.biclusters_; }
  get rowLabels(): Int32Array[] { return this.biclusters_.map(b => b.rowIndices); }
  get columnLabels(): Int32Array[] { return this.biclusters_.map(b => b.colIndices); }
}
