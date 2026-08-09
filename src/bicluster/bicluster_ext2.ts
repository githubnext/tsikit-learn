/**
 * Additional biclustering utilities: Plaid model, FABIA algorithm port.
 * Port of sklearn.bicluster extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Plaid biclustering model.
 * Fits an additive model to identify biclusters.
 */
export class PlaidBiclustering {
  private nBiclusters: number;
  private maxIter: number;
  private tol: number;
  private rowClusters_: Int32Array | null = null;
  private colClusters_: Int32Array | null = null;
  private layers_: Array<{ mu: number; rowWeights: Float64Array; colWeights: Float64Array }> = [];

  constructor(options: { nBiclusters?: number; maxIter?: number; tol?: number } = {}) {
    this.nBiclusters = options.nBiclusters ?? 3;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-6;
  }

  fit(X: Float64Array[]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    this.layers_ = [];

    let residual: Float64Array[] = X.map(row => Float64Array.from(row));

    for (let k = 0; k < this.nBiclusters; k++) {
      let rowW = new Float64Array(m).fill(0.5);
      let colW = new Float64Array(n).fill(0.5);
      let mu = 0;

      for (let iter = 0; iter < this.maxIter; iter++) {
        // Update mu
        let num = 0; let den = 0;
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            const rij = residual[i]?.[j] ?? 0;
            const rw = (rowW[i] ?? 0) * (colW[j] ?? 0);
            num += rij * rw;
            den += rw * rw;
          }
        }
        const newMu = den > 0 ? num / den : 0;

        // Update row weights
        const newRowW = new Float64Array(m);
        for (let i = 0; i < m; i++) {
          let nr = 0; let dr = 0;
          for (let j = 0; j < n; j++) {
            const cj = colW[j] ?? 0;
            nr += (residual[i]?.[j] ?? 0) * newMu * cj;
            dr += newMu * newMu * cj * cj;
          }
          newRowW[i] = dr > 0 ? Math.max(0, Math.min(1, nr / dr)) : 0;
        }

        // Update col weights
        const newColW = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          let nc = 0; let dc = 0;
          for (let i = 0; i < m; i++) {
            const ri = newRowW[i] ?? 0;
            nc += (residual[i]?.[j] ?? 0) * newMu * ri;
            dc += newMu * newMu * ri * ri;
          }
          newColW[j] = dc > 0 ? Math.max(0, Math.min(1, nc / dc)) : 0;
        }

        const diff = Math.abs(newMu - mu) + rowW.reduce((s, v, i) => s + Math.abs(v - (newRowW[i] ?? 0)), 0);
        mu = newMu;
        rowW = newRowW;
        colW = newColW;
        if (diff < this.tol) break;
      }

      this.layers_.push({ mu, rowWeights: rowW, colWeights: colW });

      // Update residual
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          residual[i]![j] = (residual[i]?.[j] ?? 0) - mu * (rowW[i] ?? 0) * (colW[j] ?? 0);
        }
      }
    }

    // Assign clusters based on max weight
    this.rowClusters_ = new Int32Array(m).fill(-1);
    this.colClusters_ = new Int32Array(n).fill(-1);
    for (let i = 0; i < m; i++) {
      let best = -1; let bestW = 0.5;
      for (let k = 0; k < this.layers_.length; k++) {
        const w = this.layers_[k]?.rowWeights[i] ?? 0;
        if (w > bestW) { bestW = w; best = k; }
      }
      this.rowClusters_[i] = best;
    }
    for (let j = 0; j < n; j++) {
      let best = -1; let bestW = 0.5;
      for (let k = 0; k < this.layers_.length; k++) {
        const w = this.layers_[k]?.colWeights[j] ?? 0;
        if (w > bestW) { bestW = w; best = k; }
      }
      this.colClusters_[j] = best;
    }

    return this;
  }

  get rowLabels_(): Int32Array {
    if (!this.rowClusters_) throw new NotFittedError("PlaidBiclustering not fitted");
    return this.rowClusters_;
  }

  get columnLabels_(): Int32Array {
    if (!this.colClusters_) throw new NotFittedError("PlaidBiclustering not fitted");
    return this.colClusters_;
  }
}

/**
 * FABIA (Factor Analysis for Bicluster Acquisition) simplified port.
 */
export class FABIABiclustering {
  private nBiclusters: number;
  private alpha: number;
  private maxIter: number;
  private fitted = false;
  private rowClusters_: Int32Array = new Int32Array(0);
  private colClusters_: Int32Array = new Int32Array(0);

  constructor(options: { nBiclusters?: number; alpha?: number; maxIter?: number } = {}) {
    this.nBiclusters = options.nBiclusters ?? 5;
    this.alpha = options.alpha ?? 0.01;
    this.maxIter = options.maxIter ?? 500;
  }

  fit(X: Float64Array[]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    const k = this.nBiclusters;

    // Initialize loading matrix L (n x k) and factor matrix Z (k x m)
    const L: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = (Math.random() - 0.5) * 0.1;
      return row;
    });

    // Simplified EM: use random projections + thresholding
    const rowScores = new Float64Array(m * k);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          s += (X[i]?.[j] ?? 0) * (L[j]?.[c] ?? 0);
        }
        rowScores[i * k + c] = s;
      }
    }

    // Update L using gradient
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let j = 0; j < n; j++) {
        for (let c = 0; c < k; c++) {
          let grad = 0;
          for (let i = 0; i < m; i++) {
            const z = rowScores[i * k + c] ?? 0;
            const xij = X[i]?.[j] ?? 0;
            grad += xij * z;
          }
          L[j]![c] = (L[j]?.[c] ?? 0) + this.alpha * grad / m - this.alpha * (L[j]?.[c] ?? 0);
        }
      }
    }

    // Assign clusters
    this.rowClusters_ = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      let best = 0; let bestS = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        const s = Math.abs(rowScores[i * k + c] ?? 0);
        if (s > bestS) { bestS = s; best = c; }
      }
      this.rowClusters_[i] = best;
    }

    this.colClusters_ = new Int32Array(n);
    for (let j = 0; j < n; j++) {
      let best = 0; let bestW = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        const w = Math.abs(L[j]?.[c] ?? 0);
        if (w > bestW) { bestW = w; best = c; }
      }
      this.colClusters_[j] = best;
    }

    this.fitted = true;
    return this;
  }

  get rowLabels(): Int32Array {
    if (!this.fitted) throw new NotFittedError("FABIABiclustering not fitted");
    return this.rowClusters_;
  }

  get columnLabels(): Int32Array {
    if (!this.fitted) throw new NotFittedError("FABIABiclustering not fitted");
    return this.colClusters_;
  }
}

/**
 * Delta-biclustering: identifies biclusters with constant values within rows.
 */
export class DeltaBiclustering {
  private delta: number;
  private minRows: number;
  private minCols: number;

  constructor(options: { delta?: number; minRows?: number; minCols?: number } = {}) {
    this.delta = options.delta ?? 1.0;
    this.minRows = options.minRows ?? 2;
    this.minCols = options.minCols ?? 2;
  }

  findBiclusters(X: Float64Array[]): Array<{ rows: number[]; cols: number[] }> {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    const biclusters: Array<{ rows: number[]; cols: number[] }> = [];

    // Greedy search: for each pair of rows, find columns with small range
    for (let i1 = 0; i1 < m - 1; i1++) {
      for (let i2 = i1 + 1; i2 < m; i2++) {
        const cols: number[] = [];
        for (let j = 0; j < n; j++) {
          const diff = Math.abs((X[i1]?.[j] ?? 0) - (X[i2]?.[j] ?? 0));
          if (diff <= this.delta) cols.push(j);
        }
        if (cols.length >= this.minCols) {
          const rows = [i1, i2];
          // Try to extend with more rows
          for (let i3 = 0; i3 < m; i3++) {
            if (i3 === i1 || i3 === i2) continue;
            const valid = cols.every(j => {
              const range = [X[i1]?.[j] ?? 0, X[i2]?.[j] ?? 0, X[i3]?.[j] ?? 0];
              return Math.max(...range) - Math.min(...range) <= this.delta;
            });
            if (valid) rows.push(i3);
          }
          if (rows.length >= this.minRows) {
            biclusters.push({ rows, cols });
          }
        }
      }
    }

    return biclusters;
  }
}
