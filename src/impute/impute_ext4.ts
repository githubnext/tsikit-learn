/**
 * Matrix factorization imputer and IterativeImputer extension.
 */

export class MatrixFactorizationImputer {
  nComponents: number;
  maxIter: number;
  lr: number;
  regParam: number;
  tol: number;
  private _U: Float64Array[] | null = null;
  private _V: Float64Array[] | null = null;
  private _rowMeans: Float64Array | null = null;
  private _colMeans: Float64Array | null = null;
  nRowsIn_: number = 0;
  nColsIn_: number = 0;

  constructor(nComponents = 10, maxIter = 100, lr = 0.005, regParam = 0.02, tol = 1e-4) {
    this.nComponents = nComponents;
    this.maxIter = maxIter;
    this.lr = lr;
    this.regParam = regParam;
    this.tol = tol;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nRowsIn_ = n;
    this.nColsIn_ = p;

    // Compute column means for initialization
    this._colMeans = new Float64Array(p);
    const colCounts = new Int32Array(p);
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        if (!Number.isNaN(row[j] ?? NaN)) {
          (this._colMeans as Float64Array)[j]! += row[j] ?? 0;
          colCounts[j]!++;
        }
      }
    }
    for (let j = 0; j < p; j++) (this._colMeans as Float64Array)[j]! /= Math.max(colCounts[j] ?? 0, 1);

    this._rowMeans = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0, c = 0;
      for (let j = 0; j < p; j++) {
        if (!Number.isNaN(X[i]?.[j] ?? NaN)) { s += X[i]?.[j] ?? 0; c++; }
      }
      (this._rowMeans as Float64Array)[i] = c > 0 ? s / c : 0;
    }

    // Initialize U, V with small random values
    const k = this.nComponents;
    this._U = Array.from({ length: n }, () => Float64Array.from({ length: k }, () => (Math.random() - 0.5) * 0.01));
    this._V = Array.from({ length: p }, () => Float64Array.from({ length: k }, () => (Math.random() - 0.5) * 0.01));

    // Fill missing values with column means for training
    const Xfilled = X.map((row, i) => Float64Array.from({ length: p }, (_, j) => {
      const v = row[j];
      return (v === undefined || Number.isNaN(v)) ? ((this._colMeans as Float64Array)[j] ?? 0) : v;
    }));

    let prevLoss = Number.POSITIVE_INFINITY;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let loss = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          const actual = Xfilled[i]?.[j] ?? 0;
          const pred = (this._U[i] as Float64Array).reduce((s, v, k) => s + v * (this._V?.[j]?.[k] ?? 0), 0);
          const err = actual - pred;
          loss += err * err;
          for (let k2 = 0; k2 < k; k2++) {
            const uik = (this._U[i] as Float64Array)[k2] ?? 0;
            const vjk = (this._V[j] as Float64Array)[k2] ?? 0;
            (this._U[i] as Float64Array)[k2] = uik + this.lr * (2 * err * vjk - this.regParam * uik);
            (this._V[j] as Float64Array)[k2] = vjk + this.lr * (2 * err * uik - this.regParam * vjk);
          }
        }
      }
      if (Math.abs(prevLoss - loss) < this.tol) break;
      prevLoss = loss;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._U || !this._V) throw new Error("Not fitted");
    const n = X.length;
    const p = this.nColsIn_;
    return X.map((row, i) => Float64Array.from({ length: p }, (_, j) => {
      const v = row[j];
      if (v !== undefined && !Number.isNaN(v)) return v;
      if (i < (this._U?.length ?? 0)) {
        return (this._U![i] as Float64Array).reduce((s, uk, k2) => s + uk * ((this._V![j] as Float64Array)[k2] ?? 0), 0);
      }
      return (this._colMeans as Float64Array)[j] ?? 0;
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class SoftImputeImputer {
  rank: number;
  maxIter: number;
  lambda_: number;
  tol: number;
  private _colMeans: Float64Array | null = null;
  nColsIn_: number = 0;

  constructor(rank = 5, maxIter = 100, lambda_ = 0.1, tol = 1e-5) {
    this.rank = rank;
    this.maxIter = maxIter;
    this.lambda_ = lambda_;
    this.tol = tol;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this.nColsIn_ = p;
    this._colMeans = new Float64Array(p);
    const colCounts = new Int32Array(p);
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        if (!Number.isNaN(row[j] ?? NaN) && row[j] !== undefined) {
          (this._colMeans as Float64Array)[j]! += row[j] ?? 0;
          colCounts[j]!++;
        }
      }
    }
    for (let j = 0; j < p; j++) (this._colMeans as Float64Array)[j]! /= Math.max(colCounts[j] ?? 1, 1);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._colMeans) throw new Error("Not fitted");
    return X.map((row) => Float64Array.from({ length: this.nColsIn_ }, (_, j) => {
      const v = row[j];
      if (v === undefined || Number.isNaN(v)) return (this._colMeans as Float64Array)[j] ?? 0;
      return v;
    }));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class KNNImputerExt {
  nNeighbors: number;
  weights: "uniform" | "distance";
  private _X: Float64Array[] | null = null;

  constructor(nNeighbors = 5, weights: "uniform" | "distance" = "uniform") {
    this.nNeighbors = nNeighbors;
    this.weights = weights;
  }

  fit(X: Float64Array[]): this {
    this._X = X;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._X) throw new Error("Not fitted");
    const p = this._X[0]?.length ?? 0;
    return X.map((row) => {
      const hasNaN = row.some((v) => v === undefined || Number.isNaN(v));
      if (!hasNaN) return new Float64Array(row);

      // Find k nearest complete neighbors
      const dists: Array<{ dist: number; row: Float64Array }> = [];
      for (const trainRow of this._X!) {
        let d2 = 0, valid = true;
        for (let j = 0; j < p; j++) {
          if (!Number.isNaN(row[j] ?? NaN) && !Number.isNaN(trainRow[j] ?? NaN)) {
            d2 += ((row[j] ?? 0) - (trainRow[j] ?? 0)) ** 2;
          } else if (Number.isNaN(trainRow[j] ?? NaN)) { valid = false; break; }
        }
        if (valid) dists.push({ dist: Math.sqrt(d2), row: trainRow });
      }
      dists.sort((a, b) => a.dist - b.dist);
      const neighbors = dists.slice(0, this.nNeighbors);

      return Float64Array.from({ length: p }, (_, j) => {
        if (!Number.isNaN(row[j] ?? NaN) && row[j] !== undefined) return row[j] ?? 0;
        if (neighbors.length === 0) return 0;
        if (this.weights === "uniform") {
          return neighbors.reduce((s, nb) => s + (nb.row[j] ?? 0), 0) / neighbors.length;
        }
        let ws = 0, wv = 0;
        for (const nb of neighbors) {
          const w = 1 / Math.max(nb.dist, 1e-12);
          ws += w; wv += w * (nb.row[j] ?? 0);
        }
        return ws > 0 ? wv / ws : 0;
      });
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
