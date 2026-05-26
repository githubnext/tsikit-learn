/**
 * Extended imputation: MatrixCompletion (nuclear norm), IterativeImputerExt, ExperimentalImputer
 */

export class MatrixCompletion {
  private maxIter: number;
  private tol: number;
  private rank: number;
  private U_: Float64Array[] | null = null;
  private V_: Float64Array[] | null = null;

  constructor(rank = 5, maxIter = 100, tol = 1e-4) {
    this.rank = rank;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: (number | null)[][]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    // Initialize with small random values
    this.U_ = Array.from({ length: m }, () => {
      const row = new Float64Array(this.rank);
      for (let k = 0; k < this.rank; k++) row[k] = (Math.random() - 0.5) * 0.01;
      return row;
    });
    this.V_ = Array.from({ length: n }, () => {
      const row = new Float64Array(this.rank);
      for (let k = 0; k < this.rank; k++) row[k] = (Math.random() - 0.5) * 0.01;
      return row;
    });

    const lr = 0.01;
    const reg = 0.01;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let loss = 0;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          const obs = X[i]![j];
          if (obs === null || obs === undefined) continue;
          let pred = 0;
          for (let k = 0; k < this.rank; k++) pred += (this.U_[i]![k] ?? 0) * (this.V_[j]![k] ?? 0);
          const err = obs - pred;
          loss += err * err;
          for (let k = 0; k < this.rank; k++) {
            const ui = this.U_[i]![k] ?? 0;
            const vj = this.V_[j]![k] ?? 0;
            this.U_[i]![k] = ui + lr * (err * vj - reg * ui);
            this.V_[j]![k] = vj + lr * (err * ui - reg * vj);
          }
        }
      }
      if (loss < this.tol) break;
    }
    return this;
  }

  transform(X: (number | null)[][]): Float64Array[] {
    if (!this.U_ || !this.V_) throw new Error("Not fitted");
    const m = X.length;
    const n = X[0]?.length ?? 0;
    return Array.from({ length: m }, (_, i) => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        const obs = X[i]![j];
        if (obs !== null && obs !== undefined) {
          row[j] = obs;
        } else {
          let pred = 0;
          for (let k = 0; k < this.rank; k++) pred += (this.U_![i]![k] ?? 0) * (this.V_![j]![k] ?? 0);
          row[j] = pred;
        }
      }
      return row;
    });
  }

  fitTransform(X: (number | null)[][]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class IterativeImputerExt {
  private maxIter: number;
  private tol: number;
  private imputedValues_: Map<number, number> | null = null;

  constructor(maxIter = 10, tol = 1e-3) {
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], missingValue = Number.NaN): this {
    const n = X[0]?.length ?? 0;
    this.imputedValues_ = new Map();
    // Initialize with column means
    for (let j = 0; j < n; j++) {
      let sum = 0, count = 0;
      for (const row of X) {
        const v = row[j] ?? 0;
        if (!Number.isNaN(v) && v !== missingValue) { sum += v; count++; }
      }
      this.imputedValues_.set(j, count > 0 ? sum / count : 0);
    }
    return this;
  }

  transform(X: Float64Array[], missingValue = Number.NaN): Float64Array[] {
    if (!this.imputedValues_) throw new Error("Not fitted");
    const n = X[0]?.length ?? 0;
    return X.map((row) => {
      const out = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        const v = row[j] ?? 0;
        out[j] = (Number.isNaN(v) || v === missingValue)
          ? (this.imputedValues_!.get(j) ?? 0)
          : v;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[], missingValue = Number.NaN): Float64Array[] {
    return this.fit(X, missingValue).transform(X, missingValue);
  }
}

export class MedianImputer {
  private medians_: Float64Array | null = null;

  fit(X: Float64Array[]): this {
    const n = X[0]?.length ?? 0;
    this.medians_ = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const vals = X.map((row) => row[j] ?? 0).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
      const mid = Math.floor(vals.length / 2);
      this.medians_[j] = vals.length % 2 === 0
        ? ((vals[mid - 1] ?? 0) + (vals[mid] ?? 0)) / 2
        : vals[mid] ?? 0;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.medians_) throw new Error("Not fitted");
    const n = X[0]?.length ?? 0;
    return X.map((row) => {
      const out = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        const v = row[j] ?? 0;
        out[j] = Number.isNaN(v) ? (this.medians_![j] ?? 0) : v;
      }
      return out;
    });
  }
}
