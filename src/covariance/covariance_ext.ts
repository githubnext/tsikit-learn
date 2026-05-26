/**
 * Covariance extensions: OAS estimator, LedoitWolf estimator, ShrunkCovariance.
 */

export class OASCovariance {
  covariance_: Float64Array[] = [];
  precision_: Float64Array[] = [];
  shrinkage_ = 0;

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const emp = this._empiricalCovariance(X);
    // OAS shrinkage estimator
    const trS = emp.reduce((s, row, i) => s + (row[i] ?? 0), 0);
    const trS2 = emp.reduce((s1, row) => s1 + row.reduce((s2, v) => s2 + v * v, 0), 0);
    const mu = trS / p;
    const rhoNum = (1 - 2 / p) * trS2 + trS ** 2;
    const rhoDenom = (n + 1 - 2 / p) * (trS2 - trS ** 2 / p);
    const rho = Math.min(1, rhoNum / Math.max(rhoDenom, 1e-10));
    this.shrinkage_ = rho;
    this.covariance_ = emp.map((row, i) => new Float64Array(row.map((v, j) => (1 - rho) * v + (i === j ? rho * mu : 0))));
    this.precision_ = this._invertMatrix(this.covariance_);
    return this;
  }

  private _empiricalCovariance(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const mean = new Float64Array(p);
    for (const x of X) for (let f = 0; f < p; f++) mean[f] = (mean[f] ?? 0) + (x[f] ?? 0) / n;
    const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    for (const x of X) {
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
        cov[i]![j] = (cov[i]![j] ?? 0) + ((x[i] ?? 0) - (mean[i] ?? 0)) * ((x[j] ?? 0) - (mean[j] ?? 0)) / n;
      }
    }
    return cov;
  }

  private _invertMatrix(M: Float64Array[]): Float64Array[] {
    const n = M.length;
    const A = M.map((row) => new Float64Array(row));
    const inv = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      row[i] = 1;
      return row;
    });
    for (let col = 0; col < n; col++) {
      let pivotRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(A[row]?.[col] ?? 0) > Math.abs(A[pivotRow]?.[col] ?? 0)) pivotRow = row;
      }
      [A[col], A[pivotRow]] = [A[pivotRow]!, A[col]!];
      [inv[col], inv[pivotRow]] = [inv[pivotRow]!, inv[col]!];
      const pivot = A[col]?.[col] ?? 1e-10;
      if (Math.abs(pivot) < 1e-10) continue;
      for (let j = 0; j < n; j++) { A[col]![j] = (A[col]![j] ?? 0) / pivot; inv[col]![j] = (inv[col]![j] ?? 0) / pivot; }
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = A[row]?.[col] ?? 0;
        for (let j = 0; j < n; j++) {
          A[row]![j] = (A[row]![j] ?? 0) - factor * (A[col]![j] ?? 0);
          inv[row]![j] = (inv[row]![j] ?? 0) - factor * (inv[col]![j] ?? 0);
        }
      }
    }
    return inv;
  }
}

export class LedoitWolfCovariance {
  covariance_: Float64Array[] = [];
  shrinkage_ = 0;
  precision_: Float64Array[] = [];

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const emp = this._empiricalCovariance(X);
    const trS = emp.reduce((s, row, i) => s + (row[i] ?? 0), 0);
    const mu = trS / p;
    const delta = emp.reduce((s1, row, i) => s1 + row.reduce((s2, v, j) => s2 + (i === j ? (v - mu) ** 2 : v ** 2), 0), 0) / p;
    const beta = 1 / (n * p) * emp.reduce((s1, row) => s1 + row.reduce((s2, v) => s2 + v ** 2, 0), 0);
    const rho = Math.min(1, (beta - delta) / Math.max(delta, 1e-10));
    this.shrinkage_ = rho;
    this.covariance_ = emp.map((row, i) => new Float64Array(row.map((v, j) => (1 - rho) * v + (i === j ? rho * mu : 0))));
    this.precision_ = this._invertMatrix(this.covariance_);
    return this;
  }

  private _empiricalCovariance(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const mean = new Float64Array(p);
    for (const x of X) for (let f = 0; f < p; f++) mean[f] = (mean[f] ?? 0) + (x[f] ?? 0) / n;
    const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    for (const x of X) {
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
        cov[i]![j] = (cov[i]![j] ?? 0) + ((x[i] ?? 0) - (mean[i] ?? 0)) * ((x[j] ?? 0) - (mean[j] ?? 0)) / n;
      }
    }
    return cov;
  }

  private _invertMatrix(M: Float64Array[]): Float64Array[] {
    const n = M.length;
    const A = M.map((row) => new Float64Array(row));
    const inv = Array.from({ length: n }, (_, i) => { const row = new Float64Array(n); row[i] = 1; return row; });
    for (let col = 0; col < n; col++) {
      const pivot = A[col]?.[col] ?? 1e-10;
      if (Math.abs(pivot) < 1e-10) continue;
      for (let j = 0; j < n; j++) { A[col]![j] = (A[col]![j] ?? 0) / pivot; inv[col]![j] = (inv[col]![j] ?? 0) / pivot; }
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = A[row]?.[col] ?? 0;
        for (let j = 0; j < n; j++) { A[row]![j] = (A[row]![j] ?? 0) - f * (A[col]![j] ?? 0); inv[row]![j] = (inv[row]![j] ?? 0) - f * (inv[col]![j] ?? 0); }
      }
    }
    return inv;
  }
}

export class ShrunkCovariance {
  covariance_: Float64Array[] = [];
  precision_: Float64Array[] = [];

  constructor(private readonly shrinkage = 0.1) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const mean = new Float64Array(p);
    for (const x of X) for (let f = 0; f < p; f++) mean[f] = (mean[f] ?? 0) + (x[f] ?? 0) / n;
    const emp: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    for (const x of X) {
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
        emp[i]![j] = (emp[i]![j] ?? 0) + ((x[i] ?? 0) - (mean[i] ?? 0)) * ((x[j] ?? 0) - (mean[j] ?? 0)) / n;
      }
    }
    const mu = emp.reduce((s, row, i) => s + (row[i] ?? 0), 0) / p;
    this.covariance_ = emp.map((row, i) => new Float64Array(row.map((v, j) => (1 - this.shrinkage) * v + (i === j ? this.shrinkage * mu : 0))));
    // Simple precision (diagonal approximation)
    this.precision_ = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = 1 / Math.max(this.covariance_[i]?.[i] ?? 1, 1e-10);
      return row;
    });
    return this;
  }
}
