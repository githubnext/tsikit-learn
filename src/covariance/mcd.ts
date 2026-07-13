/**
 * Minimum Covariance Determinant (MCD): robust covariance estimation
 */

export class MinCovDet {
  private support_fraction: number;
  private nSubsets: number;
  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  support_: Int32Array | null = null;

  constructor(support_fraction?: number, nSubsets = 500) {
    this.support_fraction = support_fraction ?? 0;
    this.nSubsets = nSubsets;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const h = Math.max(
      p + 1,
      Math.floor(
        (this.support_fraction > 0 ? this.support_fraction : (n + p + 1) / 2) *
          n,
      ),
    );

    // FastMCD approximation: multiple random subsets
    let bestDet = Number.POSITIVE_INFINITY;
    let bestSubset: number[] | null = null;

    for (let iter = 0; iter < Math.min(this.nSubsets, 500); iter++) {
      // Random initial subset of size p+1
      const subset = this.randomSubset(n, Math.min(p + 1, n));
      const expanded = this.expandSubset(X, subset, h);
      const { mean, cov } = this.computeMeanCov(X, expanded);
      const det = this.det(cov);
      if (det < bestDet) {
        bestDet = det;
        bestSubset = expanded;
      }
    }

    const finalSubset = bestSubset ?? Array.from({ length: h }, (_, i) => i);
    const { mean, cov } = this.computeMeanCov(X, finalSubset);

    this.location_ = mean;
    this.covariance_ = cov;
    this.precision_ = this.invertMatrix(cov);
    this.support_ = new Int32Array(n);
    for (const idx of finalSubset) this.support_[idx] = 1;
    return this;
  }

  private randomSubset(n: number, k: number): number[] {
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i]!;
      indices[i] = indices[j]!;
      indices[j] = tmp;
    }
    return indices.slice(0, k);
  }

  private expandSubset(
    X: Float64Array[],
    subset: number[],
    h: number,
  ): number[] {
    const { mean, cov } = this.computeMeanCov(X, subset);
    const prec = this.invertMatrix(cov);
    const dists = X.map((row, i) => ({
      i,
      d: this.mahalanobis(row, mean, prec),
    }));
    dists.sort((a, b) => a.d - b.d);
    return dists.slice(0, h).map((d) => d.i);
  }

  private mahalanobis(
    x: Float64Array,
    mean: Float64Array,
    prec: Float64Array[],
  ): number {
    const p = x.length;
    const diff = new Float64Array(p);
    for (let i = 0; i < p; i++) diff[i] = (x[i] ?? 0) - (mean[i] ?? 0);
    let dist = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++)
        dist += (diff[i] ?? 0) * (prec[i]![j] ?? 0) * (diff[j] ?? 0);
    }
    return dist;
  }

  private computeMeanCov(
    X: Float64Array[],
    indices: number[],
  ): { mean: Float64Array; cov: Float64Array[] } {
    const p = X[0]?.length ?? 0;
    const n = indices.length;
    const mean = new Float64Array(p);
    for (const idx of indices)
      for (let j = 0; j < p; j++) mean[j] += (X[idx]![j] ?? 0) / n;
    const cov: Float64Array[] = Array.from(
      { length: p },
      () => new Float64Array(p),
    );
    for (const idx of indices) {
      const diff = new Float64Array(p);
      for (let j = 0; j < p; j++) diff[j] = (X[idx]![j] ?? 0) - (mean[j] ?? 0);
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++)
          cov[i]![j] += ((diff[i] ?? 0) * (diff[j] ?? 0)) / (n - 1);
    }
    return { mean, cov };
  }

  private det(A: Float64Array[]): number {
    const n = A.length;
    if (n === 1) return A[0]![0] ?? 0;
    if (n === 2)
      return (
        (A[0]![0] ?? 0) * (A[1]![1] ?? 0) - (A[0]![1] ?? 0) * (A[1]![0] ?? 0)
      );
    let result = 1;
    const mat = A.map((row) => Float64Array.from(row));
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(mat[row]![col] ?? 0) > Math.abs(mat[maxRow]![col] ?? 0))
          maxRow = row;
      }
      if (maxRow !== col) {
        const tmp = mat[col]!;
        mat[col] = mat[maxRow]!;
        mat[maxRow] = tmp;
        result *= -1;
      }
      const pivot = mat[col]![col] ?? 0;
      if (Math.abs(pivot) < 1e-10) return 0;
      result *= pivot;
      for (let row = col + 1; row < n; row++) {
        const factor = (mat[row]![col] ?? 0) / pivot;
        for (let j = col; j < n; j++)
          mat[row]![j] = (mat[row]![j] ?? 0) - factor * (mat[col]![j] ?? 0);
      }
    }
    return result;
  }

  private invertMatrix(A: Float64Array[]): Float64Array[] {
    const n = A.length;
    const aug = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
      r[n + i] = 1;
      return r;
    });
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(aug[row]![col] ?? 0) > Math.abs(aug[maxRow]![col] ?? 0))
          maxRow = row;
      }
      const tmp = aug[col]!;
      aug[col] = aug[maxRow]!;
      aug[maxRow] = tmp;
      const pivot = aug[col]![col] ?? 1;
      for (let j = 0; j < 2 * n; j++)
        aug[col]![j] = (aug[col]![j] ?? 0) / (pivot || 1);
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = aug[row]![col] ?? 0;
        for (let j = 0; j < 2 * n; j++)
          aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
      }
    }
    return Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) row[j] = aug[i]![n + j] ?? 0;
      return row;
    });
  }

  mahalanobisDistances(X: Float64Array[]): Float64Array {
    if (!this.location_ || !this.precision_) throw new Error("Not fitted");
    return new Float64Array(
      X.map((row) => this.mahalanobis(row, this.location_!, this.precision_!)),
    );
  }
}
