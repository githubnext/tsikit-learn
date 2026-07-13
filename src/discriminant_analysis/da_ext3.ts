/**
 * Discriminant Analysis extensions — flexible LDA/QDA with shrinkage.
 */

function computeMeanAndCov(X: Float64Array[]): { mean: Float64Array; cov: Float64Array[] } {
  const n = X.length, p = X[0]?.length ?? 0;
  const mean = new Float64Array(p);
  for (const row of X) for (let j = 0; j < p; j++) mean[j]! += (row[j] ?? 0) / n;
  const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
  for (const row of X) {
    for (let j = 0; j < p; j++) {
      const dj = (row[j] ?? 0) - (mean[j] ?? 0);
      for (let k = 0; k < p; k++) (cov[j]! as Float64Array)[k]! += dj * ((row[k] ?? 0) - (mean[k] ?? 0)) / Math.max(n - 1, 1);
    }
  }
  return { mean, cov };
}

function logDet(M: Float64Array[]): number {
  const p = M.length;
  return M.reduce((s, row, i) => s + Math.log(Math.max((row as Float64Array)[i] ?? 1e-12, 1e-12)), 0) * 2; // approximation via diagonal
}

function mahalanobisFromCov(x: Float64Array, mean: Float64Array, cov: Float64Array[]): number {
  const p = x.length;
  const diff = Float64Array.from({ length: p }, (_, i) => (x[i] ?? 0) - (mean[i] ?? 0));
  return diff.reduce((s, dj, j) => {
    const covDiag = (cov[j] as Float64Array)[j] ?? 1;
    return s + dj * dj / Math.max(covDiag, 1e-12);
  }, 0);
}

export class FlexibleLDA {
  solver: "svd" | "lsqr" | "eigen";
  shrinkage: "auto" | number | null;
  nComponents: number | null;
  regularization: number;
  private _classMeans: Float64Array[] = [];
  private _classCovs: Float64Array[][] = [];
  private _classPriors: Float64Array | null = null;
  private _pooledCov: Float64Array[] | null = null;
  private _scalings: Float64Array[] | null = null;
  nClasses_: number = 0;
  nFeaturesIn_: number = 0;
  classes_: Int32Array | null = null;

  constructor(
    solver: "svd" | "lsqr" | "eigen" = "svd",
    shrinkage: "auto" | number | null = null,
    nComponents: number | null = null,
    regularization = 0.0,
  ) {
    this.solver = solver;
    this.shrinkage = shrinkage;
    this.nComponents = nComponents;
    this.regularization = regularization;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const uniqueClasses = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = Int32Array.from(uniqueClasses);
    this.nClasses_ = uniqueClasses.length;

    const classCounts = uniqueClasses.map((c) => Array.from(y).filter((yi) => yi === c).length);
    this._classPriors = Float64Array.from(classCounts, (c) => c / n);

    this._classMeans = [];
    this._classCovs = [];
    const pooledCov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));

    for (let ci = 0; ci < uniqueClasses.length; ci++) {
      const c = uniqueClasses[ci] as number;
      const Xc = X.filter((_, i) => y[i] === c);
      const { mean, cov } = computeMeanAndCov(Xc);
      this._classMeans.push(mean);

      // Apply shrinkage
      const shrinkCov = cov.map((row, i) => row.map((v, j) => {
        const shrink = this.shrinkage === null ? 0 : this.shrinkage === "auto" ? 1 / (n * p) : this.shrinkage;
        return i === j ? v * (1 - shrink) + shrink * cov.reduce((s, r) => s + (r as Float64Array)[i]!, 0) / p : v * (1 - shrink);
      }));
      this._classCovs.push(shrinkCov);

      // Accumulate pooled covariance
      const wc = classCounts[ci] ?? 0;
      for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) (pooledCov[j]! as Float64Array)[k]! += (wc / n) * ((shrinkCov[j] as Float64Array)[k] ?? 0);
    }

    // Add regularization
    for (let j = 0; j < p; j++) (pooledCov[j]! as Float64Array)[j]! += this.regularization;
    this._pooledCov = pooledCov;

    // Compute scalings (whitening transform of pooled cov, approximated by diagonal)
    const diagStd = Float64Array.from({ length: p }, (_, j) => Math.sqrt(Math.max((pooledCov[j] as Float64Array)[j] ?? 1e-12, 1e-12)));
    const k = this.nComponents ?? this.nClasses_ - 1;
    this._scalings = Array.from({ length: p }, (_, j) => {
      const row = new Float64Array(k);
      row[j % k] = 1 / (diagStd[j] ?? 1);
      return row;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this._scalings || !this._pooledCov) throw new Error("Not fitted");
    const k = this.nComponents ?? this.nClasses_ - 1;
    return X.map((x) => {
      return Float64Array.from({ length: k }, (_, comp) => {
        return (this._scalings as Float64Array[]).reduce((s, scale, j) => s + (scale[comp] ?? 0) * (x[j] ?? 0), 0);
      });
    });
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this._classMeans || !this._classPriors) throw new Error("Not fitted");
    return Int32Array.from(X, (x) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let ci = 0; ci < this.nClasses_; ci++) {
        const logPrior = Math.log(Math.max((this._classPriors as Float64Array)[ci] ?? 1e-15, 1e-15));
        const dist = mahalanobisFromCov(x, this._classMeans[ci] as Float64Array, this._pooledCov as Float64Array[]);
        const score = logPrior - 0.5 * dist;
        if (score > bestScore) { bestScore = score; best = this.classes_?.[ci] ?? ci; }
      }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}

export class ReguarizedQDA {
  regParam: number;
  private _classMeans: Float64Array[] = [];
  private _classCovs: Float64Array[][] = [];
  private _classPriors: Float64Array | null = null;
  private _logDets: Float64Array | null = null;
  nClasses_: number = 0;
  classes_: Int32Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(regParam = 0.0) {
    this.regParam = regParam;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const uniqueClasses = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = Int32Array.from(uniqueClasses);
    this.nClasses_ = uniqueClasses.length;

    const classCounts = uniqueClasses.map((c) => Array.from(y).filter((yi) => yi === c).length);
    this._classPriors = Float64Array.from(classCounts, (c) => c / n);
    this._classMeans = [];
    this._classCovs = [];
    const logDets: number[] = [];

    for (let ci = 0; ci < uniqueClasses.length; ci++) {
      const c = uniqueClasses[ci] as number;
      const Xc = X.filter((_, i) => y[i] === c);
      const { mean, cov } = computeMeanAndCov(Xc);
      this._classMeans.push(mean);
      // Regularize class covariance
      const regCov = cov.map((row, i) => row.map((v, j) => i === j ? v + this.regParam : v));
      this._classCovs.push(regCov);
      logDets.push(logDet(regCov));
    }
    this._logDets = Float64Array.from(logDets);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this._classMeans || !this._classPriors || !this._logDets) throw new Error("Not fitted");
    return Int32Array.from(X, (x) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let ci = 0; ci < this.nClasses_; ci++) {
        const logPrior = Math.log(Math.max((this._classPriors as Float64Array)[ci] ?? 1e-15, 1e-15));
        const dist = mahalanobisFromCov(x, this._classMeans[ci] as Float64Array, this._classCovs[ci] as Float64Array[]);
        const score = logPrior - 0.5 * dist - 0.5 * (this._logDets![ci] ?? 0);
        if (score > bestScore) { bestScore = score; best = this.classes_?.[ci] ?? ci; }
      }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}
