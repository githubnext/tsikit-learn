/**
 * Locally Weighted Regression (LOESS) and Weighted KNN extensions.
 */

export class LOESSRegressor {
  private XTrain_!: Float64Array[];
  private yTrain_!: Float64Array;
  private fitted_ = false;

  constructor(private bandwidth = 0.75, private degree = 1, private nIter = 3) {}

  fit(X: Float64Array[], y: Float64Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    this.fitted_ = true;
    return this;
  }

  private _tricubeKernel(u: number): number {
    return Math.abs(u) < 1 ? Math.pow(1 - Math.pow(Math.abs(u), 3), 3) : 0;
  }

  private _predictOne(x: Float64Array): number {
    const n = this.XTrain_.length;
    const dists = this.XTrain_.map(xi => Math.sqrt(xi.reduce((s, v, j) => s + (v - (x[j] ?? 0)) ** 2, 0)));
    const sorted = dists.map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d);
    const nNeigh = Math.max(2, Math.floor(n * this.bandwidth));
    const h = sorted[nNeigh - 1]?.d ?? 1;
    let weights = sorted.slice(0, nNeigh).map(({ d, i }) => ({ w: this._tricubeKernel(d / (h + 1e-10)), i }));

    // Robustness iterations
    let yHat = new Float64Array(n);
    for (let iter = 0; iter < this.nIter; iter++) {
      // Weighted least squares at x
      const wx = weights.map(({ w }) => w);
      const xMat = weights.map(({ i }) => this.XTrain_[i]!);
      const yVec = new Float64Array(weights.map(({ i }) => this.yTrain_[i] ?? 0));
      // Build augmented X with polynomial terms of degree 1
      const XA = xMat.map((xi, r) => new Float64Array([1, ...xi]));
      const p = XA[0]?.length ?? 1;
      // Weighted OLS: (X^T W X)^{-1} X^T W y
      const XtWX = Array.from({ length: p }, (_, i) =>
        new Float64Array(p).map((_, j) => XA.reduce((s, row, r) => s + (wx[r] ?? 0) * (row[i] ?? 0) * (row[j] ?? 0), 0))
      );
      const XtWy = new Float64Array(p).map((_, j) => XA.reduce((s, row, r) => s + (wx[r] ?? 0) * (row[j] ?? 0) * (yVec[r] ?? 0), 0));
      for (let i = 0; i < p; i++) XtWX[i]![i] = (XtWX[i]![i] ?? 0) + 1e-8;
      const augM = XtWX.map((row, i) => [...row, XtWy[i] ?? 0]);
      for (let col = 0; col < p; col++) {
        const piv = augM[col]![col] ?? 1;
        for (let j = col; j <= p; j++) augM[col]![j] = (augM[col]![j] ?? 0) / piv;
        for (let row = 0; row < p; row++) {
          if (row === col) continue;
          const f = augM[row]![col] ?? 0;
          for (let j = col; j <= p; j++) augM[row]![j] = (augM[row]![j] ?? 0) - f * (augM[col]![j] ?? 0);
        }
      }
      const beta = augM.map(row => row[p] ?? 0);
      const xAug = new Float64Array([1, ...x]);
      const fitVal = xAug.reduce((s, v, j) => s + v * (beta[j] ?? 0), 0);
      // Compute residuals for robustness weights
      for (let r = 0; r < weights.length; r++) {
        const { i } = weights[r]!;
        yHat[i] = xMat[r]!.reduce((s, v, j) => s + (beta[j + 1] ?? 0) * v, beta[0] ?? 0);
      }
      const resids = weights.map(({ i }) => Math.abs((this.yTrain_[i] ?? 0) - (yHat[i] ?? 0)));
      const medR = [...resids].sort((a, b) => a - b)[Math.floor(resids.length / 2)] ?? 1;
      const bisquare = (u: number) => Math.abs(u) < 1 ? (1 - u * u) ** 2 : 0;
      weights = weights.map(({ w, i }, r) => ({ w: w * bisquare(resids[r]! / (6 * medR + 1e-10)), i }));
      if (iter === this.nIter - 1) return fitVal;
    }
    return 0;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(x => this._predictOne(x)));
  }
}

export class WeightedKNNClassifier {
  private XTrain_!: Float64Array[];
  private yTrain_!: Int32Array;
  private fitted_ = false;

  constructor(private nNeighbors = 5, private weights: 'uniform' | 'distance' = 'distance') {}

  fit(X: Float64Array[], y: Int32Array): this {
    this.XTrain_ = X;
    this.yTrain_ = y;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const classes = Array.from(new Set(Array.from(this.yTrain_)));
    return new Int32Array(X.map(x => {
      const dists = this.XTrain_.map((xi, i) => ({ d: Math.sqrt(xi.reduce((s, v, j) => s + (v - (x[j] ?? 0)) ** 2, 0)), i }));
      const knn = dists.sort((a, b) => a.d - b.d).slice(0, this.nNeighbors);
      const votes = new Map<number, number>(classes.map(c => [c, 0]));
      for (const { d, i } of knn) {
        const w = this.weights === 'distance' ? 1 / (d + 1e-10) : 1;
        const cls = this.yTrain_[i] ?? 0;
        votes.set(cls, (votes.get(cls) ?? 0) + w);
      }
      return [...votes.entries()].reduce((best, [c, v]) => v > best.v ? { c, v } : best, { c: 0, v: -1 }).c;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    return pred.filter((v, i) => v === y[i]).length / pred.length;
  }
}
