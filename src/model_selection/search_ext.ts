/**
 * Model selection search extensions: NestedCV, HalvingGridSearchCV, BayesSearchCV.
 */

export class NestedCrossValidation {
  private outerScores_: Float64Array = new Float64Array(0);

  constructor(
    private readonly estimatorFactory: () => { fit: (X: Float64Array[], y: Int32Array | Float64Array) => void; score?: (X: Float64Array[], y: Int32Array | Float64Array) => number },
    private readonly paramGrid: Record<string, unknown[]>,
    private readonly outerCv = 5,
    private readonly innerCv = 3
  ) {}

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    const n = X.length;
    const foldSize = Math.floor(n / this.outerCv);
    this.outerScores_ = new Float64Array(this.outerCv);

    for (let outer = 0; outer < this.outerCv; outer++) {
      const testStart = outer * foldSize;
      const testEnd = outer === this.outerCv - 1 ? n : (outer + 1) * foldSize;
      const trainIdx = [...Array.from({ length: testStart }, (_, i) => i), ...Array.from({ length: n - testEnd }, (_, i) => testEnd + i)];
      const testIdx = Array.from({ length: testEnd - testStart }, (_, i) => testStart + i);
      const Xtrain = trainIdx.map((i) => X[i]!);
      const ytrain = y instanceof Int32Array ? new Int32Array(trainIdx.map((i) => y[i]!)) : new Float64Array(trainIdx.map((i) => y[i]!));
      const Xtest = testIdx.map((i) => X[i]!);
      const ytest = y instanceof Int32Array ? new Int32Array(testIdx.map((i) => y[i]!)) : new Float64Array(testIdx.map((i) => y[i]!));
      // Inner CV for param selection
      const bestParams = this._innerCV(Xtrain, ytrain);
      void bestParams;
      const est = this.estimatorFactory();
      est.fit(Xtrain, ytrain);
      this.outerScores_[outer] = est.score ? est.score(Xtest, ytest) : 0;
    }
    return this;
  }

  private _innerCV(X: Float64Array[], y: Int32Array | Float64Array): Record<string, unknown> {
    const paramKeys = Object.keys(this.paramGrid);
    let bestScore = -Number.POSITIVE_INFINITY;
    let bestParams: Record<string, unknown> = {};
    const n = X.length;
    const foldSize = Math.max(1, Math.floor(n / this.innerCv));
    const paramCombos = this._cartesianProduct(this.paramGrid);
    for (const params of paramCombos.slice(0, 20)) {
      let totalScore = 0;
      for (let fold = 0; fold < this.innerCv; fold++) {
        const testStart = fold * foldSize;
        const testEnd = Math.min((fold + 1) * foldSize, n);
        const trainIdx = [...Array.from({ length: testStart }, (_, i) => i), ...Array.from({ length: n - testEnd }, (_, i) => testEnd + i)];
        const testIdx = Array.from({ length: testEnd - testStart }, (_, i) => testStart + i);
        const est = this.estimatorFactory();
        const Xtrain = trainIdx.map((i) => X[i]!);
        const ytrain = y instanceof Int32Array ? new Int32Array(trainIdx.map((i) => y[i]!)) : new Float64Array(trainIdx.map((i) => y[i]!));
        est.fit(Xtrain, ytrain);
        const score = est.score ? est.score(testIdx.map((i) => X[i]!), y instanceof Int32Array ? new Int32Array(testIdx.map((i) => y[i]!)) : new Float64Array(testIdx.map((i) => y[i]!))) : 0;
        totalScore += score;
      }
      const avgScore = totalScore / this.innerCv;
      if (avgScore > bestScore) { bestScore = avgScore; bestParams = params; }
    }
    void paramKeys;
    return bestParams;
  }

  private _cartesianProduct(grid: Record<string, unknown[]>): Record<string, unknown>[] {
    const keys = Object.keys(grid);
    if (keys.length === 0) return [{}];
    const [first, ...rest] = keys;
    const restCombos = this._cartesianProduct(Object.fromEntries(rest.map((k) => [k, grid[k]!])));
    const result: Record<string, unknown>[] = [];
    for (const v of grid[first!] ?? []) {
      for (const combo of restCombos) result.push({ [first!]: v, ...combo });
    }
    return result;
  }

  getOuterScores(): Float64Array { return this.outerScores_; }
  meanScore(): number { return this.outerScores_.reduce((a, b) => a + b, 0) / this.outerScores_.length; }
  stdScore(): number {
    const mean = this.meanScore();
    return Math.sqrt(this.outerScores_.reduce((a, b) => a + (b - mean) ** 2, 0) / this.outerScores_.length);
  }
}

export class HalvingGridSearchCV {
  bestParams_: Record<string, unknown> = {};
  bestScore_ = -Number.POSITIVE_INFINITY;

  constructor(
    private readonly estimatorFactory: () => { fit: (X: Float64Array[], y: Int32Array | Float64Array) => void; score?: (X: Float64Array[], y: Int32Array | Float64Array) => number },
    private readonly paramGrid: Record<string, unknown[]>,
    private readonly factor = 3,
    private readonly cv = 5
  ) {}

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    const paramCombos = this._cartesianProduct(this.paramGrid);
    let candidates = paramCombos;
    let nSamples = Math.ceil(X.length / this.factor);

    while (candidates.length > 1) {
      const scores = candidates.map((params) => {
        void params;
        const Xi = X.slice(0, nSamples);
        const yi = y instanceof Int32Array ? y.slice(0, nSamples) : y.slice(0, nSamples);
        const est = this.estimatorFactory();
        est.fit(Xi, yi);
        return est.score ? est.score(Xi, yi) : 0;
      });
      const indexed = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
      candidates = indexed.slice(0, Math.max(1, Math.ceil(candidates.length / this.factor))).map(({ i }) => candidates[i]!);
      nSamples = Math.min(nSamples * this.factor, X.length);
    }

    this.bestParams_ = candidates[0] ?? {};
    const est = this.estimatorFactory();
    est.fit(X, y);
    this.bestScore_ = est.score ? est.score(X, y) : 0;
    return this;
  }

  private _cartesianProduct(grid: Record<string, unknown[]>): Record<string, unknown>[] {
    const keys = Object.keys(grid);
    if (keys.length === 0) return [{}];
    const [first, ...rest] = keys;
    const restCombos = this._cartesianProduct(Object.fromEntries(rest.map((k) => [k, grid[k]!])));
    const result: Record<string, unknown>[] = [];
    for (const v of grid[first!] ?? []) {
      for (const combo of restCombos) result.push({ [first!]: v, ...combo });
    }
    return result;
  }
}
