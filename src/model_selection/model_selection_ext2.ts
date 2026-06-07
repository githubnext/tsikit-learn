/**
 * Successive Halving and Hyperband model selection extensions.
 */

export interface BaseEstimatorForSearch {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
}

export class HalvingGridSearchCV {
  estimator: BaseEstimatorForSearch;
  paramGrid: Record<string, unknown[]>;
  factor: number;
  minResources: number | "exhaust";
  maxResources: number | "auto";
  aggressive_elimination: boolean;
  cv: number;
  scoring: string;
  bestParams_: Record<string, unknown> | null = null;
  bestScore_: number = Number.NEGATIVE_INFINITY;
  bestEstimator_: BaseEstimatorForSearch | null = null;
  cvResults_: Array<Record<string, unknown>> = [];
  nIterations_: number = 0;
  nPossibleIterations_: number = 0;
  nRequiredIterations_: number = 0;
  nRemainingCandidates_: number = 0;

  constructor(
    estimator: BaseEstimatorForSearch,
    paramGrid: Record<string, unknown[]>,
    factor = 3,
    minResources: number | "exhaust" = "exhaust",
    maxResources: number | "auto" = "auto",
    aggressiveElimination = false,
    cv = 5,
    scoring = "accuracy",
  ) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.factor = factor;
    this.minResources = minResources;
    this.maxResources = maxResources;
    this.aggressive_elimination = aggressiveElimination;
    this.cv = cv;
    this.scoring = scoring;
  }

  private _gridProduct(grid: Record<string, unknown[]>): Array<Record<string, unknown>> {
    const keys = Object.keys(grid);
    const result: Array<Record<string, unknown>> = [{}];
    for (const key of keys) {
      const values = grid[key] ?? [];
      const expanded: Array<Record<string, unknown>> = [];
      for (const r of result) {
        for (const v of values) {
          expanded.push({ ...r, [key]: v });
        }
      }
      result.splice(0, result.length, ...expanded);
    }
    return result;
  }

  private _applyParams(params: Record<string, unknown>): BaseEstimatorForSearch {
    const clone = Object.create(Object.getPrototypeOf(this.estimator));
    Object.assign(clone, this.estimator);
    for (const [k, v] of Object.entries(params)) {
      (clone as Record<string, unknown>)[k] = v;
    }
    return clone as BaseEstimatorForSearch;
  }

  private _cvScore(estimator: BaseEstimatorForSearch, X: Float64Array[], y: Float64Array | Int32Array, nSamples: number): number {
    const n = Math.min(nSamples, X.length);
    const foldSize = Math.floor(n / this.cv);
    let totalScore = 0;
    for (let fold = 0; fold < this.cv; fold++) {
      const valStart = fold * foldSize;
      const valEnd = Math.min(valStart + foldSize, n);
      const trainX: Float64Array[] = [];
      const trainY: number[] = [];
      const valX: Float64Array[] = [];
      const valY: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i >= valStart && i < valEnd) {
          valX.push(X[i] as Float64Array);
          valY.push(y instanceof Float64Array ? (y[i] ?? 0) : (y[i] ?? 0));
        } else {
          trainX.push(X[i] as Float64Array);
          trainY.push(y instanceof Float64Array ? (y[i] ?? 0) : (y[i] ?? 0));
        }
      }
      if (trainX.length === 0 || valX.length === 0) continue;
      const trainYArr = y instanceof Float64Array ? Float64Array.from(trainY) : Int32Array.from(trainY);
      const valYArr = y instanceof Float64Array ? Float64Array.from(valY) : Int32Array.from(valY);
      try {
        estimator.fit(trainX, trainYArr);
        totalScore += estimator.score(valX, valYArr);
      } catch {
        totalScore += Number.NEGATIVE_INFINITY;
      }
    }
    return totalScore / this.cv;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    let candidates = this._gridProduct(this.paramGrid);
    const maxRes = this.maxResources === "auto" ? n : this.maxResources;
    const minRes = this.minResources === "exhaust"
      ? Math.max(1, Math.floor(n / this.factor ** Math.ceil(Math.log(candidates.length) / Math.log(this.factor))))
      : this.minResources;

    let resources = minRes;
    this.nIterations_ = 0;
    this.cvResults_ = [];

    while (candidates.length > 1 && resources <= maxRes) {
      const nSamples = Math.min(resources, n);
      const scores = candidates.map((params) => {
        const estimator = this._applyParams(params);
        return this._cvScore(estimator, X, y, nSamples);
      });
      candidates = candidates
        .map((params, i) => ({ params, score: scores[i] ?? Number.NEGATIVE_INFINITY }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.ceil(candidates.length / this.factor))
        .map((c) => c.params);
      resources = Math.min(resources * this.factor, maxRes);
      this.nIterations_++;
    }

    // Evaluate final candidate
    if (candidates.length > 0) {
      const finalParams = candidates[0] as Record<string, unknown>;
      const est = this._applyParams(finalParams);
      const score = this._cvScore(est, X, y, n);
      this.bestParams_ = finalParams;
      this.bestScore_ = score;
      est.fit(X, y);
      this.bestEstimator_ = est;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    if (!this.bestEstimator_) throw new Error("Not fitted");
    return (this.bestEstimator_ as unknown as { predict: (X: Float64Array[]) => Float64Array | Int32Array }).predict(X);
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    if (!this.bestEstimator_) throw new Error("Not fitted");
    return this.bestEstimator_.score(X, y);
  }
}

export class HalvingRandomSearchCV extends HalvingGridSearchCV {
  nCandidates: number;
  randomState: number;

  constructor(
    estimator: BaseEstimatorForSearch,
    paramDistributions: Record<string, unknown[]>,
    nCandidates = 50,
    factor = 3,
    minResources: number | "exhaust" = "exhaust",
    maxResources: number | "auto" = "auto",
    cv = 5,
    randomState = 42,
  ) {
    super(estimator, paramDistributions, factor, minResources, maxResources, false, cv);
    this.nCandidates = nCandidates;
    this.randomState = randomState;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    // Sample random combinations
    const keys = Object.keys(this.paramGrid);
    const sampled: Array<Record<string, unknown>> = [];
    for (let i = 0; i < this.nCandidates; i++) {
      const params: Record<string, unknown> = {};
      for (const key of keys) {
        const values = this.paramGrid[key] ?? [];
        params[key] = values[Math.floor(Math.random() * values.length)];
      }
      sampled.push(params);
    }
    this.paramGrid = {};
    const tempGrid: Record<string, unknown[]> = {};
    for (const key of keys) {
      tempGrid[key] = sampled.map((s) => s[key]);
    }
    this.paramGrid = tempGrid;
    return super.fit(X, y);
  }
}
