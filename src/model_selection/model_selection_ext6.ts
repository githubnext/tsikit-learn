/**
 * Extended model selection — successive halving and custom CV strategies.
 */

export interface CVEstimator {
  fit(X: Float64Array[], y: Float64Array | Int32Array): this;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
  setParams(params: Record<string, unknown>): this;
}

export interface HalvingSearchResult {
  bestParams_: Record<string, unknown>;
  bestScore_: number;
  bestEstimator_: CVEstimator;
  cvResults_: Array<{ params: Record<string, unknown>; meanTestScore: number; nResources: number; iter: number }>;
}

export class HalvingGridSearchCVExt implements HalvingSearchResult {
  estimator: CVEstimator;
  paramGrid: Record<string, unknown[]>;
  factor: number;
  minResources: number | "smallest";
  maxResources: number | "auto";
  cv: number;
  scoring: ((est: CVEstimator, X: Float64Array[], y: Float64Array | Int32Array) => number) | null;
  bestParams_: Record<string, unknown> = {};
  bestScore_: number = -Number.POSITIVE_INFINITY;
  bestEstimator_: CVEstimator;
  cvResults_: Array<{ params: Record<string, unknown>; meanTestScore: number; nResources: number; iter: number }> = [];
  nCandidates_: number[] = [];
  nResources_: number[] = [];

  constructor(
    estimator: CVEstimator,
    paramGrid: Record<string, unknown[]>,
    factor = 3,
    minResources: number | "smallest" = "smallest",
    maxResources: number | "auto" = "auto",
    cv = 5,
    scoring: ((est: CVEstimator, X: Float64Array[], y: Float64Array | Int32Array) => number) | null = null,
  ) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.factor = factor;
    this.minResources = minResources;
    this.maxResources = maxResources;
    this.cv = cv;
    this.scoring = scoring;
    this.bestEstimator_ = estimator;
  }

  private _gridExpand(): Array<Record<string, unknown>> {
    const keys = Object.keys(this.paramGrid);
    const values = keys.map((k) => this.paramGrid[k] as unknown[]);
    const combinations: Array<Record<string, unknown>> = [{}];
    for (let ki = 0; ki < keys.length; ki++) {
      const newCombinations: Array<Record<string, unknown>> = [];
      for (const combo of combinations) {
        for (const val of values[ki] ?? []) {
          newCombinations.push({ ...combo, [keys[ki] as string]: val });
        }
      }
      combinations.length = 0;
      combinations.push(...newCombinations);
    }
    return combinations;
  }

  private _cvScore(params: Record<string, unknown>, X: Float64Array[], y: Float64Array | Int32Array, nResources: number): number {
    const est = this.estimator.setParams(params);
    const nSub = Math.min(nResources, X.length);
    const scores: number[] = [];
    const foldSize = Math.max(1, Math.floor(nSub / this.cv));
    for (let fold = 0; fold < this.cv; fold++) {
      const valStart = fold * foldSize;
      const valEnd = Math.min(valStart + foldSize, nSub);
      const XSub = X.slice(0, nSub);
      const ySub = y instanceof Float64Array ? y.slice(0, nSub) : y.slice(0, nSub);
      const XTrain = [...XSub.slice(0, valStart), ...XSub.slice(valEnd)];
      const yTrain = y instanceof Float64Array
        ? Float64Array.from([...Array.from(ySub.slice(0, valStart)), ...Array.from(ySub.slice(valEnd))])
        : Int32Array.from([...Array.from(ySub.slice(0, valStart)), ...Array.from(ySub.slice(valEnd))]);
      const XVal = XSub.slice(valStart, valEnd);
      const yVal = y instanceof Float64Array ? ySub.slice(valStart, valEnd) : ySub.slice(valStart, valEnd);
      if (XTrain.length === 0 || XVal.length === 0) continue;
      est.fit(XTrain, yTrain);
      const score = this.scoring ? this.scoring(est, XVal, yVal) : est.score(XVal, yVal);
      scores.push(score);
    }
    return scores.length > 0 ? scores.reduce((s, v) => s + v, 0) / scores.length : 0;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    const maxRes = this.maxResources === "auto" ? n : this.maxResources;
    const minRes = this.minResources === "smallest" ? Math.max(1, Math.floor(n / 10)) : this.minResources;

    let candidates = this._gridExpand();
    let resources = minRes;
    let iter = 0;

    while (candidates.length > 1 && resources <= maxRes) {
      const scores = candidates.map((params) => ({
        params,
        score: this._cvScore(params, X, y, resources),
        nResources: resources,
        iter,
      }));
      scores.sort((a, b) => b.score - a.score);

      for (const s of scores) {
        this.cvResults_.push({ params: s.params, meanTestScore: s.score, nResources: s.nResources, iter: s.iter });
      }

      this.nCandidates_.push(candidates.length);
      this.nResources_.push(resources);

      const nKeep = Math.max(1, Math.floor(candidates.length / this.factor));
      candidates = scores.slice(0, nKeep).map((s) => s.params);
      resources = Math.min(Math.floor(resources * this.factor), maxRes);
      iter++;
    }

    if (candidates.length > 0) {
      this.bestParams_ = candidates[0] as Record<string, unknown>;
      this.bestEstimator_ = this.estimator.setParams(this.bestParams_);
      this.bestEstimator_.fit(X, y);
      this.bestScore_ = this.bestEstimator_.score(X, y);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    return (this.bestEstimator_ as unknown as { predict: (X: Float64Array[]) => Float64Array | Int32Array }).predict(X);
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    return this.bestEstimator_.score(X, y);
  }
}

export class StratifiedGroupKFold {
  nSplits: number;

  constructor(nSplits = 5) {
    this.nSplits = nSplits;
  }

  split(X: Float64Array[], y: Int32Array, groups: Int32Array): Array<{ trainIdx: Int32Array; testIdx: Int32Array }> {
    const n = X.length;
    const uniqueGroups = [...new Set(Array.from(groups))];
    const nGroups = uniqueGroups.length;
    const foldSize = Math.max(1, Math.floor(nGroups / this.nSplits));
    const splits: Array<{ trainIdx: Int32Array; testIdx: Int32Array }> = [];

    for (let fold = 0; fold < this.nSplits; fold++) {
      const testGroupStart = fold * foldSize;
      const testGroupEnd = fold === this.nSplits - 1 ? nGroups : testGroupStart + foldSize;
      const testGroups = new Set(uniqueGroups.slice(testGroupStart, testGroupEnd));
      const testIdx = Int32Array.from(Array.from({ length: n }, (_, i) => i).filter((i) => testGroups.has(groups[i] ?? 0)));
      const trainIdx = Int32Array.from(Array.from({ length: n }, (_, i) => i).filter((i) => !testGroups.has(groups[i] ?? 0)));
      splits.push({ trainIdx, testIdx });
    }
    return splits;
    void y;
  }
}

export function crossValPredict(
  estimator: CVEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  cv = 5,
): Float64Array | Int32Array {
  const n = X.length;
  const foldSize = Math.max(1, Math.floor(n / cv));
  const isRegression = y instanceof Float64Array;
  const predictions = isRegression ? new Float64Array(n).fill(NaN) : new Int32Array(n).fill(-1);

  for (let fold = 0; fold < cv; fold++) {
    const valStart = fold * foldSize;
    const valEnd = fold === cv - 1 ? n : valStart + foldSize;
    const trainIdx = [...Array.from({ length: valStart }, (_, i) => i), ...Array.from({ length: n - valEnd }, (_, i) => i + valEnd)];
    const XTrain = trainIdx.map((i) => X[i] as Float64Array);
    const yTrain = isRegression
      ? Float64Array.from(trainIdx, (i) => (y as Float64Array)[i] ?? 0)
      : Int32Array.from(trainIdx, (i) => (y as Int32Array)[i] ?? 0);
    const XVal = X.slice(valStart, valEnd);

    estimator.fit(XTrain, yTrain);
    const preds = (estimator as unknown as { predict: (X: Float64Array[]) => Float64Array | Int32Array }).predict(XVal);
    for (let j = 0; j < preds.length; j++) {
      if (isRegression) (predictions as Float64Array)[valStart + j] = (preds as Float64Array)[j] ?? 0;
      else (predictions as Int32Array)[valStart + j] = (preds as Int32Array)[j] ?? 0;
    }
  }
  return predictions;
}
