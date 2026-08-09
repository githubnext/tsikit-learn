/**
 * Successive Halving search strategies.
 * Mirrors sklearn.model_selection.HalvingGridSearchCV and HalvingRandomSearchCV.
 */

import { NotFittedError } from "../exceptions.js";

export interface SHEstimator {
  fit(
    X: Float64Array[],
    y: Float64Array | Int32Array,
    sampleWeight?: Float64Array,
  ): this;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
  getParams(): Record<string, unknown>;
  setParams(params: Record<string, unknown>): this;
  clone(): this;
}

export interface HalvingGridSearchCVOptions {
  estimator: SHEstimator;
  paramGrid: Record<string, unknown[]>;
  factor?: number;
  resource?: string;
  maxResources?: number | "auto";
  minResources?: number | "exhaust";
  aggressiveElimination?: boolean;
  cv?: number;
  scoring?: string;
  refit?: boolean;
  nJobs?: number;
  verbose?: number;
  randomState?: number;
}

export interface HalvingRandomSearchCVOptions {
  estimator: SHEstimator;
  paramDistributions: Record<string, unknown[] | (() => unknown)>;
  nCandidates?: number | "exhaust";
  factor?: number;
  resource?: string;
  maxResources?: number | "auto";
  minResources?: number | "exhaust";
  aggressiveElimination?: boolean;
  cv?: number;
  scoring?: string;
  refit?: boolean;
  nJobs?: number;
  verbose?: number;
  randomState?: number;
}

interface CVResult {
  params: Record<string, unknown>;
  meanTestScore: number;
  stdTestScore: number;
  rank: number;
}

function crossValScore(
  estimator: SHEstimator,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  cv: number,
  nSamples: number,
): number {
  const foldSize = Math.floor(nSamples / cv);
  const scores: number[] = [];
  for (let fold = 0; fold < cv; fold++) {
    const start = fold * foldSize;
    const end = fold === cv - 1 ? nSamples : (fold + 1) * foldSize;
    const valIdx: number[] = [];
    const trainIdx: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if (i >= start && i < end) valIdx.push(i);
      else trainIdx.push(i);
    }
    const xTrain = trainIdx.map((i) => X[i]!);
    const xVal = valIdx.map((i) => X[i]!);
    let yTrain: Float64Array | Int32Array;
    let yVal: Float64Array | Int32Array;
    if (y instanceof Int32Array) {
      yTrain = new Int32Array(trainIdx.map((i) => y[i] ?? 0));
      yVal = new Int32Array(valIdx.map((i) => y[i] ?? 0));
    } else {
      yTrain = new Float64Array(trainIdx.map((i) => y[i] ?? 0));
      yVal = new Float64Array(valIdx.map((i) => y[i] ?? 0));
    }
    try {
      estimator.fit(xTrain, yTrain);
      scores.push(estimator.score(xVal, yVal));
    } catch {
      scores.push(Number.NaN);
    }
  }
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

function expandParamGrid(
  paramGrid: Record<string, unknown[]>,
): Record<string, unknown>[] {
  const keys = Object.keys(paramGrid);
  if (keys.length === 0) return [{}];
  const result: Record<string, unknown>[] = [];
  function expand(idx: number, current: Record<string, unknown>): void {
    if (idx === keys.length) {
      result.push({ ...current });
      return;
    }
    const key = keys[idx]!;
    for (const val of paramGrid[key]!) {
      current[key] = val;
      expand(idx + 1, current);
    }
  }
  expand(0, {});
  return result;
}

export class HalvingGridSearchCV {
  private estimator: SHEstimator;
  private paramGrid: Record<string, unknown[]>;
  private factor: number;
  private maxResources: number | "auto";
  private minResources: number | "exhaust";
  private cv: number;
  private refit: boolean;
  private randomState: number;

  bestParams_?: Record<string, unknown>;
  bestScore_?: number;
  bestEstimator_?: SHEstimator;
  cvResults_?: CVResult[];
  nIterations_?: number;
  nRequiredIterations_?: number;
  nPossibleIterations_?: number;

  constructor(options: HalvingGridSearchCVOptions) {
    this.estimator = options.estimator;
    this.paramGrid = options.paramGrid;
    this.factor = options.factor ?? 3;
    this.maxResources = options.maxResources ?? "auto";
    this.minResources = options.minResources ?? "exhaust";
    this.cv = options.cv ?? 5;
    this.refit = options.refit ?? true;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const nSamples = X.length;
    const maxRes = this.maxResources === "auto" ? nSamples : this.maxResources;
    const allCandidates = expandParamGrid(this.paramGrid);
    const nCandidates = allCandidates.length;

    // Compute min resources (floor to make halving divide evenly)
    const nIterations = Math.ceil(
      Math.log(nCandidates) / Math.log(this.factor),
    );
    const minRes =
      this.minResources === "exhaust"
        ? Math.max(1, Math.floor(maxRes / this.factor ** nIterations))
        : this.minResources;

    this.nIterations_ = nIterations;
    this.nRequiredIterations_ = nIterations;
    this.nPossibleIterations_ = nIterations;

    // Shuffle candidates
    const rng = this.randomState;
    const shuffled = [...allCandidates];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = ((rng * (i + 1)) ^ 0xdeadbeef) % (i + 1);
      const tmp = shuffled[i]!;
      shuffled[i] = shuffled[j]!;
      shuffled[j] = tmp;
    }

    let currentCandidates = shuffled.slice(0, nCandidates);
    let currentRes = minRes;
    const allResults: CVResult[] = [];

    for (let iter = 0; iter < nIterations; iter++) {
      const scores: { idx: number; score: number }[] = [];
      for (let ci = 0; ci < currentCandidates.length; ci++) {
        const params = currentCandidates[ci]!;
        const cloned = this.estimator.clone();
        cloned.setParams(params);
        const useN = Math.min(currentRes, nSamples);
        const xSub = X.slice(0, useN);
        const ySub =
          y instanceof Int32Array
            ? new Int32Array(y.buffer, 0, useN)
            : new Float64Array(y.buffer, 0, useN);
        const score = crossValScore(
          cloned,
          xSub,
          ySub,
          Math.min(this.cv, useN),
          useN,
        );
        scores.push({ idx: ci, score });
        allResults.push({
          params,
          meanTestScore: score,
          stdTestScore: 0,
          rank: 0,
        });
      }

      scores.sort((a, b) => b.score - a.score);
      const nKeep = Math.max(
        1,
        Math.floor(currentCandidates.length / this.factor),
      );
      currentCandidates = scores
        .slice(0, nKeep)
        .map((s) => currentCandidates[s.idx]!);
      currentRes = Math.min(currentRes * this.factor, maxRes);
    }

    // Rank results
    const sorted = [...allResults].sort(
      (a, b) => b.meanTestScore - a.meanTestScore,
    );
    for (let i = 0; i < sorted.length; i++) sorted[i]!.rank = i + 1;
    this.cvResults_ = sorted;

    this.bestParams_ = sorted[0]?.params ?? {};
    this.bestScore_ = sorted[0]?.meanTestScore ?? 0;

    if (this.refit) {
      const best = this.estimator.clone();
      best.setParams(this.bestParams_);
      best.fit(X, y);
      this.bestEstimator_ = best;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    if (!this.bestEstimator_)
      throw new NotFittedError("HalvingGridSearchCV is not fitted");
    return (
      this.bestEstimator_ as unknown as {
        predict(X: Float64Array[]): Float64Array | Int32Array;
      }
    ).predict(X);
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    if (!this.bestEstimator_)
      throw new NotFittedError("HalvingGridSearchCV is not fitted");
    return this.bestEstimator_.score(X, y);
  }
}

export class HalvingRandomSearchCV {
  private estimator: SHEstimator;
  private paramDistributions: Record<string, unknown[] | (() => unknown)>;
  private nCandidates: number | "exhaust";
  private factor: number;
  private maxResources: number | "auto";
  private cv: number;
  private refit: boolean;
  private randomState: number;

  bestParams_?: Record<string, unknown>;
  bestScore_?: number;
  bestEstimator_?: SHEstimator;
  cvResults_?: CVResult[];

  constructor(options: HalvingRandomSearchCVOptions) {
    this.estimator = options.estimator;
    this.paramDistributions = options.paramDistributions;
    this.nCandidates = options.nCandidates ?? 10;
    this.factor = options.factor ?? 3;
    this.maxResources = options.maxResources ?? "auto";
    this.cv = options.cv ?? 5;
    this.refit = options.refit ?? true;
    this.randomState = options.randomState ?? 0;
  }

  private sampleParams(seed: number): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    let s = seed;
    for (const [key, dist] of Object.entries(this.paramDistributions)) {
      if (typeof dist === "function") {
        params[key] = dist();
      } else {
        s = (s * 1664525 + 1013904223) & 0xffffffff;
        params[key] = dist[Math.abs(s) % dist.length];
      }
    }
    return params;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const nSamples = X.length;
    const maxRes = this.maxResources === "auto" ? nSamples : this.maxResources;
    const nCands = this.nCandidates === "exhaust" ? 20 : this.nCandidates;

    const candidates: Record<string, unknown>[] = [];
    for (let i = 0; i < nCands; i++) {
      candidates.push(this.sampleParams(this.randomState + i));
    }

    const nIterations = Math.ceil(Math.log(nCands) / Math.log(this.factor));
    const minRes = Math.max(1, Math.floor(maxRes / this.factor ** nIterations));

    let currentCandidates = [...candidates];
    let currentRes = minRes;
    const allResults: CVResult[] = [];

    for (let iter = 0; iter < nIterations; iter++) {
      const scores: { idx: number; score: number }[] = [];
      for (let ci = 0; ci < currentCandidates.length; ci++) {
        const params = currentCandidates[ci]!;
        const cloned = this.estimator.clone();
        cloned.setParams(params);
        const useN = Math.min(currentRes, nSamples);
        const xSub = X.slice(0, useN);
        const ySub =
          y instanceof Int32Array
            ? new Int32Array(y.buffer, 0, useN)
            : new Float64Array(y.buffer, 0, useN);
        const score = crossValScore(
          cloned,
          xSub,
          ySub,
          Math.min(this.cv, useN),
          useN,
        );
        scores.push({ idx: ci, score });
        allResults.push({
          params,
          meanTestScore: score,
          stdTestScore: 0,
          rank: 0,
        });
      }

      scores.sort((a, b) => b.score - a.score);
      const nKeep = Math.max(
        1,
        Math.floor(currentCandidates.length / this.factor),
      );
      currentCandidates = scores
        .slice(0, nKeep)
        .map((s) => currentCandidates[s.idx]!);
      currentRes = Math.min(currentRes * this.factor, maxRes);
    }

    const sorted = [...allResults].sort(
      (a, b) => b.meanTestScore - a.meanTestScore,
    );
    for (let i = 0; i < sorted.length; i++) sorted[i]!.rank = i + 1;
    this.cvResults_ = sorted;
    this.bestParams_ = sorted[0]?.params ?? {};
    this.bestScore_ = sorted[0]?.meanTestScore ?? 0;

    if (this.refit) {
      const best = this.estimator.clone();
      best.setParams(this.bestParams_);
      best.fit(X, y);
      this.bestEstimator_ = best;
    }

    return this;
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    if (!this.bestEstimator_)
      throw new NotFittedError("HalvingRandomSearchCV is not fitted");
    return this.bestEstimator_.score(X, y);
  }
}
