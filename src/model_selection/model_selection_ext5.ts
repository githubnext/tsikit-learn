/**
 * Model selection extensions: HalvingGridSearchCV, HalvingRandomSearchCV.
 * Mirrors sklearn.model_selection successive halving.
 */

import { BaseEstimator } from "../base.js";

type Estimator = {
  fit(X: Float64Array[], y: Float64Array | Int32Array): unknown;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
};

export interface HalvingGridSearchCVParams {
  factor?: number;
  min_resources?: number;
  max_resources?: number | "auto";
  cv?: number;
  scoring?: string | null;
  refit?: boolean;
  random_state?: number | null;
}

/** Successive halving for grid search. */
export class HalvingGridSearchCV extends BaseEstimator {
  estimator: Estimator;
  param_grid: Record<string, unknown[]>;
  factor: number;
  min_resources: number;
  max_resources: number | "auto";
  cv: number;
  refit: boolean;
  random_state: number | null;
  best_params_: Record<string, unknown> = {};
  best_score_ = -Number.POSITIVE_INFINITY;
  best_estimator_: Estimator | null = null;
  cv_results_: {
    params: Record<string, unknown>;
    mean_test_score: number;
    n_resources: number;
  }[] = [];
  n_iterations_ = 0;

  constructor(
    estimator: Estimator,
    paramGrid: Record<string, unknown[]>,
    params: HalvingGridSearchCVParams = {},
  ) {
    super();
    this.estimator = estimator;
    this.param_grid = paramGrid;
    this.factor = params.factor ?? 3;
    this.min_resources = params.min_resources ?? 10;
    this.max_resources = params.max_resources ?? "auto";
    this.cv = params.cv ?? 5;
    this.refit = params.refit ?? true;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    const maxRes = this.max_resources === "auto" ? n : this.max_resources;
    // Generate all parameter combinations
    let candidates = this._gridCombinations(this.param_grid);
    let resources = this.min_resources;
    const results: typeof this.cv_results_ = [];

    while (candidates.length > 0) {
      this.n_iterations_++;
      const nSamples = Math.min(resources, n);
      const scores: number[] = [];
      for (const params of candidates) {
        const score = this._crossValScore(X.slice(0, nSamples), (y as Float64Array).slice(0, nSamples), params);
        scores.push(score);
        results.push({ params, mean_test_score: score, n_resources: nSamples });
      }
      const topN = Math.max(1, Math.floor(candidates.length / this.factor));
      const sorted = candidates.map((p, i) => ({ p, s: scores[i] ?? 0 })).sort((a, b) => b.s - a.s);
      candidates = sorted.slice(0, topN).map((x) => x.p);
      resources = Math.min(resources * this.factor, maxRes);
      if (resources >= maxRes && candidates.length <= 1) break;
    }

    this.cv_results_ = results;
    if (results.length > 0) {
      const best = results.reduce((a, b) => a.mean_test_score > b.mean_test_score ? a : b);
      this.best_params_ = best.params;
      this.best_score_ = best.mean_test_score;
    }
    if (this.refit && this.best_params_) {
      const est = Object.create(this.estimator) as Estimator;
      Object.assign(est, this.best_params_);
      est.fit(X, y);
      this.best_estimator_ = est;
    }
    return this;
  }

  private _gridCombinations(grid: Record<string, unknown[]>): Record<string, unknown>[] {
    const keys = Object.keys(grid);
    if (keys.length === 0) return [{}];
    let result: Record<string, unknown>[] = [{}];
    for (const key of keys) {
      const vals = grid[key] ?? [];
      const newResult: Record<string, unknown>[] = [];
      for (const existing of result) for (const v of vals) newResult.push({ ...existing, [key]: v });
      result = newResult;
    }
    return result;
  }

  private _crossValScore(
    X: Float64Array[],
    y: Float64Array | Int32Array,
    params: Record<string, unknown>,
  ): number {
    const n = X.length;
    const foldSize = Math.max(1, Math.floor(n / this.cv));
    let totalScore = 0;
    for (let fold = 0; fold < this.cv; fold++) {
      const start = fold * foldSize;
      const end = Math.min(start + foldSize, n);
      if (end <= start) continue;
      const trainX = [...X.slice(0, start), ...X.slice(end)];
      const testX = X.slice(start, end);
      const trainY = this._sliceArray(y, 0, start, end, n);
      const testY = y instanceof Int32Array ? new Int32Array(y.buffer, y.byteOffset + start * 4, end - start) : new Float64Array(y.buffer, y.byteOffset + start * 8, end - start);
      const est = Object.create(this.estimator) as Estimator;
      Object.assign(est, params);
      try { est.fit(trainX, trainY); totalScore += est.score(testX, testY); } catch { /* skip */ }
    }
    return totalScore / this.cv;
  }

  private _sliceArray(
    arr: Float64Array | Int32Array,
    _start: number,
    skip_start: number,
    skip_end: number,
    n: number,
  ): Float64Array | Int32Array {
    const indices = [...Array.from({ length: skip_start }, (_, i) => i), ...Array.from({ length: n - skip_end }, (_, i) => skip_end + i)];
    if (arr instanceof Int32Array) {
      const out = new Int32Array(indices.length);
      for (let i = 0; i < indices.length; i++) out[i] = arr[indices[i]!] ?? 0;
      return out;
    }
    const out = new Float64Array(indices.length);
    for (let i = 0; i < indices.length; i++) out[i] = (arr as Float64Array)[indices[i]!] ?? 0;
    return out;
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    if (!this.best_estimator_) throw new Error("Not fitted");
    return this.best_estimator_.score(X, y);
  }
}

export interface PermutationImportanceParams {
  n_repeats?: number;
  random_state?: number | null;
  scoring?: string | null;
}

/** Permutation feature importance. */
export class PermutationImportance extends BaseEstimator {
  estimator: Estimator;
  n_repeats: number;
  random_state: number | null;
  importances_mean_: Float64Array = new Float64Array(0);
  importances_std_: Float64Array = new Float64Array(0);
  importances_: Float64Array[] = [];

  constructor(estimator: Estimator, params: PermutationImportanceParams = {}) {
    super();
    this.estimator = estimator;
    this.n_repeats = params.n_repeats ?? 5;
    this.random_state = params.random_state ?? null;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    this.estimator.fit(X, y);
    const baseScore = this.estimator.score(X, y);
    const nf = X[0]?.length ?? 0;
    this.importances_mean_ = new Float64Array(nf);
    this.importances_std_ = new Float64Array(nf);
    this.importances_ = Array.from({ length: nf }, () => new Float64Array(this.n_repeats));
    for (let k = 0; k < nf; k++) {
      for (let r = 0; r < this.n_repeats; r++) {
        const Xp = X.map((xi) => new Float64Array(xi));
        // Permute feature k
        const colVals = Xp.map((xi) => xi[k] ?? 0);
        for (let i = colVals.length - 1; i > 0; i--) {
          const j = ((this.random_state ?? 42) * 1664525 + i * 1013904223) % (i + 1);
          const tmp = colVals[i]!; colVals[i] = colVals[j]!; colVals[j] = tmp;
        }
        for (let i = 0; i < Xp.length; i++) Xp[i]![k] = colVals[i] ?? 0;
        const permScore = this.estimator.score(Xp, y);
        this.importances_[k]![r] = baseScore - permScore;
      }
      let mean = 0;
      for (let r = 0; r < this.n_repeats; r++) mean += this.importances_[k]![r] ?? 0;
      mean /= this.n_repeats;
      this.importances_mean_[k] = mean;
      let variance = 0;
      for (let r = 0; r < this.n_repeats; r++) variance += ((this.importances_[k]![r] ?? 0) - mean) ** 2;
      this.importances_std_[k] = Math.sqrt(variance / this.n_repeats);
    }
    return this;
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    return this.estimator.score(X, y);
  }
}
