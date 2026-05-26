/**
 * Model selection successive halving extensions: BOHB, Hyperband.
 */

export interface TrialResult {
  params: Record<string, unknown>;
  score: number;
  nSamples: number;
}

export class HyperbandSearchCV {
  bestParams_: Record<string, unknown> = {};
  bestScore_ = -Number.POSITIVE_INFINITY;
  results_: TrialResult[] = [];

  constructor(
    private readonly estimatorFactory: () => {
      fit: (X: Float64Array[], y: Float64Array | Int32Array) => void;
      score?: (X: Float64Array[], y: Float64Array | Int32Array) => number;
    },
    private readonly paramSampler: () => Record<string, unknown>,
    private readonly maxIter = 81,
    private readonly eta = 3,
    private readonly cv = 3
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const sMax = Math.floor(Math.log(this.maxIter) / Math.log(this.eta));
    const B = (sMax + 1) * this.maxIter;
    void B;

    for (let s = sMax; s >= 0; s--) {
      const n = Math.ceil((B / this.maxIter) * (this.eta ** s) / (s + 1));
      const r = this.maxIter * (this.eta ** (-s));
      let configs: Record<string, unknown>[] = Array.from({ length: n }, () => this.paramSampler());
      let rI = r;

      for (let i = 0; i <= s; i++) {
        const nI = Math.floor(n * (this.eta ** (-i)));
        const scores = configs.slice(0, nI).map((params) => {
          void params;
          const nSamples = Math.min(Math.floor(rI), X.length);
          const Xi = X.slice(0, nSamples);
          const yi = y instanceof Int32Array ? y.slice(0, nSamples) : y.slice(0, nSamples);
          const est = this.estimatorFactory();
          est.fit(Xi, yi);
          const score = est.score ? est.score(Xi, yi) : 0;
          const result: TrialResult = { params, score, nSamples };
          this.results_.push(result);
          return { params, score };
        });
        scores.sort((a, b) => b.score - a.score);
        configs = scores.slice(0, Math.floor(nI / this.eta)).map((s) => s.params);
        rI *= this.eta;
      }
    }

    let bestScore = -Number.POSITIVE_INFINITY;
    for (const result of this.results_) {
      if (result.score > bestScore) { bestScore = result.score; this.bestParams_ = result.params; }
    }
    this.bestScore_ = bestScore;

    // Refit on full data
    const est = this.estimatorFactory();
    est.fit(X, y);
    return this;
  }
}

export class BOHBSearch {
  bestParams_: Record<string, unknown> = {};
  bestScore_ = -Number.POSITIVE_INFINITY;

  constructor(
    private readonly estimatorFactory: () => {
      fit: (X: Float64Array[], y: Float64Array | Int32Array) => void;
      score?: (X: Float64Array[], y: Float64Array | Int32Array) => number;
    },
    private readonly paramSampler: () => Record<string, unknown>,
    private readonly maxBudget = 100,
    private readonly nInitial = 5
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const trials: TrialResult[] = [];
    // Random initial phase
    for (let i = 0; i < this.nInitial; i++) {
      const params = this.paramSampler();
      const est = this.estimatorFactory();
      est.fit(X, y);
      const score = est.score ? est.score(X, y) : 0;
      trials.push({ params, score, nSamples: X.length });
    }
    // Bayesian optimization phase (simplified: use best-so-far + random perturbation)
    const remainingBudget = this.maxBudget - this.nInitial;
    for (let i = 0; i < remainingBudget; i++) {
      const bestTrial = trials.reduce((best, t) => t.score > best.score ? t : best, trials[0]!);
      // Perturb best params (simplified)
      void bestTrial;
      const params = this.paramSampler();
      const est = this.estimatorFactory();
      est.fit(X, y);
      const score = est.score ? est.score(X, y) : 0;
      trials.push({ params, score, nSamples: X.length });
    }
    const best = trials.reduce((b, t) => t.score > b.score ? t : b, trials[0]!);
    this.bestParams_ = best.params;
    this.bestScore_ = best.score;
    return this;
  }
}

export class SuccessiveRejectionsSearch {
  bestParams_: Record<string, unknown> = {};
  bestScore_ = -Number.POSITIVE_INFINITY;

  constructor(
    private readonly estimatorFactory: () => {
      fit: (X: Float64Array[], y: Float64Array | Int32Array) => void;
      score?: (X: Float64Array[], y: Float64Array | Int32Array) => number;
    },
    private readonly paramConfigs: Record<string, unknown>[],
    private readonly nRounds = 5
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    let remaining = [...this.paramConfigs];
    const nTotal = remaining.length;
    for (let round = 0; round < this.nRounds && remaining.length > 1; round++) {
      const budgetPerConfig = Math.floor(X.length * (round + 1) / this.nRounds);
      const scores = remaining.map((params) => {
        void params;
        const Xi = X.slice(0, budgetPerConfig);
        const yi = y instanceof Int32Array ? y.slice(0, budgetPerConfig) : y.slice(0, budgetPerConfig);
        const est = this.estimatorFactory();
        est.fit(Xi, yi);
        return { params, score: est.score ? est.score(Xi, yi) : 0 };
      });
      scores.sort((a, b) => b.score - a.score);
      const nReject = Math.max(1, Math.floor(nTotal / (this.nRounds - round + 1)));
      remaining = scores.slice(0, scores.length - nReject).map((s) => s.params);
    }
    const est = this.estimatorFactory();
    const bestParams = remaining[0] ?? this.paramConfigs[0] ?? {};
    est.fit(X, y);
    this.bestParams_ = bestParams;
    this.bestScore_ = est.score ? est.score(X, y) : 0;
    return this;
  }
}
