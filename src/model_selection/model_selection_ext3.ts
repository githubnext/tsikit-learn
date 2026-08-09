/**
 * Model selection extensions: BayesianOptimization, NelderMeadSearch, SuccessiveHalvingExt
 * Port of sklearn.model_selection extensions
 */

export interface ParamSpace {
  [key: string]: { low: number; high: number; log?: boolean };
}

export interface SearchResult {
  params: Record<string, number>;
  score: number;
}

export class BayesianOptimizationCV {
  paramSpace: ParamSpace;
  nIter: number;
  cv: number;
  randomState: number;
  explorationFactor: number;

  bestParams_: Record<string, number> | null = null;
  bestScore_ = -Number.POSITIVE_INFINITY;
  results_: SearchResult[] = [];

  constructor(opts: {
    paramSpace?: ParamSpace;
    nIter?: number;
    cv?: number;
    randomState?: number;
    explorationFactor?: number;
  } = {}) {
    this.paramSpace = opts.paramSpace ?? {};
    this.nIter = opts.nIter ?? 20;
    this.cv = opts.cv ?? 3;
    this.randomState = opts.randomState ?? 42;
    this.explorationFactor = opts.explorationFactor ?? 0.1;
  }

  private sample(rng: () => number): Record<string, number> {
    const params: Record<string, number> = {};
    for (const [key, space] of Object.entries(this.paramSpace)) {
      const u = rng();
      if (space.log) {
        params[key] = Math.exp(Math.log(space.low) + u * (Math.log(space.high) - Math.log(space.low)));
      } else {
        params[key] = space.low + u * (space.high - space.low);
      }
    }
    return params;
  }

  private gaussianKernel(x1: Record<string, number>, x2: Record<string, number>): number {
    let dist = 0;
    for (const [key, space] of Object.entries(this.paramSpace)) {
      const range = space.high - space.low + 1e-15;
      dist += ((x1[key] ?? 0) - (x2[key] ?? 0)) ** 2 / (range ** 2);
    }
    return Math.exp(-0.5 * dist);
  }

  private acquisitionUCB(params: Record<string, number>): number {
    if (this.results_.length === 0) return 0;
    let mu = 0;
    let k = 0;
    for (const r of this.results_) {
      const w = this.gaussianKernel(params, r.params);
      mu += w * r.score;
      k += w;
    }
    mu /= k + 1e-15;
    let variance = 0;
    for (const r of this.results_) {
      const w = this.gaussianKernel(params, r.params);
      variance += w * (r.score - mu) ** 2;
    }
    variance /= k + 1e-15;
    return mu + this.explorationFactor * Math.sqrt(variance + 1e-15);
  }

  optimize(scoreFn: (params: Record<string, number>) => number): this {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    for (let i = 0; i < this.nIter; i++) {
      let bestAcq = -Number.POSITIVE_INFINITY;
      let candidateParams = this.sample(rng);
      if (this.results_.length >= 2) {
        for (let t = 0; t < 10; t++) {
          const p = this.sample(rng);
          const acq = this.acquisitionUCB(p);
          if (acq > bestAcq) { bestAcq = acq; candidateParams = p; }
        }
      }
      const score = scoreFn(candidateParams);
      this.results_.push({ params: candidateParams, score });
      if (score > this.bestScore_) {
        this.bestScore_ = score;
        this.bestParams_ = { ...candidateParams };
      }
    }
    return this;
  }
}

export class NelderMeadOptimizer {
  maxIter: number;
  tol: number;
  alpha: number;
  gamma: number;
  rho: number;
  sigma: number;

  result_: { x: Float64Array; fun: number } | null = null;

  constructor(opts: {
    maxIter?: number;
    tol?: number;
    alpha?: number;
    gamma?: number;
    rho?: number;
    sigma?: number;
  } = {}) {
    this.maxIter = opts.maxIter ?? 500;
    this.tol = opts.tol ?? 1e-6;
    this.alpha = opts.alpha ?? 1.0;
    this.gamma = opts.gamma ?? 2.0;
    this.rho = opts.rho ?? 0.5;
    this.sigma = opts.sigma ?? 0.5;
  }

  minimize(fn: (x: Float64Array) => number, x0: Float64Array): this {
    const n = x0.length;
    let simplex: Float64Array[] = [x0.slice()];
    for (let i = 0; i < n; i++) {
      const xi = x0.slice();
      xi[i] = (xi[i] ?? 0) + (Math.abs(xi[i] ?? 0) > 1e-10 ? 0.05 * (xi[i] ?? 0) : 0.00025);
      simplex.push(xi);
    }
    let fvals = simplex.map(fn);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const order = Array.from({ length: n + 1 }, (_, i) => i).sort((a, b) => (fvals[a] ?? 0) - (fvals[b] ?? 0));
      simplex = order.map(i => simplex[i]!);
      fvals = order.map(i => fvals[i] ?? 0);
      const spread = Math.abs((fvals[n] ?? 0) - (fvals[0] ?? 0));
      if (spread < this.tol) break;
      const centroid = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] = (centroid[j] ?? 0) + (simplex[i]![j] ?? 0) / n;
      const xr = new Float64Array(n);
      for (let j = 0; j < n; j++) xr[j] = (1 + this.alpha) * (centroid[j] ?? 0) - this.alpha * (simplex[n]![j] ?? 0);
      const fr = fn(xr);
      if (fr < (fvals[0] ?? 0)) {
        const xe = new Float64Array(n);
        for (let j = 0; j < n; j++) xe[j] = (1 + this.gamma) * (centroid[j] ?? 0) - this.gamma * (simplex[n]![j] ?? 0);
        const fe = fn(xe);
        if (fe < fr) { simplex[n] = xe; fvals[n] = fe; }
        else { simplex[n] = xr; fvals[n] = fr; }
      } else if (fr < (fvals[n - 1] ?? 0)) {
        simplex[n] = xr; fvals[n] = fr;
      } else {
        const xc = new Float64Array(n);
        for (let j = 0; j < n; j++) xc[j] = this.rho * (simplex[n]![j] ?? 0) + (1 - this.rho) * (centroid[j] ?? 0);
        const fc = fn(xc);
        if (fc < (fvals[n] ?? 0)) { simplex[n] = xc; fvals[n] = fc; }
        else {
          for (let i = 1; i <= n; i++) {
            const xs = new Float64Array(n);
            for (let j = 0; j < n; j++) xs[j] = (simplex[0]![j] ?? 0) + this.sigma * ((simplex[i]![j] ?? 0) - (simplex[0]![j] ?? 0));
            simplex[i] = xs;
            fvals[i] = fn(xs);
          }
        }
      }
      void iter;
    }
    this.result_ = { x: simplex[0]!, fun: fvals[0] ?? 0 };
    return this;
  }
}

export class SuccessiveHalvingExt {
  paramDistributions: ParamSpace;
  nCandidates: number;
  factor: number;
  cv: number;
  randomState: number;

  bestParams_: Record<string, number> | null = null;
  bestScore_ = -Number.POSITIVE_INFINITY;

  constructor(opts: {
    paramDistributions?: ParamSpace;
    nCandidates?: number;
    factor?: number;
    cv?: number;
    randomState?: number;
  } = {}) {
    this.paramDistributions = opts.paramDistributions ?? {};
    this.nCandidates = opts.nCandidates ?? 20;
    this.factor = opts.factor ?? 3;
    this.cv = opts.cv ?? 3;
    this.randomState = opts.randomState ?? 0;
  }

  fit(scoreFn: (params: Record<string, number>, budget: number) => number): this {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    let candidates: Record<string, number>[] = Array.from({ length: this.nCandidates }, () => {
      const p: Record<string, number> = {};
      for (const [key, space] of Object.entries(this.paramDistributions)) {
        const u = rng();
        p[key] = space.log ? Math.exp(Math.log(space.low) + u * (Math.log(space.high) - Math.log(space.low))) : space.low + u * (space.high - space.low);
      }
      return p;
    });
    let budget = 1;
    while (candidates.length > 1) {
      const scored = candidates.map(p => ({ params: p, score: scoreFn(p, budget) }));
      scored.sort((a, b) => b.score - a.score);
      candidates = scored.slice(0, Math.max(1, Math.floor(scored.length / this.factor))).map(s => s.params);
      if (scored[0] && scored[0].score > this.bestScore_) {
        this.bestScore_ = scored[0].score;
        this.bestParams_ = { ...scored[0].params };
      }
      budget = Math.floor(budget * this.factor);
    }
    return this;
  }
}
