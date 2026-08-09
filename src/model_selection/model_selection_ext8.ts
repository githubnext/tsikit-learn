/**
 * Model selection extensions: Bayesian optimization, cross-validation extensions
 */

export interface BayesianOptResult {
  bestParams: Record<string, number>;
  bestScore: number;
  scores: Float64Array;
  params: Record<string, number>[];
}

export class BayesianOptimizationExt {
  private observations_: Array<{ params: Record<string, number>; score: number }> = [];
  private fitted_ = false;

  constructor(
    private paramBounds: Record<string, [number, number]>,
    private nInitialPoints: number = 5,
    private acquisitionFn: 'ei' | 'ucb' | 'pi' = 'ei',
    private randomState: number = 42
  ) {}

  optimize(
    objective: (params: Record<string, number>) => number,
    nCalls: number = 20
  ): BayesianOptResult {
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
    const paramNames = Object.keys(this.paramBounds);
    const scores: number[] = [];
    const params: Record<string, number>[] = [];

    // Random initial exploration
    for (let i = 0; i < Math.min(this.nInitialPoints, nCalls); i++) {
      const p = Object.fromEntries(paramNames.map(name => {
        const [lo, hi] = this.paramBounds[name]!;
        return [name, lo + rand() * (hi - lo)];
      }));
      const score = objective(p);
      this.observations_.push({ params: p, score });
      scores.push(score); params.push(p);
    }

    // Bayesian optimization iterations
    for (let i = this.nInitialPoints; i < nCalls; i++) {
      const nextParams = this._suggestNext(rand, paramNames);
      const score = objective(nextParams);
      this.observations_.push({ params: nextParams, score });
      scores.push(score); params.push(nextParams);
    }

    const bestIdx = scores.reduce((best, s, i) => s > (scores[best] ?? 0) ? i : best, 0);
    this.fitted_ = true;
    return {
      bestParams: params[bestIdx] ?? {},
      bestScore: scores[bestIdx] ?? 0,
      scores: new Float64Array(scores),
      params
    };
  }

  private _suggestNext(rand: () => number, paramNames: string[]): Record<string, number> {
    if (this.observations_.length < 2) {
      return Object.fromEntries(paramNames.map(name => {
        const [lo, hi] = this.paramBounds[name]!;
        return [name, lo + rand() * (hi - lo)];
      }));
    }

    // Generate candidates and score by acquisition
    const nCandidates = 100;
    let bestCandidate: Record<string, number> = {};
    let bestAcq = Number.NEGATIVE_INFINITY;

    for (let c = 0; c < nCandidates; c++) {
      const candidate = Object.fromEntries(paramNames.map(name => {
        const [lo, hi] = this.paramBounds[name]!;
        return [name, lo + rand() * (hi - lo)];
      }));
      const acq = this._acquisition(candidate, paramNames);
      if (acq > bestAcq) { bestAcq = acq; bestCandidate = candidate; }
    }
    return bestCandidate;
  }

  private _acquisition(candidate: Record<string, number>, paramNames: string[]): number {
    const { mean, std } = this._surrogate(candidate, paramNames);
    const bestSoFar = Math.max(...this.observations_.map(o => o.score));

    if (this.acquisitionFn === 'ucb') return mean + 1.96 * std;
    if (this.acquisitionFn === 'pi') {
      const z = (mean - bestSoFar) / (std + 1e-10);
      return normCDF(z);
    }
    // EI
    const z = (mean - bestSoFar) / (std + 1e-10);
    return (mean - bestSoFar) * normCDF(z) + std * normPDF(z);
  }

  private _surrogate(candidate: Record<string, number>, paramNames: string[]): { mean: number; std: number } {
    // Gaussian Process surrogate (simple RBF kernel)
    const n = this.observations_.length;
    const xTrain = this.observations_.map(o => new Float64Array(paramNames.map(name => {
      const [lo, hi] = this.paramBounds[name]!;
      return ((o.params[name] ?? 0) - lo) / ((hi - lo) + 1e-10);
    })));
    const yTrain = new Float64Array(this.observations_.map(o => o.score));
    const xNew = new Float64Array(paramNames.map(name => {
      const [lo, hi] = this.paramBounds[name]!;
      return ((candidate[name] ?? 0) - lo) / ((hi - lo) + 1e-10);
    }));

    const K = Array.from({ length: n }, (_, i) => new Float64Array(n).map((_, j) => this._rbf(xTrain[i]!, xTrain[j]!) + (i === j ? 1e-6 : 0)));
    const kStar = new Float64Array(n).map((_, i) => this._rbf(xNew, xTrain[i]!));
    const kStarStar = this._rbf(xNew, xNew);

    const KInvK = this._solveLinear(K, n, kStar);
    const mean = KInvK.reduce((s, v, i) => s + v * (yTrain[i] ?? 0), 0);
    const variance = Math.max(0, kStarStar - kStar.reduce((s, v, i) => s + v * (KInvK[i] ?? 0), 0));
    return { mean, std: Math.sqrt(variance) };
  }

  private _rbf(x: Float64Array, y: Float64Array): number {
    const d2 = x.reduce((s, v, j) => s + (v - (y[j] ?? 0)) ** 2, 0);
    return Math.exp(-0.5 * d2);
  }

  private _solveLinear(A: Float64Array[], n: number, b: Float64Array): Float64Array {
    const mat = A.map((row, j) => { const r = new Float64Array(n + 1); for (let k = 0; k < n; k++) r[k] = row[k] ?? 0; r[n] = b[j] ?? 0; return r; });
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) if (Math.abs(mat[row]?.[col] ?? 0) > Math.abs(mat[maxRow]?.[col] ?? 0)) maxRow = row;
      const tmp = mat[col]!; mat[col] = mat[maxRow]!; mat[maxRow] = tmp;
      const pivot = mat[col]?.[col] ?? 1e-10;
      for (let row = col + 1; row < n; row++) {
        const f = (mat[row]?.[col] ?? 0) / pivot;
        for (let k = col; k <= n; k++) mat[row]![k] = (mat[row]?.[k] ?? 0) - f * (mat[col]?.[k] ?? 0);
      }
    }
    const x = new Float64Array(n);
    for (let j = n - 1; j >= 0; j--) {
      let s = mat[j]?.[n] ?? 0;
      for (let k = j + 1; k < n; k++) s -= (mat[j]?.[k] ?? 0) * (x[k] ?? 0);
      x[j] = s / (mat[j]?.[j] ?? 1e-10);
    }
    return x;
  }
}

function normCDF(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989422820 * Math.exp(-(z * z) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))));
  return z > 0 ? 1 - p : p;
}

function normPDF(z: number): number { return Math.exp(-(z * z) / 2) / Math.sqrt(2 * Math.PI); }

export function timeSeriesSplit(n: number, nSplits: number = 5): Array<{ trainIdx: number[]; testIdx: number[] }> {
  const splits: Array<{ trainIdx: number[]; testIdx: number[] }> = [];
  const foldSize = Math.floor(n / (nSplits + 1));
  for (let i = 0; i < nSplits; i++) {
    const testStart = (i + 1) * foldSize;
    const testEnd = testStart + foldSize;
    splits.push({
      trainIdx: Array.from({ length: testStart }, (_, j) => j),
      testIdx: Array.from({ length: Math.min(foldSize, n - testStart) }, (_, j) => testStart + j)
    });
  }
  return splits;
}

export function groupTimeSeriesSplit(
  groups: Int32Array,
  nSplits: number = 5
): Array<{ trainIdx: number[]; testIdx: number[] }> {
  const uniqueGroups = [...new Set(Array.from(groups))].sort((a, b) => a - b);
  const nGroups = uniqueGroups.length;
  const splits: Array<{ trainIdx: number[]; testIdx: number[] }> = [];
  const groupSize = Math.floor(nGroups / (nSplits + 1));

  for (let i = 0; i < nSplits; i++) {
    const trainGroups = new Set(uniqueGroups.slice(0, (i + 1) * groupSize));
    const testGroups = new Set(uniqueGroups.slice((i + 1) * groupSize, (i + 2) * groupSize));
    splits.push({
      trainIdx: Array.from(groups).map((g, j) => trainGroups.has(g) ? j : -1).filter(j => j >= 0),
      testIdx: Array.from(groups).map((g, j) => testGroups.has(g) ? j : -1).filter(j => j >= 0)
    });
  }
  return splits;
}

export function learningCurve(
  estimator: { fit: (X: Float64Array[], y: Float64Array | Int32Array) => void; score: (X: Float64Array[], y: Float64Array | Int32Array) => number },
  X: Float64Array[],
  y: Float64Array | Int32Array,
  trainSizes: Float64Array = new Float64Array([0.1, 0.33, 0.55, 0.78, 1.0]),
  cv: number = 5
): { trainSizesAbs: Int32Array; trainScores: Float64Array[]; testScores: Float64Array[] } {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  const trainSizesAbs = new Int32Array(trainSizes.map(s => Math.max(1, Math.floor(s * n * (cv - 1) / cv))));
  const trainScores: Float64Array[] = [], testScores: Float64Array[] = [];

  for (const trainSize of trainSizesAbs) {
    const foldTrainScores: number[] = [], foldTestScores: number[] = [];
    for (let fold = 0; fold < cv; fold++) {
      const valStart = fold * foldSize, valEnd = Math.min(valStart + foldSize, n);
      const trainIdx: number[] = [], testIdx: number[] = [];
      for (let i = 0; i < n; i++) { if (i >= valStart && i < valEnd) testIdx.push(i); else trainIdx.push(i); }
      const subset = trainIdx.slice(0, trainSize);
      estimator.fit(subset.map(i => X[i]!), y instanceof Float64Array ? new Float64Array(subset.map(i => y[i] ?? 0)) : new Int32Array(subset.map(i => y[i] ?? 0)));
      foldTrainScores.push(estimator.score(subset.map(i => X[i]!), y instanceof Float64Array ? new Float64Array(subset.map(i => y[i] ?? 0)) : new Int32Array(subset.map(i => y[i] ?? 0))));
      foldTestScores.push(estimator.score(testIdx.map(i => X[i]!), y instanceof Float64Array ? new Float64Array(testIdx.map(i => y[i] ?? 0)) : new Int32Array(testIdx.map(i => y[i] ?? 0))));
    }
    trainScores.push(new Float64Array(foldTrainScores));
    testScores.push(new Float64Array(foldTestScores));
  }
  return { trainSizesAbs, trainScores, testScores };
}
