/**
 * Extended scorer utilities: makeScorer, PermutationImportanceScorer
 */

export type MetricFn = (yTrue: Float64Array, yPred: Float64Array) => number;

export interface ScorerOptions {
  greater_is_better?: boolean;
  needs_proba?: boolean;
  needs_threshold?: boolean;
}

export class Scorer {
  private fn: MetricFn;
  private sign: number;

  constructor(fn: MetricFn, options: ScorerOptions = {}) {
    this.fn = fn;
    this.sign = (options.greater_is_better ?? true) ? 1 : -1;
  }

  score(yTrue: Float64Array, yPred: Float64Array): number {
    return this.sign * this.fn(yTrue, yPred);
  }
}

export function makeScorer(fn: MetricFn, options: ScorerOptions = {}): Scorer {
  return new Scorer(fn, options);
}

export class PermutationImportanceScorer {
  private baseScore: number;
  private importances: Float64Array;
  private nFeatures: number;
  nRepeats: number;

  constructor(nFeatures: number, nRepeats = 5) {
    this.nFeatures = nFeatures;
    this.nRepeats = nRepeats;
    this.baseScore = 0;
    this.importances = new Float64Array(nFeatures);
  }

  fit(
    X: Float64Array[],
    y: Float64Array,
    scorer: MetricFn,
    predictFn: (X: Float64Array[]) => Float64Array
  ): this {
    this.baseScore = scorer(y, predictFn(X));
    for (let f = 0; f < this.nFeatures; f++) {
      let totalDrop = 0;
      for (let r = 0; r < this.nRepeats; r++) {
        const Xperm = X.map((row) => {
          const copy = new Float64Array(row);
          const j = Math.floor(Math.random() * X.length);
          const tmp = copy[f] ?? 0;
          copy[f] = (X[j] ?? row)[f] ?? 0;
          copy[j < copy.length ? j : 0] = tmp;
          return copy;
        });
        totalDrop += this.baseScore - scorer(y, predictFn(Xperm));
      }
      this.importances[f] = totalDrop / this.nRepeats;
    }
    return this;
  }

  getImportances(): Float64Array {
    return this.importances;
  }
}

export class CheckScoringMixin {
  checkScoring(
    scoring: string | MetricFn | null
  ): MetricFn {
    if (typeof scoring === "function") return scoring;
    if (scoring === "r2") {
      return (yTrue: Float64Array, yPred: Float64Array) => {
        const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
        let ss_tot = 0;
        let ss_res = 0;
        for (let i = 0; i < yTrue.length; i++) {
          ss_tot += ((yTrue[i] ?? 0) - mean) ** 2;
          ss_res += ((yTrue[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
        }
        return ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;
      };
    }
    return (yTrue: Float64Array, yPred: Float64Array) => {
      let correct = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if ((yTrue[i] ?? 0) === (yPred[i] ?? 0)) correct++;
      }
      return correct / yTrue.length;
    };
  }
}
