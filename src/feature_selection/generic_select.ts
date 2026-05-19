/**
 * GenericUnivariateSelect — configurable univariate feature selection.
 * Mirrors sklearn.feature_selection.GenericUnivariateSelect.
 */

export type SelectionMode = "percentile" | "k_best" | "fpr" | "fdr" | "fwe";

export interface GenericUnivariateSelectOptions {
  scoreFunc?: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pvalues: Float64Array };
  mode?: SelectionMode;
  param?: number;
}

/**
 * Univariate feature selector with configurable selection mode.
 */
export class GenericUnivariateSelect {
  private scoreFunc: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pvalues: Float64Array };
  mode: SelectionMode;
  param: number;

  scores_: Float64Array | null = null;
  pvalues_: Float64Array | null = null;
  supportMask_: boolean[] | null = null;

  constructor(options: GenericUnivariateSelectOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? defaultScoreFunc;
    this.mode = options.mode ?? "percentile";
    this.param = options.param ?? 10;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const { scores, pvalues } = this.scoreFunc(X, y);
    this.scores_ = scores;
    this.pvalues_ = pvalues;
    const nFeatures = scores.length;

    const ranked = Array.from({ length: nFeatures }, (_, i) => i)
      .sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));

    this.supportMask_ = new Array(nFeatures).fill(false);

    switch (this.mode) {
      case "percentile": {
        const k = Math.max(1, Math.round(this.param / 100 * nFeatures));
        for (let i = 0; i < k; i++) this.supportMask_[ranked[i]!] = true;
        break;
      }
      case "k_best": {
        const k = Math.min(Math.round(this.param), nFeatures);
        for (let i = 0; i < k; i++) this.supportMask_[ranked[i]!] = true;
        break;
      }
      case "fpr":
      case "fdr":
      case "fwe": {
        // False positive rate / discovery rate / family-wise error
        const alpha = this.param;
        for (let j = 0; j < nFeatures; j++) {
          if ((pvalues[j] ?? 1) < alpha) this.supportMask_[j] = true;
        }
        break;
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.supportMask_) throw new Error("GenericUnivariateSelect not fitted");
    const selectedIdxs = this.supportMask_.reduce<number[]>((acc, v, i) => {
      if (v) acc.push(i);
      return acc;
    }, []);
    return X.map(row => new Float64Array(selectedIdxs.map(j => row[j] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupportMask(): boolean[] {
    if (!this.supportMask_) throw new Error("GenericUnivariateSelect not fitted");
    return [...this.supportMask_];
  }

  getFeatureNamesOut(inputFeatures?: string[]): string[] {
    if (!this.supportMask_) throw new Error("GenericUnivariateSelect not fitted");
    if (!inputFeatures) return this.supportMask_.reduce<string[]>((acc, v, i) => {
      if (v) acc.push(`x${i}`);
      return acc;
    }, []);
    return inputFeatures.filter((_, i) => this.supportMask_![i]);
  }
}

function defaultScoreFunc(X: Float64Array[], y: Float64Array | Int32Array): { scores: Float64Array; pvalues: Float64Array } {
  const nFeatures = X[0]?.length ?? 0;
  const nSamples = X.length;
  const scores = new Float64Array(nFeatures);
  const pvalues = new Float64Array(nFeatures).fill(0.5);

  // ANOVA F-test
  const classes = Array.from(new Set(Array.from(y as Int32Array)));
  for (let j = 0; j < nFeatures; j++) {
    const vals = Array.from({ length: nSamples }, (_, i) => X[i]?.[j] ?? 0);
    const globalMean = vals.reduce((s, v) => s + v, 0) / nSamples;
    const groupMeans = classes.map(cls => {
      const clsVals = vals.filter((_, i) => y[i] === cls);
      return clsVals.reduce((s, v) => s + v, 0) / (clsVals.length || 1);
    });
    const groupCounts = classes.map(cls => vals.filter((_, i) => y[i] === cls).length);
    let ssBetween = 0;
    let ssWithin = 0;
    groupMeans.forEach((gm, ci) => {
      ssBetween += (groupCounts[ci] ?? 0) * (gm - globalMean) ** 2;
    });
    for (let i = 0; i < nSamples; i++) {
      const ci = classes.indexOf(y[i] as number);
      ssWithin += (vals[i]! - (groupMeans[ci] ?? 0)) ** 2;
    }
    const dfBetween = Math.max(classes.length - 1, 1);
    const dfWithin = Math.max(nSamples - classes.length, 1);
    scores[j] = (ssBetween / dfBetween) / ((ssWithin / dfWithin) || 1e-10);
    pvalues[j] = Math.exp(-scores[j]! / 2);
  }
  return { scores, pvalues };
}

/**
 * SelectPercentileExt — select features in the top percentile by score.
 */
export class SelectPercentileExt extends GenericUnivariateSelect {
  constructor(
    scoreFunc?: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pvalues: Float64Array },
    percentile = 10
  ) {
    super({ scoreFunc, mode: "percentile", param: percentile });
  }
}
