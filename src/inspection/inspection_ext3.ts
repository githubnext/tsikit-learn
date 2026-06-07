/**
 * Model inspection utilities — partial dependence, ICE, and permutation importance.
 */

export interface PredictorLike {
  predict(X: Float64Array[]): Float64Array | Int32Array;
}

export interface PartialDependenceResult {
  pdpValues: Float64Array;
  gridValues: Float64Array;
  featureIdx: number;
}

export function partialDependence(
  estimator: PredictorLike,
  X: Float64Array[],
  features: number[],
  nGridPoints = 100,
): PartialDependenceResult[] {
  const results: PartialDependenceResult[] = [];
  for (const feat of features) {
    const vals = X.map((row) => row[feat] ?? 0).sort((a, b) => a - b);
    const mn = vals[0] ?? 0, mx = vals[vals.length - 1] ?? 1;
    const gridValues = Float64Array.from({ length: nGridPoints }, (_, i) => mn + (i / (nGridPoints - 1)) * (mx - mn));
    const pdpValues = Float64Array.from(gridValues, (gv) => {
      const Xcopy = X.map((row) => {
        const newRow = new Float64Array(row);
        newRow[feat] = gv;
        return newRow;
      });
      const preds = estimator.predict(Xcopy);
      return Array.from(preds).reduce((s, v) => s + v, 0) / preds.length;
    });
    results.push({ pdpValues, gridValues, featureIdx: feat });
  }
  return results;
}

export interface ICEResult {
  iceValues: Float64Array[];
  gridValues: Float64Array;
  featureIdx: number;
}

export function individualConditionalExpectation(
  estimator: PredictorLike,
  X: Float64Array[],
  feature: number,
  nGridPoints = 50,
): ICEResult {
  const vals = X.map((row) => row[feature] ?? 0).sort((a, b) => a - b);
  const mn = vals[0] ?? 0, mx = vals[vals.length - 1] ?? 1;
  const gridValues = Float64Array.from({ length: nGridPoints }, (_, i) => mn + (i / (nGridPoints - 1)) * (mx - mn));
  const iceValues: Float64Array[] = X.map((row) => {
    return Float64Array.from(gridValues, (gv) => {
      const newRow = new Float64Array(row);
      newRow[feature] = gv;
      const pred = estimator.predict([newRow]);
      return pred[0] ?? 0;
    });
  });
  return { iceValues, gridValues, featureIdx: feature };
}

export interface PermutationImportanceResult {
  importancesMean: Float64Array;
  importancesStd: Float64Array;
  importances: Float64Array[];
}

export function permutationImportance(
  estimator: { predict(X: Float64Array[]): Float64Array | Int32Array },
  X: Float64Array[],
  y: Float64Array | Int32Array,
  scorer: (yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number,
  nRepeats = 5,
): PermutationImportanceResult {
  const p = X[0]?.length ?? 0;
  const baseScore = scorer(y, estimator.predict(X));
  const importances: Float64Array[] = Array.from({ length: p }, () => new Float64Array(nRepeats));

  for (let feat = 0; feat < p; feat++) {
    for (let r = 0; r < nRepeats; r++) {
      const Xperm = X.map((row) => new Float64Array(row));
      const colVals = X.map((row) => row[feat] ?? 0);
      for (let i = colVals.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = colVals[i]; colVals[i] = colVals[j] as number; colVals[j] = tmp as number;
      }
      for (let i = 0; i < X.length; i++) (Xperm[i] as Float64Array)[feat] = colVals[i] ?? 0;
      const permScore = scorer(y, estimator.predict(Xperm));
      (importances[feat] as Float64Array)[r] = baseScore - permScore;
    }
  }

  const importancesMean = Float64Array.from({ length: p }, (_, feat) => {
    const imp = importances[feat] as Float64Array;
    return imp.reduce((s, v) => s + v, 0) / nRepeats;
  });
  const importancesStd = Float64Array.from({ length: p }, (_, feat) => {
    const imp = importances[feat] as Float64Array;
    const mean = imp.reduce((s, v) => s + v, 0) / nRepeats;
    return Math.sqrt(imp.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(nRepeats - 1, 1));
  });

  return { importancesMean, importancesStd, importances };
}

export class ModelInspector {
  private estimator: PredictorLike;

  constructor(estimator: PredictorLike) {
    this.estimator = estimator;
  }

  partialDependence(X: Float64Array[], features: number[], nGridPoints = 100): PartialDependenceResult[] {
    return partialDependence(this.estimator, X, features, nGridPoints);
  }

  ice(X: Float64Array[], feature: number, nGridPoints = 50): ICEResult {
    return individualConditionalExpectation(this.estimator, X, feature, nGridPoints);
  }

  shapApproximation(X: Float64Array[], background: Float64Array[], maxSamples = 100): Float64Array[] {
    const n = Math.min(X.length, maxSamples);
    const p = X[0]?.length ?? 0;
    return X.slice(0, n).map((x) => {
      const shapValues = new Float64Array(p);
      for (let feat = 0; feat < p; feat++) {
        let marginal = 0;
        for (const bg of background) {
          const withFeat = new Float64Array(bg);
          withFeat[feat] = x[feat] ?? 0;
          const withoutFeat = new Float64Array(bg);
          const predWith = this.estimator.predict([withFeat])[0] ?? 0;
          const predWithout = this.estimator.predict([withoutFeat])[0] ?? 0;
          marginal += predWith - predWithout;
        }
        shapValues[feat] = marginal / Math.max(background.length, 1);
      }
      return shapValues;
    });
  }
}
