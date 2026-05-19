/**
 * Partial dependence plots and Individual Conditional Expectation (ICE).
 * Mirrors sklearn.inspection.partial_dependence and PartialDependenceDisplay.
 */

export interface PartialDependenceOptions {
  percentiles?: [number, number];
  gridResolution?: number;
  kind?: "average" | "individual" | "both";
}

export interface PartialDependencePlotData {
  averages: Float64Array[] | null;
  individual: Float64Array[][] | null;
  gridValues: Float64Array[];
  features: number[];
}

/**
 * Compute the partial dependence of features.
 * For each feature (or pair), averages predictions while varying that feature.
 */
export function computePartialDependence(
  estimator: {
    predict(X: Float64Array[]): Float64Array;
  },
  X: Float64Array[],
  features: number[],
  options: PartialDependenceOptions = {}
): PartialDependencePlotData {
  const percentiles = options.percentiles ?? [0.05, 0.95];
  const gridResolution = options.gridResolution ?? 100;
  const kind = options.kind ?? "average";
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;

  const gridValues: Float64Array[] = [];
  const averages: Float64Array[] = [];
  const individual: Float64Array[][] = [];

  for (const featureIdx of features) {
    if (featureIdx >= nFeatures) {
      throw new RangeError(`Feature index ${featureIdx} out of range`);
    }

    // Get feature values and compute quantile range
    const vals = Array.from({ length: nSamples }, (_, i) => X[i]?.[featureIdx] ?? 0);
    vals.sort((a, b) => a - b);

    const lowerIdx = Math.floor(percentiles[0] * nSamples);
    const upperIdx = Math.min(Math.ceil(percentiles[1] * nSamples), nSamples - 1);
    const lower = vals[lowerIdx] ?? 0;
    const upper = vals[upperIdx] ?? 1;

    // Create grid
    const grid = new Float64Array(gridResolution);
    for (let g = 0; g < gridResolution; g++) {
      grid[g] = lower + (g / (gridResolution - 1)) * (upper - lower);
    }
    gridValues.push(grid);

    // Compute partial dependence
    if (kind === "average" || kind === "both") {
      const pdp = new Float64Array(gridResolution);
      for (let g = 0; g < gridResolution; g++) {
        const Xmod = X.map(row => {
          const newRow = new Float64Array(row);
          newRow[featureIdx] = grid[g] ?? 0;
          return newRow;
        });
        const preds = estimator.predict(Xmod);
        pdp[g] = preds.reduce((s, v) => s + v, 0) / preds.length;
      }
      averages.push(pdp);
    }

    if (kind === "individual" || kind === "both") {
      const iceCurves: Float64Array[] = [];
      for (let i = 0; i < nSamples; i++) {
        const iceCurve = new Float64Array(gridResolution);
        for (let g = 0; g < gridResolution; g++) {
          const row = new Float64Array(X[i]!);
          row[featureIdx] = grid[g] ?? 0;
          const pred = estimator.predict([row]);
          iceCurve[g] = pred[0] ?? 0;
        }
        iceCurves.push(iceCurve);
      }
      individual.push(iceCurves);
    }
  }

  return {
    averages: averages.length > 0 ? averages : null,
    individual: individual.length > 0 ? individual : null,
    gridValues,
    features,
  };
}

/**
 * Partial Dependence Display class.
 */
export class PartialDependencePlot {
  pdResult: PartialDependencePlotData;
  featureNames: string[] | null;

  constructor(pdResult: PartialDependencePlotData, featureNames: string[] | null = null) {
    this.pdResult = pdResult;
    this.featureNames = featureNames;
  }

  static fromEstimator(
    estimator: { predict(X: Float64Array[]): Float64Array },
    X: Float64Array[],
    features: number[],
    options: PartialDependenceOptions & { featureNames?: string[] } = {}
  ): PartialDependencePlot {
    const { featureNames = null, ...pdOptions } = options;
    const pdResult = computePartialDependence(estimator, X, features, pdOptions);
    return new PartialDependencePlot(pdResult, featureNames);
  }

  toJSON(): object {
    return {
      features: this.pdResult.features,
      featureNames: this.featureNames,
      gridResolution: this.pdResult.gridValues[0]?.length ?? 0,
      hasAverages: this.pdResult.averages !== null,
      hasIndividual: this.pdResult.individual !== null,
    };
  }
}
