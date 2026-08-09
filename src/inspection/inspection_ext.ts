/**
 * Extended inspection utilities: PartialDependenceExt, feature importance aggregation
 */

export interface PDPResult {
  gridValues: Float64Array;
  averageValues: Float64Array;
}

export function computePartialDependenceExt(
  predictFn: (X: Float64Array[]) => Float64Array,
  X: Float64Array[],
  featureIdx: number,
  gridResolution = 100
): PDPResult {
  const featureVals = new Float64Array(X.map((row) => row[featureIdx] ?? 0));
  const sorted = Float64Array.from(featureVals).sort();
  const n = sorted.length;
  const gridValues = new Float64Array(gridResolution);
  for (let i = 0; i < gridResolution; i++) {
    const idx = Math.floor((i / (gridResolution - 1)) * (n - 1));
    gridValues[i] = sorted[idx] ?? 0;
  }
  const averageValues = new Float64Array(gridResolution);
  for (let gi = 0; gi < gridResolution; gi++) {
    const Xmod = X.map((row) => {
      const copy = new Float64Array(row);
      copy[featureIdx] = gridValues[gi] ?? 0;
      return copy;
    });
    const preds = predictFn(Xmod);
    averageValues[gi] = preds.reduce((a, b) => a + b, 0) / preds.length;
  }
  return { gridValues, averageValues };
}

export interface ShapleyEstimate {
  featureIdx: number;
  shapValue: number;
}

export class ShapleyImportanceEstimator {
  private nSamples: number;
  private nPermutations: number;

  constructor(nSamples = 100, nPermutations = 10) {
    this.nSamples = nSamples;
    this.nPermutations = nPermutations;
  }

  explain(
    predictFn: (X: Float64Array[]) => Float64Array,
    X: Float64Array[],
    instanceIdx: number
  ): ShapleyEstimate[] {
    const nFeatures = X[0]?.length ?? 0;
    const instance = X[instanceIdx] ?? new Float64Array(nFeatures);
    const shapValues: ShapleyEstimate[] = [];

    for (let f = 0; f < nFeatures; f++) {
      let totalContrib = 0;
      for (let p = 0; p < this.nPermutations; p++) {
        const bgIdx = Math.floor(Math.random() * X.length);
        const background = X[bgIdx] ?? instance;
        const withFeature = new Float64Array(background);
        withFeature[f] = instance[f] ?? 0;
        const without = new Float64Array(background);
        const predWith = predictFn([withFeature]);
        const predWithout = predictFn([without]);
        totalContrib += (predWith[0] ?? 0) - (predWithout[0] ?? 0);
      }
      shapValues.push({ featureIdx: f, shapValue: totalContrib / this.nPermutations });
    }
    return shapValues;
  }
}

export class LIMEExplainerExt {
  private nSamples: number;
  private kernelWidth: number;

  constructor(nSamples = 500, kernelWidth = 0.25) {
    this.nSamples = nSamples;
    this.kernelWidth = kernelWidth;
  }

  explain(
    predictFn: (X: Float64Array[]) => Float64Array,
    instance: Float64Array,
    X: Float64Array[]
  ): Float64Array {
    const nFeatures = instance.length;
    const nSamples = Math.min(this.nSamples, X.length);

    // Sample perturbations
    const perturbations: Float64Array[] = [];
    const weights: number[] = [];
    for (let s = 0; s < nSamples; s++) {
      const bgIdx = Math.floor(Math.random() * X.length);
      const bg = X[bgIdx] ?? instance;
      const perturb = new Float64Array(nFeatures);
      let dist = 0;
      for (let j = 0; j < nFeatures; j++) {
        const v = Math.random() < 0.5 ? (instance[j] ?? 0) : (bg[j] ?? 0);
        perturb[j] = v;
        dist += (v - (instance[j] ?? 0)) ** 2;
      }
      perturbations.push(perturb);
      weights.push(Math.exp(-dist / (2 * this.kernelWidth ** 2)));
    }

    const preds = predictFn(perturbations);

    // Weighted least squares
    const XtW = Array.from({ length: nFeatures }, (_, j) => {
      let sum = 0;
      for (let s = 0; s < nSamples; s++) sum += (perturbations[s]![j] ?? 0) * (weights[s] ?? 0);
      return sum;
    });
    const coefs = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let num = 0, denom = 1e-8;
      for (let s = 0; s < nSamples; s++) {
        const x = perturbations[s]![j] ?? 0;
        const w = weights[s] ?? 0;
        num += w * x * (preds[s] ?? 0);
        denom += w * x * x;
      }
      coefs[j] = num / denom;
    }
    return coefs;
  }
}
