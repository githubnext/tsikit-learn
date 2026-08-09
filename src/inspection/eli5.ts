/**
 * Feature importance explanation utilities (ELI5-style).
 * Mirrors scikit-learn's inspection and ELI5 feature weights.
 */

export interface FeatureWeight {
  feature: string;
  weight: number;
  std?: number;
}

export interface ExplainedPrediction {
  target: number;
  score: number;
  featureWeights: FeatureWeight[];
}

export interface WeightExplanation {
  estimatorName: string;
  targets: ExplainedPrediction[];
}

/**
 * Explain weights of a linear model.
 */
export function explainWeights(
  coef: Float64Array | Float64Array[],
  featureNames?: string[],
  classNames?: string[],
  intercept?: Float64Array | number,
): WeightExplanation {
  const isMulticlass = Array.isArray(coef);
  const targets: ExplainedPrediction[] = [];

  const coefs = isMulticlass ? coef : [coef];

  for (let cls = 0; cls < coefs.length; cls++) {
    const c = coefs[cls]!;
    const weights: FeatureWeight[] = Array.from(c, (w, i) => ({
      feature: featureNames?.[i] ?? `x${i}`,
      weight: w,
    })).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

    const inter = Array.isArray(intercept)
      ? (intercept[cls] ?? 0)
      : typeof intercept === "number"
        ? intercept
        : 0;

    targets.push({
      target: cls,
      score: inter,
      featureWeights: weights,
    });
  }

  return {
    estimatorName: "LinearModel",
    targets,
  };
}

/**
 * Explain a prediction using LIME-style local perturbations.
 */
export interface LIMEOptions {
  nSamples?: number;
  kernel?: "gaussian" | "uniform";
  kernelWidth?: number;
  randomState?: number;
}

export function explainPredictionLinear(
  estimator: {
    predict: (X: Float64Array[]) => Float64Array | Int32Array;
  },
  instance: Float64Array,
  featureNames?: string[],
  options: LIMEOptions = {},
): ExplainedPrediction {
  const {
    nSamples = 500,
    kernel = "gaussian",
    kernelWidth,
    randomState = 42,
  } = options;

  const nFeatures = instance.length;
  const kw = kernelWidth ?? Math.sqrt(nFeatures) * 0.75;

  // Generate perturbed samples around the instance
  let s = randomState;
  const rng = (): number => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  const samples: Float64Array[] = [];
  const distances: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const sample = Float64Array.from(
      instance,
      (v) => v + (rng() * 2 - 1) * 0.1,
    );
    let d = 0;
    for (let j = 0; j < nFeatures; j++)
      d += ((sample[j] ?? 0) - (instance[j] ?? 0)) ** 2;
    distances.push(Math.sqrt(d));
    samples.push(sample);
  }

  const weights = distances.map((d) =>
    kernel === "gaussian" ? Math.exp(-(d * d) / (kw * kw)) : 1,
  );

  const preds = estimator.predict(samples);

  // Weighted least squares (ridge) for local explanation
  const WX = samples.map((s, i) =>
    Float64Array.from(s, (v) => v * Math.sqrt(weights[i] ?? 1)),
  );
  const wy = Float64Array.from(preds, (v, i) => v * Math.sqrt(weights[i] ?? 1));

  // Normal equations: (X^T X + I) w = X^T y (simple ridge)
  const coefs = new Float64Array(nFeatures);
  for (let j = 0; j < nFeatures; j++) {
    let xTx = 1; // L2 reg
    let xTy = 0;
    for (let i = 0; i < nSamples; i++) {
      xTx += (WX[i]?.[j] ?? 0) ** 2;
      xTy += (WX[i]?.[j] ?? 0) * (wy[i] ?? 0);
    }
    coefs[j] = xTy / xTx;
  }

  const featureWeights: FeatureWeight[] = Array.from(coefs, (w, j) => ({
    feature: featureNames?.[j] ?? `x${j}`,
    weight: w,
  })).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  return {
    target: 0,
    score: (estimator.predict([instance])[0] as number) ?? 0,
    featureWeights,
  };
}
