/**
 * Inspection extensions: LIME explanations, integrated gradients, anchor explanations
 * Port of sklearn.inspection extensions
 */

export interface BlackBoxModel {
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export interface LIMEExplanation {
  featureWeights: Float64Array;
  intercept: number;
  score: number;
  instance: Float64Array;
}

export class LIMETabularExplainer {
  nSamples: number;
  kernelWidth: number;
  randomState: number;
  featureSelection: "auto" | "highest_weights";
  nTopFeatures: number;

  constructor(opts: {
    nSamples?: number;
    kernelWidth?: number;
    randomState?: number;
    featureSelection?: "auto" | "highest_weights";
    nTopFeatures?: number;
  } = {}) {
    this.nSamples = opts.nSamples ?? 5000;
    this.kernelWidth = opts.kernelWidth ?? 0.75;
    this.nTopFeatures = opts.nTopFeatures ?? 10;
    this.randomState = opts.randomState ?? 0;
    this.featureSelection = opts.featureSelection ?? "auto";
  }

  explain(
    instance: Float64Array,
    model: BlackBoxModel,
    trainingData: Float64Array[]
  ): LIMEExplanation {
    const p = instance.length;
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const means = new Float64Array(p);
    const stds = new Float64Array(p);
    for (const xi of trainingData) for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) + (xi[j] ?? 0) / trainingData.length;
    for (const xi of trainingData) for (let j = 0; j < p; j++) stds[j] = (stds[j] ?? 0) + ((xi[j] ?? 0) - (means[j] ?? 0)) ** 2 / trainingData.length;
    for (let j = 0; j < p; j++) stds[j] = Math.sqrt(stds[j] ?? 0) + 1e-15;

    const samples: Float64Array[] = Array.from({ length: this.nSamples }, () => {
      const s = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const u1 = rng();
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
        s[j] = (means[j] ?? 0) + (stds[j] ?? 1) * z;
      }
      return s;
    });
    samples.unshift(instance.slice());

    const predictions = model.predictProba ? model.predictProba(samples) : null;
    const labels = model.predict(samples);
    const instanceLabel = labels[0] ?? 0;

    const weights = new Float64Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      let dist = 0;
      for (let j = 0; j < p; j++) dist += ((samples[i]![j] ?? 0) - (instance[j] ?? 0)) ** 2 / ((stds[j] ?? 1) ** 2);
      weights[i] = Math.exp(-dist / (this.kernelWidth ** 2 + 1e-15));
    }

    const yTarget = predictions
      ? Float64Array.from(predictions.map((p2, i) => p2[instanceLabel] ?? (labels[i] === instanceLabel ? 1 : 0)))
      : Float64Array.from(labels.map(l => l === instanceLabel ? 1.0 : 0.0));

    const XtWX = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
    const XtWy = new Float64Array(p + 1);
    for (let i = 0; i < samples.length; i++) {
      const xi = samples[i]!;
      const wi = weights[i] ?? 0;
      const yi = yTarget[i] ?? 0;
      for (let j = 0; j < p; j++) {
        XtWy[j] = (XtWy[j] ?? 0) + wi * (xi[j] ?? 0) * yi;
        for (let k = 0; k < p; k++) XtWX[j]![k] = (XtWX[j]![k] ?? 0) + wi * (xi[j] ?? 0) * (xi[k] ?? 0);
      }
      XtWy[p] = (XtWy[p] ?? 0) + wi * yi;
      for (let j = 0; j < p; j++) XtWX[p]![j] = (XtWX[p]![j] ?? 0) + wi * (xi[j] ?? 0);
      XtWX[p]![p] = (XtWX[p]![p] ?? 0) + wi;
    }
    for (let j = 0; j <= p; j++) XtWX[j]![j] = (XtWX[j]![j] ?? 0) + 1e-3;

    const coefs = new Float64Array(p + 1);
    for (let iter = 0; iter < 100; iter++) {
      for (let j = 0; j <= p; j++) {
        let s = XtWy[j] ?? 0;
        for (let k = 0; k <= p; k++) if (k !== j) s -= (XtWX[j]![k] ?? 0) * (coefs[k] ?? 0);
        coefs[j] = s / ((XtWX[j]![j] ?? 1) + 1e-15);
      }
    }

    let ssRes = 0;
    let ssTot = 0;
    const yMean = yTarget.reduce((a, b) => a + b, 0) / yTarget.length;
    for (let i = 0; i < samples.length; i++) {
      const xi = samples[i]!;
      let pred = coefs[p] ?? 0;
      for (let j = 0; j < p; j++) pred += (coefs[j] ?? 0) * (xi[j] ?? 0);
      ssRes += (weights[i] ?? 0) * ((yTarget[i] ?? 0) - pred) ** 2;
      ssTot += (weights[i] ?? 0) * ((yTarget[i] ?? 0) - yMean) ** 2;
    }
    const score = 1 - ssRes / (ssTot + 1e-15);

    return {
      featureWeights: coefs.slice(0, p),
      intercept: coefs[p] ?? 0,
      score,
      instance,
    };
  }
}

export function integratedGradients(
  instance: Float64Array,
  baseline: Float64Array,
  gradFn: (x: Float64Array) => Float64Array,
  nSteps = 50
): Float64Array {
  const p = instance.length;
  const ig = new Float64Array(p);
  for (let s = 0; s <= nSteps; s++) {
    const alpha = s / nSteps;
    const interpolated = Float64Array.from({ length: p }, (_, j) => (baseline[j] ?? 0) + alpha * ((instance[j] ?? 0) - (baseline[j] ?? 0)));
    const grad = gradFn(interpolated);
    const weight = s === 0 || s === nSteps ? 0.5 : 1.0;
    for (let j = 0; j < p; j++) ig[j] = (ig[j] ?? 0) + weight * (grad[j] ?? 0);
  }
  for (let j = 0; j < p; j++) ig[j] = (ig[j] ?? 0) / nSteps * ((instance[j] ?? 0) - (baseline[j] ?? 0));
  return ig;
}

export class SHAPKernelExplainer {
  model: BlackBoxModel;
  background: Float64Array[];

  constructor(model: BlackBoxModel, background: Float64Array[]) {
    this.model = model;
    this.background = background;
  }

  shapValues(instance: Float64Array, nSamples = 100): Float64Array {
    const p = instance.length;
    let seed = 0;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const baseVal = (() => {
      const preds = this.model.predict(this.background);
      return preds.reduce((a, b) => a + b, 0) / preds.length;
    })();
    const shapVals = new Float64Array(p);
    for (let iter = 0; iter < nSamples; iter++) {
      const coalitionSize = Math.floor(rng() * (p + 1));
      const features = Array.from({ length: p }, (_, i) => i);
      for (let i = features.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const tmp = features[i]!; features[i] = features[j]!; features[j] = tmp; }
      const coalition = new Set(features.slice(0, coalitionSize));
      const background = this.background[Math.floor(rng() * this.background.length)]!;
      const masked = Float64Array.from({ length: p }, (_, j) => coalition.has(j) ? (instance[j] ?? 0) : (background[j] ?? 0));
      const pred = this.model.predict([masked])[0] ?? 0;
      for (const j of coalition) {
        const maskedWithout = masked.slice();
        maskedWithout[j] = background[j] ?? 0;
        const predWithout = this.model.predict([maskedWithout])[0] ?? 0;
        shapVals[j] = (shapVals[j] ?? 0) + (pred - predWithout) / (nSamples + 1e-15);
      }
      void baseVal;
    }
    return shapVals;
  }
}
