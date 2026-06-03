/**
 * Inspection extensions: counterfactual explanations, anchor explanations, contrast sets.
 * Mirrors sklearn.inspection additional methods.
 */

import { BaseEstimator } from "../base.js";

type Predictor = {
  predict(X: Float64Array[]): Int32Array;
};

/** Generate counterfactual explanation: find minimal change to flip prediction. */
export function counterfactualExplanation(
  estimator: Predictor,
  x: Float64Array,
  target_class: number,
  X_train: Float64Array[],
  max_iter = 1000,
): { counterfactual: Float64Array; distance: number; n_changed_features: number } {
  const d = x.length;
  const originalPred = estimator.predict([x])[0] ?? 0;
  if (originalPred === target_class) {
    return { counterfactual: new Float64Array(x), distance: 0, n_changed_features: 0 };
  }

  let best = new Float64Array(x);
  let bestDist = Number.POSITIVE_INFINITY;
  let bestChanged = d;

  // Random perturbation search
  for (let iter = 0; iter < max_iter; iter++) {
    const candidate = new Float64Array(x);
    const nPerturb = Math.ceil(Math.random() * d);
    const features = new Int32Array(d).map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, nPerturb);
    const refSample = X_train[Math.floor(Math.random() * X_train.length)]!;
    for (const f of features) candidate[f] = refSample[f] ?? 0;

    if ((estimator.predict([candidate])[0] ?? 0) === target_class) {
      let dist = 0;
      let changed = 0;
      for (let f = 0; f < d; f++) {
        const diff = Math.abs((candidate[f] ?? 0) - (x[f] ?? 0));
        dist += diff;
        if (diff > 1e-6) changed++;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestChanged = changed;
        best = candidate;
      }
    }
  }
  return { counterfactual: best, distance: bestDist, n_changed_features: bestChanged };
}

export interface AnchorExplanationResult {
  anchor: Array<{ feature: number; condition: string; threshold: number }>;
  precision: number;
  coverage: number;
}

/** Anchor explanation: find rule-based anchors for a prediction. */
export function anchorExplanation(
  estimator: Predictor,
  x: Float64Array,
  X_train: Float64Array[],
  n_samples = 500,
  precision_threshold = 0.95,
): AnchorExplanationResult {
  const d = x.length;
  const prediction = estimator.predict([x])[0] ?? 0;
  const anchor: Array<{ feature: number; condition: string; threshold: number }> = [];
  const usedFeatures = new Set<number>();

  for (let a = 0; a < d; a++) {
    let bestFeature = -1;
    let bestPrecision = -1;
    let bestThreshold = 0;
    let bestCondition = ">";

    for (let f = 0; f < d; f++) {
      if (usedFeatures.has(f)) continue;
      const threshold = x[f] ?? 0;
      for (const cond of [">", "<="]) {
        // Sample from neighborhood satisfying current anchor + new condition
        const neighbors = X_train.filter(row => {
          for (const anc of anchor) {
            const v = row[anc.feature] ?? 0;
            if (anc.condition === ">" && v <= anc.threshold) return false;
            if (anc.condition === "<=" && v > anc.threshold) return false;
          }
          const v = row[f] ?? 0;
          if (cond === ">" && v <= threshold) return false;
          if (cond === "<=" && v > threshold) return false;
          return true;
        }).slice(0, n_samples);

        if (neighbors.length < 10) continue;
        const preds = estimator.predict(neighbors);
        const prec = Array.from(preds).filter(p => p === prediction).length / preds.length;
        if (prec > bestPrecision) {
          bestPrecision = prec;
          bestFeature = f;
          bestThreshold = threshold;
          bestCondition = cond;
        }
      }
    }

    if (bestFeature >= 0 && bestPrecision > 0.5) {
      anchor.push({ feature: bestFeature, condition: bestCondition, threshold: bestThreshold });
      usedFeatures.add(bestFeature);
    }
    if (bestPrecision >= precision_threshold) break;
  }

  const satisfying = X_train.filter(row => {
    for (const anc of anchor) {
      const v = row[anc.feature] ?? 0;
      if (anc.condition === ">" && v <= anc.threshold) return false;
      if (anc.condition === "<=" && v > anc.threshold) return false;
    }
    return true;
  });
  const preds = estimator.predict(satisfying);
  const precision = satisfying.length > 0 ? Array.from(preds).filter(p => p === prediction).length / preds.length : 0;
  const coverage = satisfying.length / (X_train.length || 1);

  return { anchor, precision, coverage };
}

/** Model-agnostic feature attribution via expected gradients (approximation). */
export function expectedGradients(
  estimator: { predict(X: Float64Array[]): Float64Array },
  x: Float64Array,
  background: Float64Array[],
  n_samples = 50,
): Float64Array {
  const d = x.length;
  const attributions = new Float64Array(d);

  for (let s = 0; s < n_samples; s++) {
    const ref = background[Math.floor(Math.random() * background.length)]!;
    const alpha = Math.random();
    const interpolated = new Float64Array(d).map((_, f) => (ref[f] ?? 0) + alpha * ((x[f] ?? 0) - (ref[f] ?? 0)));
    const eps = 1e-4;
    for (let f = 0; f < d; f++) {
      const xPlus = new Float64Array(interpolated);
      xPlus[f] = (xPlus[f] ?? 0) + eps;
      const xMinus = new Float64Array(interpolated);
      xMinus[f] = (xMinus[f] ?? 0) - eps;
      const grad = ((estimator.predict([xPlus])[0] ?? 0) - (estimator.predict([xMinus])[0] ?? 0)) / (2 * eps);
      attributions[f] = (attributions[f] ?? 0) + grad * ((x[f] ?? 0) - (ref[f] ?? 0)) / n_samples;
    }
  }
  return attributions;
}

/** Integrated gradients for a differentiable model. */
export class IntegratedGradients extends BaseEstimator {
  n_steps: number;
  constructor(n_steps = 50) {
    super();
    this.n_steps = n_steps;
  }

  attribute(
    estimator: { predict(X: Float64Array[]): Float64Array },
    x: Float64Array,
    baseline: Float64Array,
  ): Float64Array {
    const d = x.length;
    const attrs = new Float64Array(d);
    const eps = 1e-4;
    for (let step = 0; step <= this.n_steps; step++) {
      const alpha = step / this.n_steps;
      const interp = new Float64Array(d).map((_, f) => (baseline[f] ?? 0) + alpha * ((x[f] ?? 0) - (baseline[f] ?? 0)));
      for (let f = 0; f < d; f++) {
        const xPlus = new Float64Array(interp);
        xPlus[f] = (xPlus[f] ?? 0) + eps;
        const xMinus = new Float64Array(interp);
        xMinus[f] = (xMinus[f] ?? 0) - eps;
        const grad = ((estimator.predict([xPlus])[0] ?? 0) - (estimator.predict([xMinus])[0] ?? 0)) / (2 * eps);
        const w = step === 0 || step === this.n_steps ? 0.5 : 1;
        attrs[f] = (attrs[f] ?? 0) + w * grad;
      }
    }
    const scale = 1 / this.n_steps;
    for (let f = 0; f < d; f++) attrs[f] = (attrs[f] ?? 0) * scale * ((x[f] ?? 0) - (baseline[f] ?? 0));
    return attrs;
  }
}
