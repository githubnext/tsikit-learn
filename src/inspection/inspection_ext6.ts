/**
 * Inspection extensions: model inspection utilities.
 * Mirrors sklearn.inspection advanced utilities.
 */

import { BaseEstimator } from "../base.js";

type PredictingEstimator = {
  predict(X: Float64Array[]): Float64Array | Int32Array;
  fit(X: Float64Array[], y: Float64Array | Int32Array): unknown;
};

/** Individual Conditional Expectation (ICE) plot data. */
export function iceValues(
  estimator: PredictingEstimator,
  X: Float64Array[],
  feature_idx: number,
  grid_resolution = 100,
  percentiles: [number, number] = [0.05, 0.95],
): { grid: Float64Array; ice: Float64Array[] } {
  const colVals = X.map((xi) => xi[feature_idx] ?? 0).sort((a, b) => a - b);
  const lo = colVals[Math.floor(percentiles[0] * colVals.length)] ?? colVals[0] ?? 0;
  const hi = colVals[Math.ceil(percentiles[1] * colVals.length)] ?? colVals[colVals.length - 1] ?? 1;
  const grid = new Float64Array(grid_resolution).map((_, i) => lo + (i / (grid_resolution - 1)) * (hi - lo));
  const ice = X.map((xi) => {
    const row = new Float64Array(grid_resolution);
    for (let g = 0; g < grid_resolution; g++) {
      const xmod = new Float64Array(xi);
      xmod[feature_idx] = grid[g] ?? lo;
      const pred = estimator.predict([xmod]);
      row[g] = pred[0] ?? 0;
    }
    return row;
  });
  return { grid, ice };
}

/** Partial dependence plot data for two features. */
export function partialDependence2D(
  estimator: PredictingEstimator,
  X: Float64Array[],
  features: [number, number],
  grid_resolution = 20,
): { grid0: Float64Array; grid1: Float64Array; pdp: Float64Array[] } {
  const f0 = features[0], f1 = features[1];
  const vals0 = X.map((xi) => xi[f0] ?? 0).sort((a, b) => a - b);
  const vals1 = X.map((xi) => xi[f1] ?? 0).sort((a, b) => a - b);
  const lo0 = vals0[0] ?? 0, hi0 = vals0[vals0.length - 1] ?? 1;
  const lo1 = vals1[0] ?? 0, hi1 = vals1[vals1.length - 1] ?? 1;
  const grid0 = new Float64Array(grid_resolution).map((_, i) => lo0 + (i / (grid_resolution - 1)) * (hi0 - lo0));
  const grid1 = new Float64Array(grid_resolution).map((_, i) => lo1 + (i / (grid_resolution - 1)) * (hi1 - lo1));
  const pdp = Array.from({ length: grid_resolution }, (_, i) => {
    return new Float64Array(grid_resolution).map((_, j) => {
      const preds = X.map((xi) => {
        const xmod = new Float64Array(xi);
        xmod[f0] = grid0[i] ?? lo0;
        xmod[f1] = grid1[j] ?? lo1;
        return estimator.predict([xmod])[0] ?? 0;
      });
      return preds.reduce((s, v) => s + v, 0) / preds.length;
    });
  });
  return { grid0, grid1, pdp };
}

/** SHAP interaction values (simplified tree-based approximation). */
export function shapInteractionValues(
  estimator: PredictingEstimator,
  X: Float64Array[],
  background: Float64Array[] | null = null,
): Float64Array[][] {
  const nf = X[0]?.length ?? 0;
  const ref = background ?? [new Float64Array(nf)];
  const basePred = estimator.predict(ref).reduce ? Array.from(estimator.predict(ref)).reduce((s: number, v) => s + (v as number), 0) / ref.length : estimator.predict(ref)[0] ?? 0;
  return X.map((xi) => {
    const interactions = Array.from({ length: nf }, () => new Float64Array(nf));
    const baseRow = new Float64Array(xi);
    for (let i = 0; i < nf; i++) {
      for (let j = i; j < nf; j++) {
        // Simplified: marginal contribution of pair
        const xij = new Float64Array(xi);
        const xi_only = new Float64Array(xi);
        const xj_only = new Float64Array(xi);
        const xnone = new Float64Array(xi);
        for (let k = 0; k < nf; k++) {
          if (k !== i && k !== j) { xij[k] = ref[0]?.[k] ?? 0; xi_only[k] = ref[0]?.[k] ?? 0; xj_only[k] = ref[0]?.[k] ?? 0; xnone[k] = ref[0]?.[k] ?? 0; }
          else if (k === i) { xj_only[k] = ref[0]?.[k] ?? 0; xnone[k] = ref[0]?.[k] ?? 0; }
          else if (k === j) { xi_only[k] = ref[0]?.[k] ?? 0; xnone[k] = ref[0]?.[k] ?? 0; }
        }
        const v_ij = estimator.predict([xij])[0] ?? 0;
        const v_i = estimator.predict([xi_only])[0] ?? 0;
        const v_j = estimator.predict([xj_only])[0] ?? 0;
        const v_0 = estimator.predict([xnone])[0] ?? 0;
        const interaction = (v_ij - v_i - v_j + v_0) / 2;
        interactions[i]![j] = interaction;
        interactions[j]![i] = interaction;
      }
      interactions[i]![i] = (estimator.predict([baseRow])[0] ?? 0) - basePred - Array.from({ length: nf }, (_, j) => j !== i ? (interactions[i]?.[j] ?? 0) : 0).reduce((s, v) => s + v, 0);
    }
    return interactions;
  });
}

/** CounterfactualExplainer: find minimal feature changes to change prediction. */
export class CounterfactualExplainer extends BaseEstimator {
  estimator: PredictingEstimator;
  n_iter: number;
  step_size: number;

  constructor(estimator: PredictingEstimator, nIter = 100, stepSize = 0.01) {
    super();
    this.estimator = estimator;
    this.n_iter = nIter;
    this.step_size = stepSize;
  }

  explain(
    x: Float64Array,
    target_class: number,
  ): { counterfactual: Float64Array; distance: number; n_iter: number } {
    let cf = new Float64Array(x);
    let iter = 0;
    for (iter = 0; iter < this.n_iter; iter++) {
      const pred = this.estimator.predict([cf])[0] ?? 0;
      if (pred === target_class) break;
      // Gradient-free: perturb each feature
      for (let k = 0; k < cf.length; k++) {
        const cfPlus = new Float64Array(cf);
        cfPlus[k] = (cf[k] ?? 0) + this.step_size;
        const cfMinus = new Float64Array(cf);
        cfMinus[k] = (cf[k] ?? 0) - this.step_size;
        const pPlus = this.estimator.predict([cfPlus])[0] ?? 0;
        if (pPlus === target_class) { cf = cfPlus; break; }
        const pMinus = this.estimator.predict([cfMinus])[0] ?? 0;
        if (pMinus === target_class) { cf = cfMinus; break; }
      }
    }
    let dist = 0;
    for (let k = 0; k < x.length; k++) dist += ((cf[k] ?? 0) - (x[k] ?? 0)) ** 2;
    return { counterfactual: cf, distance: Math.sqrt(dist), n_iter: iter };
  }
}
