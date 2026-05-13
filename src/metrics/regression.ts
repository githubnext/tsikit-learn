/**
 * Regression metrics.
 * Mirrors sklearn.metrics (regression subset).
 */

import { ValueError } from "../exceptions.js";

/** Mean squared error. */
export function mean_squared_error(
  yTrue: Float64Array,
  yPred: Float64Array,
  options: { sampleWeight?: Float64Array; squared?: boolean } = {},
): number {
  const { sampleWeight, squared = true } = options;
  if (yTrue.length !== yPred.length) {
    throw new ValueError("yTrue and yPred must have the same length");
  }
  let total = 0;
  let wSum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const diff = (yTrue[i] ?? 0) - (yPred[i] ?? 0);
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    total += w * diff * diff;
    wSum += w;
  }
  const mse = wSum > 0 ? total / wSum : 0;
  return squared ? mse : Math.sqrt(mse);
}

/** Mean absolute error. */
export function mean_absolute_error(
  yTrue: Float64Array,
  yPred: Float64Array,
  sampleWeight?: Float64Array,
): number {
  if (yTrue.length !== yPred.length) {
    throw new ValueError("yTrue and yPred must have the same length");
  }
  let total = 0;
  let wSum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    total += w * Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
    wSum += w;
  }
  return wSum > 0 ? total / wSum : 0;
}

/** R² score (coefficient of determination). */
export function r2_score(
  yTrue: Float64Array,
  yPred: Float64Array,
  sampleWeight?: Float64Array,
): number {
  if (yTrue.length !== yPred.length) {
    throw new ValueError("yTrue and yPred must have the same length");
  }
  let wSum = 0;
  let yMeanNum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    yMeanNum += w * (yTrue[i] ?? 0);
    wSum += w;
  }
  const yMean = wSum > 0 ? yMeanNum / wSum : 0;

  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const w = sampleWeight ? (sampleWeight[i] ?? 1) : 1;
    const diff = (yTrue[i] ?? 0) - yMean;
    ssTot += w * diff * diff;
    ssRes += w * ((yTrue[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

/** Mean absolute percentage error. */
export function mean_absolute_percentage_error(
  yTrue: Float64Array,
  yPred: Float64Array,
): number {
  if (yTrue.length !== yPred.length) {
    throw new ValueError("yTrue and yPred must have the same length");
  }
  let total = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yt = yTrue[i] ?? 0;
    if (yt === 0) continue;
    total += Math.abs((yt - (yPred[i] ?? 0)) / yt);
  }
  return total / yTrue.length;
}

/** Explained variance score. */
export function explained_variance_score(
  yTrue: Float64Array,
  yPred: Float64Array,
): number {
  const n = yTrue.length;
  let meanTrue = 0;
  let meanErr = 0;
  for (let i = 0; i < n; i++) {
    meanTrue += yTrue[i] ?? 0;
    meanErr += (yTrue[i] ?? 0) - (yPred[i] ?? 0);
  }
  meanTrue /= n;
  meanErr /= n;

  let varTrue = 0;
  let varErr = 0;
  for (let i = 0; i < n; i++) {
    varTrue += ((yTrue[i] ?? 0) - meanTrue) ** 2;
    varErr += ((yTrue[i] ?? 0) - (yPred[i] ?? 0) - meanErr) ** 2;
  }
  varTrue /= n;
  varErr /= n;

  return varTrue === 0 ? 0 : 1 - varErr / varTrue;
}
