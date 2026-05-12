/**
 * Class weight utilities.
 * Mirrors sklearn.utils.class_weight.
 */

import { ValueError } from "../exceptions.js";

/**
 * Compute class weights for imbalanced datasets.
 * For 'balanced': n_samples / (n_classes * bincount(y))
 */
export function computeClassWeight(
  classWeight: "balanced" | Record<number, number>,
  classes: Int32Array,
  y: Float64Array | Int32Array,
): Float64Array {
  const weights = new Float64Array(classes.length);

  if (classWeight === "balanced") {
    const nSamples = y.length;
    const nClasses = classes.length;
    const counts = new Map<number, number>();
    for (const c of classes) counts.set(c, 0);
    for (const v of y) {
      const cur = counts.get(v);
      if (cur !== undefined) counts.set(v, cur + 1);
    }
    for (let i = 0; i < classes.length; i++) {
      const c = classes[i] ?? 0;
      const count = counts.get(c) ?? 0;
      if (count === 0) {
        throw new ValueError(`Class ${c} is not present in y`);
      }
      weights[i] = nSamples / (nClasses * count);
    }
  } else {
    for (let i = 0; i < classes.length; i++) {
      const c = classes[i] ?? 0;
      const w = classWeight[c];
      if (w === undefined) {
        throw new ValueError(`Class ${c} is not in classWeight`);
      }
      weights[i] = w;
    }
  }
  return weights;
}

/**
 * Compute per-sample weights from class weights.
 */
export function computeSampleWeight(
  classWeight: "balanced" | Record<number, number>,
  y: Float64Array | Int32Array,
): Float64Array {
  const uniqueClasses = new Set<number>();
  for (const v of y) uniqueClasses.add(v);
  const classes = new Int32Array([...uniqueClasses].sort((a, b) => a - b));
  const cw = computeClassWeight(classWeight, classes, y);
  const classToWeight = new Map<number, number>();
  for (let i = 0; i < classes.length; i++) {
    classToWeight.set(classes[i] ?? 0, cw[i] ?? 1.0);
  }
  const sampleWeights = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    sampleWeights[i] = classToWeight.get(y[i] ?? 0) ?? 1.0;
  }
  return sampleWeights;
}
