/**
 * Input validation utilities.
 * Mirrors sklearn.utils.validation.
 */

import { ValueError } from "../exceptions.js";

/** Validate that X is a non-empty 2D array of Float64Arrays. */
export function checkArray(
  X: Float64Array[],
  options: {
    minSamples?: number;
    minFeatures?: number;
    allowNd?: boolean;
  } = {},
): Float64Array[] {
  const { minSamples = 1, minFeatures = 1 } = options;
  if (!Array.isArray(X)) {
    throw new ValueError("X must be an array of Float64Arrays");
  }
  if (X.length < minSamples) {
    throw new ValueError(
      `X must have at least ${minSamples} samples, got ${X.length}`,
    );
  }
  const nFeatures = (X[0] ?? new Float64Array(0)).length;
  if (nFeatures < minFeatures) {
    throw new ValueError(
      `X must have at least ${minFeatures} features, got ${nFeatures}`,
    );
  }
  for (let i = 0; i < X.length; i++) {
    const row = X[i];
    if (!(row instanceof Float64Array)) {
      throw new ValueError(`X[${i}] must be a Float64Array`);
    }
    if (row.length !== nFeatures) {
      throw new ValueError(
        `X rows must all have the same length. Row 0 has ${nFeatures}, row ${i} has ${row.length}`,
      );
    }
  }
  return X;
}

/** Validate that X and y have compatible shapes. */
export function checkXy(
  X: Float64Array[],
  y: Float64Array | Int32Array,
): [Float64Array[], Float64Array | Int32Array] {
  checkArray(X);
  if (X.length !== y.length) {
    throw new ValueError(
      `X and y have inconsistent first dimensions: X has ${X.length} samples, y has ${y.length}`,
    );
  }
  return [X, y];
}

/** Return the number of features in X. */
export function getNumFeatures(X: Float64Array[]): number {
  if (X.length === 0) return 0;
  return (X[0] ?? new Float64Array(0)).length;
}

/** Validate that test features match training features. */
export function checkFeaturesConsistency(
  XTrain: Float64Array[],
  XTest: Float64Array[],
): void {
  const trainFeats = getNumFeatures(XTrain);
  const testFeats = getNumFeatures(XTest);
  if (trainFeats !== testFeats) {
    throw new ValueError(
      `X has ${testFeats} features, but the estimator was trained with ${trainFeats} features`,
    );
  }
}

/** Convert a number array to Float64Array. */
export function asFloat64Array(arr: number[] | Float64Array): Float64Array {
  if (arr instanceof Float64Array) return arr;
  return new Float64Array(arr);
}

/** Convert a number array to Int32Array. */
export function asInt32Array(arr: number[] | Int32Array): Int32Array {
  if (arr instanceof Int32Array) return arr;
  return new Int32Array(arr);
}

/** Validate sample weights, returning a uniform weight array if null. */
export function checkSampleWeight(
  sampleWeight: Float64Array | null | undefined,
  nSamples: number,
): Float64Array {
  if (sampleWeight == null) {
    const w = new Float64Array(nSamples);
    w.fill(1.0);
    return w;
  }
  if (sampleWeight.length !== nSamples) {
    throw new ValueError(
      `sampleWeight.length (${sampleWeight.length}) != n_samples (${nSamples})`,
    );
  }
  return sampleWeight;
}
