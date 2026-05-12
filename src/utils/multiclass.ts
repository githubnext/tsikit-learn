/**
 * Multiclass utilities.
 * Mirrors sklearn.utils.multiclass.
 */

import { ValueError } from "../exceptions.js";

export type MulticlassType =
  | "binary"
  | "multiclass"
  | "multiclass-multioutput"
  | "multilabel-indicator"
  | "continuous"
  | "continuous-multioutput"
  | "unknown";

/** Determine the type of target variable. */
export function typeOfTarget(y: Float64Array | Int32Array): MulticlassType {
  const unique = new Set<number>();
  for (const v of y) unique.add(v);
  const nUnique = unique.size;

  // Check if all values are integers
  const allInt = Array.from(unique).every((v) => Number.isInteger(v));
  if (!allInt) return "continuous";

  if (nUnique <= 2) return "binary";
  return "multiclass";
}

/** Return sorted unique class labels. */
export function uniqueLabels(...ys: (Float64Array | Int32Array)[]): Int32Array {
  const all = new Set<number>();
  for (const y of ys) {
    for (const v of y) all.add(v);
  }
  return new Int32Array([...all].sort((a, b) => a - b));
}

/** Check if classification is binary. */
export function isBinaryClassification(y: Float64Array | Int32Array): boolean {
  const unique = new Set<number>();
  for (const v of y) unique.add(v);
  return unique.size === 2;
}

/** Check if classification is multilabel. */
export function isMultilabel(_y: Float64Array[]): boolean {
  // For dense arrays this is always false in our simplified implementation
  return false;
}

/** Return the number of classes for a label array. */
export function classCount(y: Float64Array | Int32Array): number {
  const unique = new Set<number>();
  for (const v of y) unique.add(v);
  return unique.size;
}

/** Validate that y only contains values in the expected classes. */
export function checkClassificationTargets(y: Float64Array | Int32Array): void {
  const t = typeOfTarget(y);
  if (t === "continuous") {
    throw new ValueError(
      `Unknown label type: ${t}. Maybe you are trying to fit a classifier, which expects discrete classes.`,
    );
  }
}
