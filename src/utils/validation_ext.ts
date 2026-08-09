/**
 * Extended validation utilities.
 * Mirrors sklearn.utils.validation: check_symmetric, check_non_negative,
 * check_consistent_length, check_scalar, indexable, check_same_n_features.
 * Note: column_or_1d is in utils/bunch.ts,
 *       check_classification_targets is in utils/multiclass.ts.
 */

/** Check that arrays/sequences have consistent (matching) lengths. */
export function checkConsistentLength(...arrays: Array<{ length: number } | null | undefined>): void {
  const lengths = arrays.filter(a => a != null).map(a => a!.length);
  const unique = new Set(lengths);
  if (unique.size > 1) {
    throw new Error(
      `Inconsistent numbers of samples: ${Array.from(unique).join(", ")}`,
    );
  }
}

/**
 * Raise if any value in X is negative.
 * Mirrors sklearn.utils.validation.check_non_negative.
 */
export function checkNonNegative(X: Float64Array[], whom: string): void {
  for (const row of X) {
    for (const v of row) {
      if (v < 0)
        throw new Error(`Negative values in data passed to ${whom}`);
    }
  }
}

/**
 * Check that a matrix is symmetric.
 * Mirrors sklearn.utils.check_symmetric.
 */
export function checkSymmetric(
  A: Float64Array[],
  tol: number = 1e-10,
  raiseWarning: boolean = false,
  raiseException: boolean = false,
): boolean {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs((A[i]![j] ?? 0) - (A[j]![i] ?? 0)) > tol) {
        if (raiseException)
          throw new Error("Array must be symmetric");
        if (raiseWarning)
          console.warn("Array is not symmetric and will be converted");
        return false;
      }
    }
  }
  return true;
}

/**
 * Check a scalar value is within optional bounds.
 * Mirrors sklearn.utils.validation.check_scalar.
 */
export function checkScalar(
  value: number,
  name: string,
  options: {
    minVal?: number;
    maxVal?: number;
    includeMin?: boolean;
    includeMax?: boolean;
  } = {},
): number {
  const { minVal, maxVal, includeMin = true, includeMax = true } = options;
  if (typeof value !== "number" || Number.isNaN(value))
    throw new TypeError(`${name} must be a number, got ${typeof value}`);
  if (minVal !== undefined) {
    if (includeMin ? value < minVal : value <= minVal)
      throw new RangeError(
        `${name} = ${value} must be ${includeMin ? ">=" : ">"} ${minVal}`,
      );
  }
  if (maxVal !== undefined) {
    if (includeMax ? value > maxVal : value >= maxVal)
      throw new RangeError(
        `${name} = ${value} must be ${includeMax ? "<=" : "<"} ${maxVal}`,
      );
  }
  return value;
}

/**
 * Returns the objects for duck-type indexable objects.
 * Mirrors sklearn.utils.indexable.
 */
export function indexable<T>(...arrays: T[]): T[] {
  return arrays;
}

/**
 * Check that arrays have the same number of features.
 */
export function checkSameNFeatures(
  X1: Float64Array[],
  X2: Float64Array[],
  name: string = "X",
): void {
  const p1 = (X1[0] ?? new Float64Array(0)).length;
  const p2 = (X2[0] ?? new Float64Array(0)).length;
  if (p1 !== p2)
    throw new Error(
      `${name}: number of features mismatch: ${p1} vs ${p2}`,
    );
}
