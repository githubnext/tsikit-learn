/**
 * Low-level array utility functions — analogous to sklearn.utils._arrayfuncs.
 */

/** Returns the index of the minimum positive value in arr, or -1 if none. */
export function minPosIndex(arr: Float64Array): number {
  let idx = -1;
  let minVal = Number.POSITIVE_INFINITY;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    if (v > 0 && v < minVal) {
      minVal = v;
      idx = i;
    }
  }
  return idx;
}

/** Returns the minimum positive value in arr, or Infinity if none. */
export function minPos(arr: Float64Array): number {
  let minVal = Number.POSITIVE_INFINITY;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    if (v > 0 && v < minVal) minVal = v;
  }
  return minVal;
}

/** In-place L1 normalization of each row of a 2-D matrix (nRows x nCols). */
export function inplaceRowNormalizeL1(
  X: Float64Array,
  nRows: number,
  nCols: number,
): void {
  for (let i = 0; i < nRows; i++) {
    let sum = 0;
    for (let j = 0; j < nCols; j++) sum += Math.abs(X[i * nCols + j]!);
    if (sum === 0) continue;
    for (let j = 0; j < nCols; j++) X[i * nCols + j]! /= sum;
  }
}

/** In-place L2 normalization of each row of a 2-D matrix (nRows x nCols). */
export function inplaceRowNormalizeL2(
  X: Float64Array,
  nRows: number,
  nCols: number,
): void {
  for (let i = 0; i < nRows; i++) {
    let sum = 0;
    for (let j = 0; j < nCols; j++) {
      const v = X[i * nCols + j]!;
      sum += v * v;
    }
    if (sum === 0) continue;
    const norm = Math.sqrt(sum);
    for (let j = 0; j < nCols; j++) X[i * nCols + j]! /= norm;
  }
}

/** In-place column scaling: multiplies column j of X (nRows x nCols) by scale[j]. */
export function inplaceColumnScale(
  X: Float64Array,
  nRows: number,
  nCols: number,
  scale: Float64Array,
): void {
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      X[i * nCols + j]! *= scale[j];
    }
  }
}

/** Computes cumulative sum in-place (modifies arr). */
export function cumsum(arr: Float64Array): Float64Array {
  for (let i = 1; i < arr.length; i++) arr[i]! += arr[i - 1];
  return arr;
}

/**
 * Fast row-wise dot product: returns a Float64Array of length nRows where
 * result[i] = sum_j X[i,j] * w[j].
 */
export function rowDot(
  X: Float64Array,
  nRows: number,
  nCols: number,
  w: Float64Array,
): Float64Array {
  const out = new Float64Array(nRows);
  for (let i = 0; i < nRows; i++) {
    let s = 0;
    for (let j = 0; j < nCols; j++) s += X[i * nCols + j]! * w[j]!;
    out[i] = s;
  }
  return out;
}

/** Clips values of arr in-place to [lo, hi]. */
export function clipInplace(arr: Float64Array, lo: number, hi: number): void {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    arr[i] = v < lo ? lo : v > hi ? hi : v;
  }
}
