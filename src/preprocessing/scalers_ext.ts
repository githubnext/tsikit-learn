/**
 * Extended scaler implementations.
 * Mirrors scikit-learn's preprocessing.MaxAbsScaler and additional scalers.
 */

/**
 * Scale each feature by its maximum absolute value.
 * Transforms each feature to [-1, 1] without shifting (preserves zero/sparsity).
 */
export class MaxAbsScaler {
  private _maxAbs: Float64Array | null = null;
  private _nFeatures = 0;

  fit(X: Float64Array[]): this {
    if (X.length === 0) throw new RangeError("X must have at least one sample");
    const nFeatures = X[0]!.length;
    this._nFeatures = nFeatures;
    const maxAbs = new Float64Array(nFeatures);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        const v = Math.abs(row[j] ?? 0);
        if (v > (maxAbs[j] ?? 0)) maxAbs[j] = v;
      }
    }
    this._maxAbs = maxAbs;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this._maxAbs === null) throw new Error("MaxAbsScaler must be fitted first");
    return X.map((row) =>
      Float64Array.from(row, (v, j) => {
        const m = this._maxAbs![j] ?? 1;
        return m === 0 ? 0 : v / m;
      }),
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this._maxAbs === null) throw new Error("MaxAbsScaler must be fitted first");
    return X.map((row) =>
      Float64Array.from(row, (v, j) => v * (this._maxAbs![j] ?? 1)),
    );
  }
}

/**
 * Center and scale to unit variance, handling sparse data correctly.
 * Mirrors scikit-learn's preprocessing.scale function.
 */
export function scale(
  X: Float64Array[],
  withMean = true,
  withStd = true,
): Float64Array[] {
  if (X.length === 0) return [];
  const nFeatures = X[0]!.length;
  const n = X.length;
  const mean = new Float64Array(nFeatures);
  const std = new Float64Array(nFeatures);

  if (withMean) {
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) mean[j] = (mean[j] ?? 0) + (row[j] ?? 0) / n;
    }
  }
  if (withStd) {
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        const diff = (row[j] ?? 0) - (mean[j] ?? 0);
        std[j] = (std[j] ?? 0) + diff * diff / n;
      }
    }
    for (let j = 0; j < nFeatures; j++) std[j] = Math.sqrt(std[j] ?? 0);
  }

  return X.map((row) =>
    Float64Array.from(row, (v, j) => {
      let val = v;
      if (withMean) val -= mean[j] ?? 0;
      const s = std[j] ?? 1;
      if (withStd && s > 1e-10) val /= s;
      return val;
    }),
  );
}

/**
 * Clip features to given range then scale.
 * Mirrors scikit-learn's preprocessing.minmax_scale function.
 */
export function minmaxScale(
  X: Float64Array[],
  featureRange: [number, number] = [0, 1],
): Float64Array[] {
  if (X.length === 0) return [];
  const nFeatures = X[0]!.length;
  const xMin = new Float64Array(nFeatures).fill(Number.POSITIVE_INFINITY);
  const xMax = new Float64Array(nFeatures).fill(Number.NEGATIVE_INFINITY);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) {
      const v = row[j] ?? 0;
      if (v < (xMin[j] ?? Number.POSITIVE_INFINITY)) xMin[j] = v;
      if (v > (xMax[j] ?? Number.NEGATIVE_INFINITY)) xMax[j] = v;
    }
  }
  const [rMin, rMax] = featureRange;
  return X.map((row) =>
    Float64Array.from(row, (v, j) => {
      const mn = xMin[j] ?? 0;
      const mx = xMax[j] ?? 1;
      const range = mx - mn;
      const scaled = range < 1e-10 ? 0 : (v - mn) / range;
      return scaled * ((rMax ?? 1) - (rMin ?? 0)) + (rMin ?? 0);
    }),
  );
}
