/**
 * Additional functional preprocessing wrappers.
 * Mirrors sklearn.preprocessing functional API extensions:
 * maxabs_scale (functional wrapper), binarize (functional wrapper).
 * Note: scale, minmax_scale, normalize, robust_scale are in data.ts.
 */

/**
 * Scale each feature by its maximum absolute value.
 * Mirrors sklearn.preprocessing.maxabs_scale.
 */
export function maxabsScale(X: Float64Array[]): Float64Array[] {
  const p = (X[0] ?? new Float64Array(0)).length;
  const maxAbs = new Float64Array(p);
  for (const xi of X)
    for (let j = 0; j < p; j++) {
      const v = Math.abs(xi[j] ?? 0);
      if (v > (maxAbs[j] ?? 0)) maxAbs[j] = v;
    }
  return X.map(xi => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++)
      out[j] = (xi[j] ?? 0) / ((maxAbs[j] ?? 0) || 1);
    return out;
  });
}

/**
 * Binarize a data matrix by threshold.
 * Mirrors sklearn.preprocessing.binarize (functional form).
 */
export function binarize(X: Float64Array[], threshold: number = 0): Float64Array[] {
  return X.map(xi => new Float64Array(xi.map(v => (v > threshold ? 1 : 0))));
}

/**
 * Quantize features to a fixed number of decimal places.
 */
export function quantizeFeatures(X: Float64Array[], decimals: number = 2): Float64Array[] {
  const factor = Math.pow(10, decimals);
  return X.map(xi => new Float64Array(xi.map(v => Math.round(v * factor) / factor)));
}

/**
 * Center the data matrix by subtracting the column means.
 */
export function centerData(X: Float64Array[]): { Xc: Float64Array[]; mean: Float64Array } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const mean = new Float64Array(p);
  for (const xi of X) for (let j = 0; j < p; j++) mean[j]! += (xi[j] ?? 0) / n;
  const Xc = X.map(xi => new Float64Array(p).map((_, j) => (xi[j] ?? 0) - (mean[j] ?? 0)));
  return { Xc, mean };
}

/**
 * Clip feature values to a range [low, high].
 * Mirrors numpy.clip applied per-sample.
 */
export function clipData(
  X: Float64Array[],
  low: number = 0,
  high: number = 1,
): Float64Array[] {
  return X.map(xi =>
    new Float64Array(xi.map(v => Math.max(low, Math.min(high, v)))),
  );
}

