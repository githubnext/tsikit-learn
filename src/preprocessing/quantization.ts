/**
 * Preprocessing: feature quantization and discretization utilities.
 * Mirrors sklearn.preprocessing (quantile-based transforms, discretizers).
 */

/**
 * Winsorize data: clip values beyond a quantile range.
 * Values below lower_quantile or above upper_quantile are replaced by
 * the corresponding quantile value.
 *
 * @param X - Input data (n_samples x n_features)
 * @param lowerQuantile - Lower clipping quantile (0-1, default 0.05)
 * @param upperQuantile - Upper clipping quantile (0-1, default 0.95)
 */
export function winsorize(
  X: Float64Array[],
  lowerQuantile = 0.05,
  upperQuantile = 0.95,
): Float64Array[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const lowers = new Float64Array(p);
  const uppers = new Float64Array(p);

  const getQuantile = (sorted: number[], q: number): number => {
    const idx = q * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const frac = idx - lo;
    return (sorted[lo] ?? 0) * (1 - frac) + (sorted[hi] ?? 0) * frac;
  };

  for (let j = 0; j < p; j++) {
    const col: number[] = [];
    for (let i = 0; i < n; i++) col.push(X[i]![j] ?? 0);
    col.sort((a, b) => a - b);
    lowers[j] = getQuantile(col, lowerQuantile);
    uppers[j] = getQuantile(col, upperQuantile);
  }

  return X.map(row => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const v = row[j] ?? 0;
      out[j] = Math.max(lowers[j] ?? 0, Math.min(uppers[j] ?? 0, v));
    }
    return out;
  });
}

/**
 * Subtract the per-sample mean (center each sample individually).
 */
export function meanCenter(X: Float64Array[]): Float64Array[] {
  return X.map(row => {
    const mean = Array.from(row).reduce((s, v) => s + v, 0) / row.length;
    const out = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) out[j] = (row[j] ?? 0) - mean;
    return out;
  });
}

/**
 * Compute pairwise squared Euclidean distances.
 * Equivalent to sklearn.metrics.pairwise.euclidean_distances(X, squared=True).
 */
export function squaredEuclideanDistances(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ymat = Y ?? X;
  const n = X.length;
  const m = Ymat.length;
  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    const xi = X[i]!;
    for (let j = 0; j < m; j++) {
      const yj = Ymat[j]!;
      let d2 = 0;
      for (let k = 0; k < xi.length; k++) {
        d2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      }
      row[j] = d2;
    }
    return row;
  });
}

/**
 * Power transform a single array: apply Box-Cox or Yeo-Johnson.
 * Returns the transformed array and the fitted lambda.
 */
export function boxCox1d(
  y: Float64Array,
  lmbda: number | null = null,
): { transformed: Float64Array; lambda: number } {
  // Estimate lambda via MLE if null
  const n = y.length;

  if (lmbda === null) {
    // Grid search over lambda values
    let bestLambda = 0;
    let bestLogLik = -Number.POSITIVE_INFINITY;
    for (let l = -2; l <= 2; l += 0.1) {
      const t = _boxCoxTransform(y, l);
      if (t === null) continue;
      const mean = Array.from(t).reduce((s, v) => s + v, 0) / n;
      let variance = 0;
      for (const v of t) variance += (v - mean) ** 2;
      variance /= n;
      if (variance < 1e-10) continue;
      const logLik = -0.5 * n * Math.log(variance)
        + (l - 1) * Array.from(y).reduce((s, v) => s + Math.log(v), 0);
      if (logLik > bestLogLik) { bestLogLik = logLik; bestLambda = l; }
    }
    lmbda = bestLambda;
  }

  const transformed = _boxCoxTransform(y, lmbda) ?? y;
  return { transformed, lambda: lmbda };
}

function _boxCoxTransform(y: Float64Array, lmbda: number): Float64Array | null {
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const v = y[i] ?? 0;
    if (v <= 0) return null; // Box-Cox requires positive values
    out[i] = lmbda === 0 ? Math.log(v) : (v ** lmbda - 1) / lmbda;
  }
  return out;
}

/**
 * Yeo-Johnson transform (works with both positive and negative values).
 */
export function yeoJohnson1d(
  y: Float64Array,
  lmbda = 0.0,
): Float64Array {
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const v = y[i] ?? 0;
    if (v >= 0) {
      out[i] = lmbda === 0
        ? Math.log1p(v)
        : ((v + 1) ** lmbda - 1) / lmbda;
    } else {
      out[i] = lmbda === 2
        ? -Math.log1p(-v)
        : -((-v + 1) ** (2 - lmbda) - 1) / (2 - lmbda);
    }
  }
  return out;
}

/**
 * Estimate the number of bins for a histogram (Sturges, FD, or Rice rule).
 */
export function estimateNBins(
  n: number,
  method: "sturges" | "fd" | "rice" | "sqrt" = "sturges",
): number {
  if (n <= 1) return 1;
  switch (method) {
    case "sturges": return Math.ceil(Math.log2(n)) + 1;
    case "rice": return Math.ceil(2 * n ** (1 / 3));
    case "sqrt": return Math.ceil(Math.sqrt(n));
    case "fd": return Math.ceil(2 * n ** (1 / 3)); // simplified
    default: return Math.ceil(Math.log2(n)) + 1;
  }
}

/**
 * Compute bin edges for equal-width or equal-frequency binning.
 *
 * @param values - 1D array of values to bin
 * @param nBins - Number of bins
 * @param strategy - 'uniform' (equal-width) or 'quantile' (equal-frequency)
 */
export function computeBinEdges(
  values: Float64Array,
  nBins: number,
  strategy: "uniform" | "quantile" = "uniform",
): Float64Array {
  const sorted = values.slice().sort();
  const edges = new Float64Array(nBins + 1);

  if (strategy === "uniform") {
    const min = sorted[0] ?? 0;
    const max = sorted[sorted.length - 1] ?? 1;
    const width = (max - min) / nBins;
    for (let i = 0; i <= nBins; i++) edges[i] = min + i * width;
  } else {
    for (let i = 0; i <= nBins; i++) {
      const q = i / nBins;
      const idx = q * (sorted.length - 1);
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      edges[i] = (sorted[lo] ?? 0) * (1 - (idx - lo)) + (sorted[hi] ?? 0) * (idx - lo);
    }
  }

  return edges;
}

/**
 * Assign each value to a bin given bin edges.
 * Returns bin indices (0-indexed, clipped to [0, nBins-1]).
 */
export function digitize(
  values: Float64Array,
  edges: Float64Array,
): Int32Array {
  const nBins = edges.length - 1;
  const result = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? 0;
    let bin = nBins - 1;
    for (let b = 0; b < nBins; b++) {
      if (v < (edges[b + 1] ?? Number.POSITIVE_INFINITY)) { bin = b; break; }
    }
    result[i] = Math.min(nBins - 1, Math.max(0, bin));
  }
  return result;
}
