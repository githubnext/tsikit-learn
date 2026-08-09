/**
 * Standalone functional preprocessing utilities.
 */

/** Standardize features by removing mean and scaling to unit variance. */
export function scale(
  X: Float64Array[],
  withMean = true,
  withStd = true,
): Float64Array[] {
  const n = X.length;
  if (n === 0) return [];
  const p = (X[0] ?? new Float64Array(0)).length;
  const means = new Float64Array(p);
  const stds = new Float64Array(p);

  if (withMean || withStd) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (X[i] ?? new Float64Array(0))[j] ?? 0;
      means[j] = s / n;
    }
  }
  if (withStd) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++)
        s += ((X[i] ?? new Float64Array(0))[j] ?? 0 - (means[j] ?? 0)) ** 2;
      stds[j] = Math.sqrt(s / n) || 1;
    }
  }

  return X.map((row) => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let v = row[j] ?? 0;
      if (withMean) v -= means[j] ?? 0;
      if (withStd) v /= stds[j] ?? 1;
      out[j] = v;
    }
    return out;
  });
}

/** Scale features to a given range [featureRange[0], featureRange[1]]. */
export function minmaxScale(
  X: Float64Array[],
  featureRange: [number, number] = [0, 1],
): Float64Array[] {
  const n = X.length;
  if (n === 0) return [];
  const p = (X[0] ?? new Float64Array(0)).length;
  const mins = new Float64Array(p).fill(Number.POSITIVE_INFINITY);
  const maxs = new Float64Array(p).fill(Number.NEGATIVE_INFINITY);

  for (let i = 0; i < n; i++) {
    const row = X[i] ?? new Float64Array(0);
    for (let j = 0; j < p; j++) {
      const v = row[j] ?? 0;
      if (v < (mins[j] ?? Number.POSITIVE_INFINITY)) mins[j] = v;
      if (v > (maxs[j] ?? Number.NEGATIVE_INFINITY)) maxs[j] = v;
    }
  }

  const [lo, hi] = featureRange;
  return X.map((row) => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const range = (maxs[j] ?? 0) - (mins[j] ?? 0);
      out[j] =
        range === 0
          ? lo
          : lo + (((row[j] ?? 0) - (mins[j] ?? 0)) * (hi - lo)) / range;
    }
    return out;
  });
}

/** Normalize samples individually to unit norm. */
export function normalizeArr(
  X: Float64Array[],
  norm: "l1" | "l2" | "max" = "l2",
): Float64Array[] {
  return X.map((row) => {
    let normalizer = 0;
    if (norm === "l1") {
      for (let j = 0; j < row.length; j++) normalizer += Math.abs(row[j] ?? 0);
    } else if (norm === "l2") {
      for (let j = 0; j < row.length; j++) normalizer += (row[j] ?? 0) ** 2;
      normalizer = Math.sqrt(normalizer);
    } else {
      for (let j = 0; j < row.length; j++)
        normalizer = Math.max(normalizer, Math.abs(row[j] ?? 0));
    }
    if (normalizer === 0) return row.slice();
    const out = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) out[j] = (row[j] ?? 0) / normalizer;
    return out;
  });
}

/** Scale features using statistics that are robust to outliers. */
export function robustScale(
  X: Float64Array[],
  quantileRange: [number, number] = [25, 75],
): Float64Array[] {
  const n = X.length;
  if (n === 0) return [];
  const p = (X[0] ?? new Float64Array(0)).length;
  const medians = new Float64Array(p);
  const iqrs = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const col = Float64Array.from(
      { length: n },
      (_, i) => (X[i] ?? new Float64Array(0))[j] ?? 0,
    );
    col.sort();
    medians[j] = quantile(col, 0.5);
    const q1 = quantile(col, quantileRange[0] / 100);
    const q3 = quantile(col, quantileRange[1] / 100);
    iqrs[j] = q3 - q1 || 1;
  }

  return X.map((row) => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++)
      out[j] = ((row[j] ?? 0) - (medians[j] ?? 0)) / (iqrs[j] ?? 1);
    return out;
  });
}

function quantile(sorted: Float64Array, q: number): number {
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return (sorted[lo] ?? 0) * (1 - frac) + (sorted[hi] ?? 0) * frac;
}

/** Add a dummy feature (constant bias column) to X. */
export function addDummyFeature(
  X: Float64Array[],
  value = 1.0,
): Float64Array[] {
  return X.map((row) => {
    const out = new Float64Array(row.length + 1);
    out[0] = value;
    for (let j = 0; j < row.length; j++) out[j + 1] = row[j] ?? 0;
    return out;
  });
}
