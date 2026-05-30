/**
 * Utils extensions: weight_vector, spline utilities, window functions.
 * Mirrors sklearn.utils advanced helpers.
 */

/**
 * Compute sample weights for imbalanced datasets.
 * Implements sklearn.utils.class_weight.compute_sample_weight.
 */
export function computeSampleWeightExt(
  class_weight: Record<number, number> | "balanced",
  y: Int32Array,
): Float64Array {
  const n = y.length;
  const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
  const k = classes.length;
  let weights: Map<number, number>;
  if (class_weight === "balanced") {
    weights = new Map();
    const counts = new Map<number, number>();
    for (const c of y) counts.set(c, (counts.get(c) ?? 0) + 1);
    for (const c of classes) weights.set(c, n / (k * (counts.get(c) ?? 1)));
  } else {
    weights = new Map(Object.entries(class_weight).map(([k, v]) => [Number(k), v]));
  }
  return new Float64Array(n).map((_, i) => weights.get(y[i] ?? 0) ?? 1.0);
}

/** B-spline basis functions. */
export function bsplineBasis(
  x: Float64Array,
  knots: Float64Array,
  degree: number,
): Float64Array[] {
  const n = x.length;
  const nKnots = knots.length;
  const nBasis = nKnots - degree - 1;
  const basis: Float64Array[] = Array.from({ length: nBasis }, () => new Float64Array(n));
  // Cox-de Boor recursion
  const B = Array.from({ length: nKnots - 1 }, () => new Float64Array(n));
  // Order 0
  for (let j = 0; j < nKnots - 1; j++) {
    const lo = knots[j] ?? 0, hi = knots[j + 1] ?? 1;
    for (let i = 0; i < n; i++) B[j]![i] = (x[i] ?? 0) >= lo && (x[i] ?? 0) < hi ? 1 : 0;
  }
  for (let d = 1; d <= degree; d++) {
    const Bnew = Array.from({ length: nKnots - d - 1 }, () => new Float64Array(n));
    for (let j = 0; j < nKnots - d - 1; j++) {
      const lo1 = knots[j] ?? 0, hi1 = knots[j + d] ?? 1;
      const lo2 = knots[j + 1] ?? 0, hi2 = knots[j + d + 1] ?? 1;
      for (let i = 0; i < n; i++) {
        let v = 0;
        if (hi1 - lo1 > 1e-10) v += ((x[i] ?? 0) - lo1) / (hi1 - lo1) * (B[j]?.[i] ?? 0);
        if (hi2 - lo2 > 1e-10) v += (hi2 - (x[i] ?? 0)) / (hi2 - lo2) * (B[j + 1]?.[i] ?? 0);
        Bnew[j]![i] = v;
      }
    }
    for (let j = 0; j < Bnew.length; j++) for (let i = 0; i < n; i++) B[j]![i] = Bnew[j]?.[i] ?? 0;
  }
  for (let j = 0; j < nBasis; j++) basis[j] = B[j] ?? new Float64Array(n);
  return basis;
}

/** Check if array is monotonically increasing/decreasing. */
export function checkMonotone(
  x: Float64Array,
  increasing = true,
): boolean {
  for (let i = 1; i < x.length; i++) {
    const prev = x[i - 1] ?? 0, curr = x[i] ?? 0;
    if (increasing && curr < prev) return false;
    if (!increasing && curr > prev) return false;
  }
  return true;
}

/** Generate random samples from a multivariate normal distribution. */
export function multivariateNormal(
  mean: Float64Array,
  cov: Float64Array[],
  n: number,
  seed = 42,
): Float64Array[] {
  const p = mean.length;
  // Cholesky decomposition of cov
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = cov[i]?.[j] ?? 0;
      for (let k = 0; k < j; k++) s -= (L[i]?.[k] ?? 0) * (L[j]?.[k] ?? 0);
      L[i]![j] = i === j ? Math.sqrt(Math.max(s, 0)) : (L[j]?.[j] ?? 1) > 1e-10 ? s / (L[j]![j] ?? 1) : 0;
    }
  }
  // Box-Muller transform for standard normals
  let state = seed;
  const rand = (): number => {
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    const u1 = ((state >>> 0) + 0.5) / 0x100000000;
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    const u2 = ((state >>> 0) + 0.5) / 0x100000000;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  return Array.from({ length: n }, () => {
    const z = new Float64Array(p).map(() => rand());
    const x = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      x[i] = mean[i] ?? 0;
      for (let j = 0; j <= i; j++) x[i] = (x[i] ?? 0) + (L[i]?.[j] ?? 0) * (z[j] ?? 0);
    }
    return x;
  });
}

/** Shuffle arrays in unison. */
export function shuffleUnison(
  X: Float64Array[],
  y: Float64Array | Int32Array,
  seed = 42,
): { X: Float64Array[]; y: Float64Array | Int32Array } {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  let state = seed;
  for (let i = n - 1; i > 0; i--) {
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    const j = ((state >>> 0) % (i + 1));
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  const newX = indices.map((i) => X[i]!);
  const newY = y instanceof Int32Array
    ? new Int32Array(indices.map((i) => y[i] ?? 0))
    : new Float64Array(indices.map((i) => (y as Float64Array)[i] ?? 0));
  return { X: newX, y: newY };
}

/** Running statistics (mean and variance via Welford's algorithm). */
export class RunningStats {
  n = 0;
  mean_ = 0;
  M2_ = 0;

  update(x: number): void {
    this.n++;
    const delta = x - this.mean_;
    this.mean_ += delta / this.n;
    const delta2 = x - this.mean_;
    this.M2_ += delta * delta2;
  }

  get variance(): number {
    return this.n < 2 ? 0 : this.M2_ / (this.n - 1);
  }

  get std(): number {
    return Math.sqrt(this.variance);
  }
}
