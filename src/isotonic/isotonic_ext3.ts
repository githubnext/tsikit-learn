/**
 * Isotonic regression extensions: monotone splines, PAVA.
 * Mirrors sklearn.isotonic advanced methods.
 */

/**
 * Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
 */
export function pava(
  y: Float64Array,
  weights?: Float64Array,
): Float64Array {
  const n = y.length;
  const result = new Float64Array(y);
  const w = weights ?? new Float64Array(n).fill(1);
  const means: number[] = [];
  const wSums: number[] = [];
  const sizes: number[] = [];
  for (let i = 0; i < n; i++) {
    means.push(y[i] ?? 0);
    wSums.push(w[i] ?? 1);
    sizes.push(1);
    // Merge while decreasing
    while (means.length > 1) {
      const m = means.length;
      if ((means[m - 2] ?? 0) <= (means[m - 1] ?? 0)) break;
      const w1 = wSums[m - 2] ?? 1, w2 = wSums[m - 1] ?? 1;
      const newMean = ((means[m - 2] ?? 0) * w1 + (means[m - 1] ?? 0) * w2) / (w1 + w2);
      means.splice(m - 2, 2, newMean);
      wSums.splice(m - 2, 2, w1 + w2);
      sizes.splice(m - 2, 2, (sizes[m - 2] ?? 0) + (sizes[m - 1] ?? 0));
    }
  }
  let pos = 0;
  for (let g = 0; g < means.length; g++) {
    const s = sizes[g] ?? 0;
    for (let i = 0; i < s; i++) result[pos++] = means[g] ?? 0;
  }
  return result;
}

/** Monotone cubic spline (Fritsch-Carlson). */
export function monotoneCubicInterpolant(
  x: Float64Array,
  y: Float64Array,
): (t: Float64Array) => Float64Array {
  const n = x.length;
  // Compute slopes
  const delta = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) delta[i] = ((y[i + 1] ?? 0) - (y[i] ?? 0)) / ((x[i + 1] ?? 1) - (x[i] ?? 0));
  // Tangents
  const m = new Float64Array(n);
  m[0] = delta[0] ?? 0;
  m[n - 1] = delta[n - 2] ?? 0;
  for (let i = 1; i < n - 1; i++) m[i] = ((delta[i - 1] ?? 0) + (delta[i] ?? 0)) / 2;
  // Monotonicity conditions
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(delta[i] ?? 0) < 1e-10) { m[i] = 0; m[i + 1] = 0; continue; }
    const alpha = (m[i] ?? 0) / (delta[i] ?? 1);
    const beta = (m[i + 1] ?? 0) / (delta[i] ?? 1);
    const h = Math.sqrt(alpha ** 2 + beta ** 2);
    if (h > 3) { m[i] = (3 / h) * alpha * (delta[i] ?? 0); m[i + 1] = (3 / h) * beta * (delta[i] ?? 0); }
  }
  return (t: Float64Array): Float64Array => {
    return new Float64Array(t.length).map((_, k) => {
      const tk = t[k] ?? 0;
      if (tk <= (x[0] ?? 0)) return y[0] ?? 0;
      if (tk >= (x[n - 1] ?? 0)) return y[n - 1] ?? 0;
      // Binary search
      let lo = 0, hi = n - 2;
      while (lo < hi) { const mid = (lo + hi) >> 1; if ((x[mid + 1] ?? 0) < tk) lo = mid + 1; else hi = mid; }
      const h = (x[lo + 1] ?? 1) - (x[lo] ?? 0);
      if (Math.abs(h) < 1e-10) return y[lo] ?? 0;
      const t2 = (tk - (x[lo] ?? 0)) / h;
      const t2sq = t2 * t2, t2cu = t2sq * t2;
      const h00 = 2 * t2cu - 3 * t2sq + 1;
      const h10 = t2cu - 2 * t2sq + t2;
      const h01 = -2 * t2cu + 3 * t2sq;
      const h11 = t2cu - t2sq;
      return h00 * (y[lo] ?? 0) + h10 * h * (m[lo] ?? 0) + h01 * (y[lo + 1] ?? 0) + h11 * h * (m[lo + 1] ?? 0);
    });
  };
}

/** SplineIsotonicRegression: isotonic regression with spline smoothing. */
export class SplineIsotonicRegression {
  increasing: boolean;
  f_: ((t: Float64Array) => Float64Array) | null = null;
  x_thresholds_: Float64Array = new Float64Array(0);
  y_thresholds_: Float64Array = new Float64Array(0);

  constructor(increasing = true) {
    this.increasing = increasing;
  }

  fit(X: Float64Array, y: Float64Array): this {
    const n = X.length;
    // Sort by X
    const sorted = Array.from({ length: n }, (_, i) => ({ x: X[i] ?? 0, y: y[i] ?? 0 })).sort((a, b) => a.x - b.x);
    const xs = new Float64Array(sorted.map((s) => s.x));
    let ys = new Float64Array(sorted.map((s) => s.y));
    // Apply isotonic regression
    if (!this.increasing) for (let i = 0; i < n; i++) ys[i] = -(ys[i] ?? 0);
    ys = pava(ys) as Float64Array<ArrayBuffer>;
    if (!this.increasing) for (let i = 0; i < n; i++) ys[i] = -(ys[i] ?? 0);
    this.x_thresholds_ = xs;
    this.y_thresholds_ = ys;
    this.f_ = monotoneCubicInterpolant(xs, ys);
    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (!this.f_) throw new Error("Not fitted");
    return this.f_(X);
  }

  score(X: Float64Array, y: Float64Array): number {
    const yp = this.predict(X);
    let sr = 0, st = 0, ym = 0;
    for (const v of y) ym += v;
    ym /= y.length;
    for (let i = 0; i < y.length; i++) {
      sr += ((y[i] ?? 0) - (yp[i] ?? 0)) ** 2;
      st += ((y[i] ?? 0) - ym) ** 2;
    }
    return st === 0 ? 1 : 1 - sr / st;
  }
}
