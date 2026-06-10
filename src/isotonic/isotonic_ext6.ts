/**
 * Weighted isotonic regression and monotone spline fitting.
 */

export class WeightedIsotonicRegression {
  private x_!: Float64Array;
  private y_!: Float64Array;
  private fitted_ = false;

  constructor(private increasing = true) {}

  fit(X: Float64Array, y: Float64Array, sampleWeight?: Float64Array): this {
    const n = X.length;
    const w = sampleWeight ?? new Float64Array(n).fill(1);
    // Sort by X
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (X[a] ?? 0) - (X[b] ?? 0));
    const xs = new Float64Array(order.map(i => X[i] ?? 0));
    let ys = order.map(i => y[i] ?? 0);
    const ws = order.map(i => w[i] ?? 1);

    if (!this.increasing) ys = ys.map(v => -v);

    // Weighted Pool Adjacent Violators
    const pools: Array<{ sum: number; wsum: number; indices: number[] }> = ys.map((v, i) => ({
      sum: v * (ws[i] ?? 1),
      wsum: ws[i] ?? 1,
      indices: [i]
    }));

    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < pools.length - 1; i++) {
        const cur = pools[i]!, next = pools[i + 1]!;
        const curMean = cur.sum / cur.wsum;
        const nextMean = next.sum / next.wsum;
        if (curMean > nextMean) {
          pools[i] = {
            sum: cur.sum + next.sum,
            wsum: cur.wsum + next.wsum,
            indices: [...cur.indices, ...next.indices]
          };
          pools.splice(i + 1, 1);
          changed = true;
          break;
        }
      }
    }

    const result = new Float64Array(n);
    for (const pool of pools) {
      const mean = pool.sum / pool.wsum;
      for (const i of pool.indices) result[i] = this.increasing ? mean : -mean;
    }

    this.x_ = xs;
    this.y_ = result;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Float64Array(X.map(x => {
      if (x <= (this.x_[0] ?? 0)) return this.y_[0] ?? 0;
      if (x >= (this.x_[this.x_.length - 1] ?? 0)) return this.y_[this.y_.length - 1] ?? 0;
      let lo = 0, hi = this.x_.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if ((this.x_[mid] ?? 0) <= x) lo = mid; else hi = mid;
      }
      const t = ((this.x_[hi] ?? 0) - (this.x_[lo] ?? 0)) > 0
        ? (x - (this.x_[lo] ?? 0)) / ((this.x_[hi] ?? 0) - (this.x_[lo] ?? 0))
        : 0;
      return (this.y_[lo] ?? 0) * (1 - t) + (this.y_[hi] ?? 0) * t;
    }));
  }

  score(X: Float64Array, y: Float64Array): number {
    if (!this.fitted_) throw new Error('Not fitted');
    const yPred = this.predict(X);
    const n = y.length;
    const meanY = y.reduce((s, v) => s + v, 0) / n;
    const ssTot = y.reduce((s, v) => s + (v - meanY) ** 2, 0);
    const ssRes = y.reduce((s, v, i) => s + (v - (yPred[i] ?? 0)) ** 2, 0);
    return 1 - ssRes / (ssTot + 1e-10);
  }
}

export class MonotoneCubicSpline {
  private knots_!: Float64Array;
  private values_!: Float64Array;
  private derivatives_!: Float64Array;
  private fitted_ = false;

  fit(X: Float64Array, y: Float64Array): this {
    const n = X.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (X[a] ?? 0) - (X[b] ?? 0));
    this.knots_ = new Float64Array(order.map(i => X[i] ?? 0));
    this.values_ = new Float64Array(order.map(i => y[i] ?? 0));

    // Fritsch-Carlson monotone cubic spline
    const delta = new Float64Array(n - 1).map((_, k) => {
      const dx = (this.knots_[k + 1] ?? 0) - (this.knots_[k] ?? 0);
      return dx > 0 ? ((this.values_[k + 1] ?? 0) - (this.values_[k] ?? 0)) / dx : 0;
    });
    const m = new Float64Array(n);
    m[0] = delta[0] ?? 0;
    m[n - 1] = delta[n - 2] ?? 0;
    for (let k = 1; k < n - 1; k++) m[k] = ((delta[k - 1] ?? 0) + (delta[k] ?? 0)) / 2;
    // Ensure monotone
    for (let k = 0; k < n - 1; k++) {
      const dk = delta[k] ?? 0;
      if (Math.abs(dk) < 1e-10) { m[k] = 0; m[k + 1] = 0; continue; }
      const alpha = (m[k] ?? 0) / dk, beta = (m[k + 1] ?? 0) / dk;
      const r2 = alpha ** 2 + beta ** 2;
      if (r2 > 9) {
        const tau = 3 / Math.sqrt(r2);
        m[k] = tau * alpha * dk;
        m[k + 1] = tau * beta * dk;
      }
    }
    this.derivatives_ = m;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = this.knots_.length;
    return new Float64Array(X.map(x => {
      if (x <= (this.knots_[0] ?? 0)) return this.values_[0] ?? 0;
      if (x >= (this.knots_[n - 1] ?? 0)) return this.values_[n - 1] ?? 0;
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if ((this.knots_[mid] ?? 0) <= x) lo = mid; else hi = mid;
      }
      const h = (this.knots_[hi] ?? 0) - (this.knots_[lo] ?? 0);
      const t = h > 0 ? (x - (this.knots_[lo] ?? 0)) / h : 0;
      // Cubic Hermite spline
      const t2 = t * t, t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1;
      const h10 = t3 - 2 * t2 + t;
      const h01 = -2 * t3 + 3 * t2;
      const h11 = t3 - t2;
      return h00 * (this.values_[lo] ?? 0) + h10 * h * (this.derivatives_[lo] ?? 0)
        + h01 * (this.values_[hi] ?? 0) + h11 * h * (this.derivatives_[hi] ?? 0);
    }));
  }
}
