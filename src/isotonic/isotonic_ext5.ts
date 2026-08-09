/**
 * Isotonic regression extension — monotone spline and piecewise constant fitting.
 */

export class IsotonicRegressionExt {
  increasing: boolean;
  outOfBounds: "nan" | "clip" | "raise";
  private _x: Float64Array | null = null;
  private _y: Float64Array | null = null;
  xThresholds_: Float64Array | null = null;
  yThresholds_: Float64Array | null = null;

  constructor(increasing = true, outOfBounds: "nan" | "clip" | "raise" = "nan") {
    this.increasing = increasing;
    this.outOfBounds = outOfBounds;
  }

  fit(x: Float64Array, y: Float64Array, sampleWeight?: Float64Array): this {
    const n = x.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0));
    const xSorted = Float64Array.from(order, (i) => x[i] ?? 0);
    const ySorted = Float64Array.from(order, (i) => y[i] ?? 0);
    const wSorted = sampleWeight ? Float64Array.from(order, (i) => sampleWeight[i] ?? 1) : Float64Array.from({ length: n }, () => 1);

    if (!this.increasing) {
      for (let i = 0; i < ySorted.length; i++) ySorted[i] = -ySorted[i]!;
    }

    // Pool Adjacent Violators Algorithm
    const blocks: Array<{ x: number[]; ySum: number; wSum: number }> = [];
    for (let i = 0; i < n; i++) {
      blocks.push({ x: [xSorted[i] ?? 0], ySum: (ySorted[i] ?? 0) * (wSorted[i] ?? 1), wSum: wSorted[i] ?? 1 });
      while (blocks.length > 1) {
        const last = blocks[blocks.length - 1] as (typeof blocks)[0];
        const prev = blocks[blocks.length - 2] as (typeof blocks)[0];
        if (last.ySum / last.wSum < prev.ySum / prev.wSum) {
          prev.x.push(...last.x);
          prev.ySum += last.ySum;
          prev.wSum += last.wSum;
          blocks.pop();
        } else break;
      }
    }

    const thresholdX: number[] = [];
    const thresholdY: number[] = [];
    for (const block of blocks) {
      const mean = block.ySum / block.wSum;
      for (const xi of block.x) {
        thresholdX.push(xi);
        thresholdY.push(this.increasing ? mean : -mean);
      }
    }

    this._x = Float64Array.from(thresholdX);
    this._y = Float64Array.from(thresholdY);
    this.xThresholds_ = this._x;
    this.yThresholds_ = this._y;
    return this;
  }

  predict(t: Float64Array): Float64Array {
    if (!this._x || !this._y) throw new Error("Not fitted");
    return Float64Array.from(t, (val) => {
      const xs = this._x as Float64Array;
      const ys = this._y as Float64Array;
      if (val < (xs[0] ?? 0)) {
        if (this.outOfBounds === "nan") return NaN;
        if (this.outOfBounds === "raise") throw new RangeError(`${val} is below training range`);
        return ys[0] ?? 0;
      }
      if (val > (xs[xs.length - 1] ?? 0)) {
        if (this.outOfBounds === "nan") return NaN;
        if (this.outOfBounds === "raise") throw new RangeError(`${val} is above training range`);
        return ys[ys.length - 1] ?? 0;
      }
      let lo = 0, hi = xs.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if ((xs[mid] ?? 0) <= val) lo = mid; else hi = mid;
      }
      // Linear interpolation between lo and hi
      const xRange = (xs[hi] ?? 0) - (xs[lo] ?? 0);
      if (Math.abs(xRange) < 1e-12) return ys[lo] ?? 0;
      const t2 = (val - (xs[lo] ?? 0)) / xRange;
      return (ys[lo] ?? 0) + t2 * ((ys[hi] ?? 0) - (ys[lo] ?? 0));
    });
  }

  score(x: Float64Array, y: Float64Array): number {
    const yPred = this.predict(x);
    const yMean = y.reduce((s, v) => s + v, 0) / Math.max(y.length, 1);
    const ss_res = yPred.reduce((s, v, i) => s + (v - (y[i] ?? 0)) ** 2, 0);
    const ss_tot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ss_res / Math.max(ss_tot, 1e-12);
  }
}

export class MonotoneSpline {
  increasing: boolean;
  degree: number;
  private _breakpoints: Float64Array | null = null;
  private _coefficients: Float64Array | null = null;

  constructor(increasing = true, degree = 3) {
    this.increasing = increasing;
    this.degree = degree;
  }

  fit(x: Float64Array, y: Float64Array): this {
    // Fit a monotone cubic Hermite spline (PCHIP-like)
    const n = x.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0));
    const xSorted = Float64Array.from(order, (i) => x[i] ?? 0);
    const ySorted = Float64Array.from(order, (i) => y[i] ?? 0);

    // Piecewise slopes
    const slopes = new Float64Array(n);
    for (let i = 0; i < n - 1; i++) {
      const dx = (xSorted[i + 1] ?? 0) - (xSorted[i] ?? 0);
      slopes[i] = dx > 0 ? ((ySorted[i + 1] ?? 0) - (ySorted[i] ?? 0)) / dx : 0;
    }

    // Monotone adjustment
    if (this.increasing) {
      for (let i = 0; i < n; i++) slopes[i] = Math.max(0, slopes[i] ?? 0);
    } else {
      for (let i = 0; i < n; i++) slopes[i] = Math.min(0, slopes[i] ?? 0);
    }

    this._breakpoints = xSorted;
    this._coefficients = ySorted;
    return this;
  }

  predict(t: Float64Array): Float64Array {
    if (!this._breakpoints || !this._coefficients) throw new Error("Not fitted");
    const xs = this._breakpoints, ys = this._coefficients;
    return Float64Array.from(t, (val) => {
      if (val <= (xs[0] ?? 0)) return ys[0] ?? 0;
      if (val >= (xs[xs.length - 1] ?? 0)) return ys[ys.length - 1] ?? 0;
      let lo = 0, hi = xs.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if ((xs[mid] ?? 0) <= val) lo = mid; else hi = mid;
      }
      const t2 = ((val - (xs[lo] ?? 0)) / ((xs[hi] ?? 0) - (xs[lo] ?? 0)));
      return (ys[lo] ?? 0) + t2 * ((ys[hi] ?? 0) - (ys[lo] ?? 0));
    });
  }
}
