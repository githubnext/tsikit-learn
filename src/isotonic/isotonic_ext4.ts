/**
 * Extended isotonic regression utilities.
 * Port of sklearn.isotonic extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Isotonic regression with weighted observations.
 */
export class WeightedIsotonicRegression {
  private yThresh_: Float64Array = new Float64Array(0);
  private xThresh_: Float64Array = new Float64Array(0);
  private fitted = false;

  fit(X: Float64Array, y: Float64Array, sampleWeight?: Float64Array): this {
    const n = X.length;
    const w = sampleWeight ?? new Float64Array(n).fill(1.0);

    // Sort by X
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (X[a] ?? 0) - (X[b] ?? 0));
    const xs = Float64Array.from(idx, i => X[i] ?? 0);
    const ys = Float64Array.from(idx, i => y[i] ?? 0);
    const ws = Float64Array.from(idx, i => w[i] ?? 1);

    // Pool adjacent violators
    const blocks: Array<{ sumY: number; sumW: number; x: number }> = [];
    for (let i = 0; i < n; i++) {
      blocks.push({ sumY: (ys[i] ?? 0) * (ws[i] ?? 1), sumW: ws[i] ?? 1, x: xs[i] ?? 0 });
      while (blocks.length >= 2) {
        const last = blocks[blocks.length - 1]!;
        const prev = blocks[blocks.length - 2]!;
        if (last.sumY / last.sumW < prev.sumY / prev.sumW) {
          blocks.splice(blocks.length - 2, 2, {
            sumY: last.sumY + prev.sumY,
            sumW: last.sumW + prev.sumW,
            x: prev.x,
          });
        } else break;
      }
    }

    this.xThresh_ = Float64Array.from(blocks, b => b.x);
    this.yThresh_ = Float64Array.from(blocks, b => b.sumY / b.sumW);
    this.fitted = true;
    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (!this.fitted) throw new NotFittedError("WeightedIsotonicRegression not fitted");
    const out = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const x = X[i] ?? 0;
      // Find interpolation position
      let lo = 0; let hi = this.xThresh_.length - 1;
      if (x <= (this.xThresh_[0] ?? 0)) { out[i] = this.yThresh_[0] ?? 0; continue; }
      if (x >= (this.xThresh_[hi] ?? 0)) { out[i] = this.yThresh_[hi] ?? 0; continue; }
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if ((this.xThresh_[mid] ?? 0) <= x) lo = mid; else hi = mid;
      }
      const x0 = this.xThresh_[lo] ?? 0;
      const x1 = this.xThresh_[hi] ?? 1;
      const t = x1 > x0 ? (x - x0) / (x1 - x0) : 0;
      out[i] = (this.yThresh_[lo] ?? 0) * (1 - t) + (this.yThresh_[hi] ?? 0) * t;
    }
    return out;
  }
}

/**
 * Antitonic (monotone decreasing) regression.
 */
export class AntitonicRegression {
  private inner_: WeightedIsotonicRegression;

  constructor() {
    this.inner_ = new WeightedIsotonicRegression();
  }

  fit(X: Float64Array, y: Float64Array, sampleWeight?: Float64Array): this {
    // Negate y, fit isotonic, will give decreasing
    const negY = Float64Array.from(y, v => -v);
    this.inner_.fit(X, negY, sampleWeight);
    return this;
  }

  predict(X: Float64Array): Float64Array {
    const pred = this.inner_.predict(X);
    return Float64Array.from(pred, v => -v);
  }
}

/**
 * Piecewise linear isotonic regression with knot detection.
 */
export class PiecewiseIsotonicRegression {
  private knots_: Float64Array = new Float64Array(0);
  private slopes_: Float64Array = new Float64Array(0);
  private intercepts_: Float64Array = new Float64Array(0);
  private fitted = false;

  fit(X: Float64Array, y: Float64Array): this {
    const base = new WeightedIsotonicRegression();
    base.fit(X, y);
    const yHat = base.predict(X);

    // Detect change points (knots)
    const knots = [X[0] ?? 0];
    const slopes: number[] = [];
    const intercepts: number[] = [];

    let segStart = 0;
    for (let i = 1; i < X.length; i++) {
      const dx = (X[i] ?? 0) - (X[segStart] ?? 0);
      const dy = (yHat[i] ?? 0) - (yHat[segStart] ?? 0);
      // Check if slope changes significantly
      if (i < X.length - 1) {
        const slope1 = dx > 0 ? dy / dx : 0;
        const dx2 = (X[i + 1] ?? 0) - (X[i] ?? 0);
        const dy2 = (yHat[i + 1] ?? 0) - (yHat[i] ?? 0);
        const slope2 = dx2 > 0 ? dy2 / dx2 : 0;
        if (Math.abs(slope2 - slope1) > 1e-6) {
          knots.push(X[i] ?? 0);
          slopes.push(slope1);
          intercepts.push((yHat[segStart] ?? 0) - slope1 * (X[segStart] ?? 0));
          segStart = i;
        }
      }
    }
    const finalDx = (X[X.length - 1] ?? 0) - (X[segStart] ?? 0);
    const finalDy = (yHat[y.length - 1] ?? 0) - (yHat[segStart] ?? 0);
    const finalSlope = finalDx > 0 ? finalDy / finalDx : 0;
    slopes.push(finalSlope);
    intercepts.push((yHat[segStart] ?? 0) - finalSlope * (X[segStart] ?? 0));

    this.knots_ = Float64Array.from(knots);
    this.slopes_ = Float64Array.from(slopes);
    this.intercepts_ = Float64Array.from(intercepts);
    this.fitted = true;
    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (!this.fitted) throw new NotFittedError("PiecewiseIsotonicRegression not fitted");
    return Float64Array.from(X, x => {
      // Find segment
      let seg = this.knots_.length - 1;
      for (let k = 0; k < this.knots_.length - 1; k++) {
        if (x < (this.knots_[k + 1] ?? Number.POSITIVE_INFINITY)) { seg = k; break; }
      }
      return (this.slopes_[seg] ?? 0) * x + (this.intercepts_[seg] ?? 0);
    });
  }
}

/**
 * Check if an array is monotonically increasing.
 */
export function isMonotoneIncreasing(arr: Float64Array, strict = false): boolean {
  for (let i = 1; i < arr.length; i++) {
    if (strict ? (arr[i] ?? 0) <= (arr[i - 1] ?? 0) : (arr[i] ?? 0) < (arr[i - 1] ?? 0)) return false;
  }
  return true;
}

/**
 * Check if an array is monotonically decreasing.
 */
export function isMonotoneDecreasing(arr: Float64Array, strict = false): boolean {
  for (let i = 1; i < arr.length; i++) {
    if (strict ? (arr[i] ?? 0) >= (arr[i - 1] ?? 0) : (arr[i] ?? 0) > (arr[i - 1] ?? 0)) return false;
  }
  return true;
}
