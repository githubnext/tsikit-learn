/**
 * Isotonic Regression.
 * Mirrors sklearn.isotonic.IsotonicRegression.
 */

import { NotFittedError } from "../exceptions.js";

/** Pool Adjacent Violators (PAV) algorithm for isotonic regression. */
function poolAdjacentViolators(y: Float64Array, increasing: boolean): Float64Array {
  const n = y.length;
  const result = new Float64Array(y);

  // Simple PAVA
  let changed = true;
  while (changed) {
    changed = false;
    let i = 0;
    while (i < n - 1) {
      if (increasing ? (result[i] ?? 0) > (result[i + 1] ?? 0) : (result[i] ?? 0) < (result[i + 1] ?? 0)) {
        // Merge block
        const mean = ((result[i] ?? 0) + (result[i + 1] ?? 0)) / 2;
        result[i] = mean;
        result[i + 1] = mean;
        changed = true;
      }
      i++;
    }
  }

  return result;
}

export class IsotonicRegression {
  increasing: boolean | "auto";
  outOfBounds: string;

  XThresholds_: Float64Array | null = null;
  yThresholds_: Float64Array | null = null;

  constructor(
    options: { increasing?: boolean | "auto"; outOfBounds?: string } = {},
  ) {
    this.increasing = options.increasing ?? true;
    this.outOfBounds = options.outOfBounds ?? "nan";
  }

  fit(X: Float64Array, y: Float64Array): this {
    const n = X.length;
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (X[a] ?? 0) - (X[b] ?? 0),
    );

    const xSorted = new Float64Array(order.map((i) => X[i] ?? 0));
    const ySorted = new Float64Array(order.map((i) => y[i] ?? 0));

    const incr =
      this.increasing === "auto"
        ? (() => {
            // Estimate direction from correlation
            const xMean = Array.from(xSorted).reduce((a, b) => a + b, 0) / n;
            const yMean = Array.from(ySorted).reduce((a, b) => a + b, 0) / n;
            let cov = 0;
            for (let i = 0; i < n; i++) {
              cov += ((xSorted[i] ?? 0) - xMean) * ((ySorted[i] ?? 0) - yMean);
            }
            return cov >= 0;
          })()
        : this.increasing;

    const fitted = poolAdjacentViolators(ySorted, incr as boolean);

    this.XThresholds_ = xSorted;
    this.yThresholds_ = fitted;

    return this;
  }

  predict(X: Float64Array): Float64Array {
    if (this.XThresholds_ === null || this.yThresholds_ === null)
      throw new NotFittedError("IsotonicRegression");

    const xThresh = this.XThresholds_;
    const yThresh = this.yThresholds_;

    return new Float64Array(
      Array.from(X).map((xi) => {
        if (xi <= (xThresh[0] ?? xi)) return yThresh[0] ?? 0;
        if (xi >= (xThresh[xThresh.length - 1] ?? xi)) return yThresh[yThresh.length - 1] ?? 0;

        // Binary search for interpolation
        let lo = 0;
        let hi = xThresh.length - 1;
        while (lo < hi - 1) {
          const mid = Math.floor((lo + hi) / 2);
          if ((xThresh[mid] ?? 0) <= xi) lo = mid;
          else hi = mid;
        }

        const x0 = xThresh[lo] ?? 0;
        const x1 = xThresh[hi] ?? 0;
        const y0 = yThresh[lo] ?? 0;
        const y1 = yThresh[hi] ?? 0;

        if (x1 === x0) return (y0 + y1) / 2;
        return y0 + ((y1 - y0) * (xi - x0)) / (x1 - x0);
      }),
    );
  }

  score(X: Float64Array, y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}
