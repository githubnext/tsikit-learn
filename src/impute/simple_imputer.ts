/**
 * Imputers for missing values.
 * Mirrors sklearn.impute.SimpleImputer.
 */

import { NotFittedError } from "../exceptions.js";

export class SimpleImputer {
  strategy: string;
  fillValue: number;
  missingValues: number;

  statistics_: Float64Array | null = null;

  constructor(
    options: {
      strategy?: string;
      fillValue?: number;
      missingValues?: number;
    } = {},
  ) {
    this.strategy = options.strategy ?? "mean";
    this.fillValue = options.fillValue ?? 0;
    this.missingValues = options.missingValues ?? Number.NaN;
  }

  private _isMissing(x: number): boolean {
    return Number.isNaN(this.missingValues)
      ? Number.isNaN(x)
      : x === this.missingValues;
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    const stats = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const vals: number[] = [];
      for (const xi of X) {
        const v = xi[j] ?? 0;
        if (!this._isMissing(v)) vals.push(v);
      }

      if (this.strategy === "mean") {
        stats[j] =
          vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      } else if (this.strategy === "median") {
        vals.sort((a, b) => a - b);
        const mid = Math.floor(vals.length / 2);
        stats[j] =
          vals.length % 2 === 0
            ? ((vals[mid - 1] ?? 0) + (vals[mid] ?? 0)) / 2
            : (vals[mid] ?? 0);
      } else if (this.strategy === "most_frequent") {
        const counts = new Map<number, number>();
        for (const v of vals) counts.set(v, (counts.get(v) ?? 0) + 1);
        let best = 0;
        let bestCnt = 0;
        for (const [v, cnt] of counts) {
          if (cnt > bestCnt) {
            bestCnt = cnt;
            best = v;
          }
        }
        stats[j] = best;
      } else {
        stats[j] = this.fillValue;
      }
    }

    this.statistics_ = stats;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.statistics_ === null) throw new NotFittedError("SimpleImputer");
    const stats = this.statistics_;
    return X.map((xi) => {
      const result = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        const v = xi[j] ?? 0;
        result[j] = this._isMissing(v) ? (stats[j] ?? 0) : v;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
