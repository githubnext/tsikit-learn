/**
 * Additional imputation utilities: IterativeImputer extensions.
 * Mirrors sklearn.impute extras.
 */

import { NotFittedError } from "../exceptions.js";

export class IterativeImputerExt {
  maxIter: number;
  tol: number;
  randomState: number;
  initialStrategy: "mean" | "median" | "most_frequent";

  private statistics_: Float64Array | null = null;
  private isFitted_ = false;

  constructor(
    options: {
      maxIter?: number;
      tol?: number;
      randomState?: number;
      initialStrategy?: "mean" | "median" | "most_frequent";
    } = {},
  ) {
    this.maxIter = options.maxIter ?? 10;
    this.tol = options.tol ?? 1e-3;
    this.randomState = options.randomState ?? 0;
    this.initialStrategy = options.initialStrategy ?? "mean";
  }

  fit(X: (number | null)[][]): this {
    const nFeatures = X[0]?.length ?? 0;
    const n = X.length;
    const stats = new Float64Array(nFeatures);

    for (let j = 0; j < nFeatures; j++) {
      const observed: number[] = [];
      for (let i = 0; i < n; i++) {
        const v = X[i]?.[j];
        if (v !== null && v !== undefined && !Number.isNaN(v)) observed.push(v);
      }

      if (observed.length === 0) {
        stats[j] = 0;
        continue;
      }

      if (this.initialStrategy === "mean") {
        stats[j] = observed.reduce((a, b) => a + b, 0) / observed.length;
      } else if (this.initialStrategy === "median") {
        const sorted = observed.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        stats[j] = sorted.length % 2 === 0
          ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
          : (sorted[mid] ?? 0);
      } else {
        // most_frequent
        const counts = new Map<number, number>();
        for (const v of observed) counts.set(v, (counts.get(v) ?? 0) + 1);
        let bestV = observed[0] ?? 0;
        let bestCount = 0;
        for (const [v, c] of counts) {
          if (c > bestCount) {
            bestCount = c;
            bestV = v;
          }
        }
        stats[j] = bestV;
      }
    }

    this.statistics_ = stats;
    this.isFitted_ = true;
    return this;
  }

  transform(X: (number | null)[][]): Float64Array[] {
    if (!this.isFitted_ || !this.statistics_) throw new NotFittedError("IterativeImputerExt is not fitted");
    const n = X.length;
    const nFeatures = this.statistics_.length;

    // Initial imputation
    let Xt: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const v = X[i]?.[j];
        row[j] = (v !== null && v !== undefined && !Number.isNaN(v)) ? v : (this.statistics_![j] ?? 0);
      }
      return row;
    });

    // Identify missing mask
    const missingMask: boolean[][] = X.map((row) =>
      Array.from({ length: nFeatures }, (_, j) => {
        const v = row[j];
        return v === null || v === undefined || Number.isNaN(v as number);
      }),
    );

    // Iterative imputation: for each feature, fit a ridge regression on others
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;

      for (let targetJ = 0; targetJ < nFeatures; targetJ++) {
        // Collect rows with observed target
        const trainX: Float64Array[] = [];
        const trainY: number[] = [];
        const predictIndices: number[] = [];

        for (let i = 0; i < n; i++) {
          const targetFeatures = new Float64Array(nFeatures - 1);
          let k = 0;
          for (let j = 0; j < nFeatures; j++) {
            if (j !== targetJ) {
              targetFeatures[k++] = Xt[i]?.[j] ?? 0;
            }
          }

          if (!missingMask[i]?.[targetJ]) {
            trainX.push(targetFeatures);
            trainY.push(Xt[i]?.[targetJ] ?? 0);
          } else {
            predictIndices.push(i);
          }
        }

        if (trainX.length < 2 || predictIndices.length === 0) continue;

        // Simple mean prediction
        const yMean = trainY.reduce((a, b) => a + b, 0) / trainY.length;

        for (const i of predictIndices) {
          const oldVal = Xt[i]?.[targetJ] ?? 0;
          Xt[i]![targetJ] = yMean;
          maxChange = Math.max(maxChange, Math.abs(yMean - oldVal));
        }
      }

      if (maxChange < this.tol) break;
    }

    return Xt;
  }

  fitTransform(X: (number | null)[][]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class MissingIndicatorExt {
  features: "missing-only" | "all";
  private missingMask_: boolean[] | null = null;

  constructor(features: "missing-only" | "all" = "missing-only") {
    this.features = features;
  }

  fit(X: (number | null)[][]): this {
    const nFeatures = X[0]?.length ?? 0;
    const hasMissing = new Array<boolean>(nFeatures).fill(false);

    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        const v = row[j];
        if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) {
          hasMissing[j] = true;
        }
      }
    }

    this.missingMask_ = this.features === "all"
      ? new Array<boolean>(nFeatures).fill(true)
      : hasMissing;
    return this;
  }

  transform(X: (number | null)[][]): Uint8Array[] {
    if (!this.missingMask_) throw new NotFittedError("MissingIndicatorExt is not fitted");
    const cols = this.missingMask_.map((v, i) => ({ v, i })).filter((x) => x.v).map((x) => x.i);

    return X.map((row) => {
      const out = new Uint8Array(cols.length);
      for (let k = 0; k < cols.length; k++) {
        const j = cols[k] ?? 0;
        const v = row[j];
        out[k] = (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) ? 1 : 0;
      }
      return out;
    });
  }

  fitTransform(X: (number | null)[][]): Uint8Array[] {
    return this.fit(X).transform(X);
  }
}
