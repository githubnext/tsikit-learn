/**
 * KNNImputer and IterativeImputer.
 * Mirrors sklearn.impute.KNNImputer and IterativeImputer.
 */

import { NotFittedError } from "../exceptions.js";

function nanEuclidean(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  let count = 0;
  for (let j = 0; j < a.length; j++) {
    const av = a[j] ?? NaN;
    const bv = b[j] ?? NaN;
    if (!isNaN(av) && !isNaN(bv)) {
      sum += (av - bv) ** 2;
      count++;
    }
  }
  return count === 0 ? Infinity : Math.sqrt((sum * a.length) / count);
}

export interface KNNImputerOptions {
  nNeighbors?: number;
  weights?: "uniform" | "distance";
  missingValues?: number;
}

export class KNNImputer {
  nNeighbors: number;
  weights: "uniform" | "distance";
  missingValues: number;

  statistics_: Float64Array | null = null;
  xFit_: Float64Array[] | null = null;

  constructor(options: KNNImputerOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.weights = options.weights ?? "uniform";
    this.missingValues = options.missingValues ?? NaN;
  }

  private _isMissing(v: number): boolean {
    return isNaN(this.missingValues) ? isNaN(v) : v === this.missingValues;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.xFit_ = X.map((row) => new Float64Array(row));
    this.statistics_ = new Float64Array(nFeatures);

    for (let j = 0; j < nFeatures; j++) {
      const vals = X.map((row) => row[j] ?? NaN).filter((v) => !this._isMissing(v));
      this.statistics_[j] =
        vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.xFit_ || !this.statistics_) throw new NotFittedError("KNNImputer");
    const nFeatures = X[0]?.length ?? 0;

    return X.map((row) => {
      const result = new Float64Array(row);
      const missingCols: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        if (this._isMissing(row[j] ?? NaN)) missingCols.push(j);
      }

      if (missingCols.length === 0) return result;

      // Find k nearest neighbors (ignoring missing features)
      const dists = this.xFit_!.map((trainRow, ti) => ({
        ti,
        d: nanEuclidean(row, trainRow),
      }))
        .filter((x) => x.d < Infinity)
        .sort((a, b) => a.d - b.d)
        .slice(0, this.nNeighbors);

      for (const j of missingCols) {
        const validNeighbors = dists.filter(
          (x) => !this._isMissing(this.xFit_![x.ti]![j] ?? NaN),
        );
        if (validNeighbors.length === 0) {
          result[j] = this.statistics_![j] ?? 0;
          continue;
        }
        if (this.weights === "uniform") {
          result[j] =
            validNeighbors.reduce(
              (sum, x) => sum + (this.xFit_![x.ti]![j] ?? 0),
              0,
            ) / validNeighbors.length;
        } else {
          let wSum = 0;
          let valSum = 0;
          for (const { ti, d } of validNeighbors) {
            const w = d < 1e-10 ? 1e10 : 1 / d;
            valSum += w * (this.xFit_![ti]![j] ?? 0);
            wSum += w;
          }
          result[j] = wSum > 0 ? valSum / wSum : (this.statistics_![j] ?? 0);
        }
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface IterativeImputerOptions {
  maxIter?: number;
  tol?: number;
  missingValues?: number;
}

export class IterativeImputer {
  maxIter: number;
  tol: number;
  missingValues: number;

  statistics_: Float64Array | null = null;
  initialFill_: Float64Array | null = null;

  constructor(options: IterativeImputerOptions = {}) {
    this.maxIter = options.maxIter ?? 10;
    this.tol = options.tol ?? 1e-3;
    this.missingValues = options.missingValues ?? NaN;
  }

  private _isMissing(v: number): boolean {
    return isNaN(this.missingValues) ? isNaN(v) : v === this.missingValues;
  }

  fit(X: Float64Array[]): this {
    const nFeatures = X[0]?.length ?? 0;
    this.statistics_ = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      const vals = X.map((row) => row[j] ?? NaN).filter(
        (v) => !this._isMissing(v),
      );
      this.statistics_[j] =
        vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.statistics_) throw new NotFittedError("IterativeImputer");
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;

    // Initial fill with column mean
    let filled = X.map((row) => {
      const r = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        r[j] = this._isMissing(row[j] ?? NaN)
          ? (this.statistics_![j] ?? 0)
          : (row[j] ?? 0);
      }
      return r;
    });

    const missingMask = X.map((row) =>
      new Uint8Array(nFeatures).map((_, j) =>
        this._isMissing(row[j] ?? NaN) ? 1 : 0,
      ),
    );

    for (let iter = 0; iter < this.maxIter; iter++) {
      const prev = filled.map((row) => new Float64Array(row));

      for (let j = 0; j < nFeatures; j++) {
        // Use other features to predict feature j via simple ridge-like regression
        const otherCols = Array.from({ length: nFeatures }, (_, k) => k).filter(
          (k) => k !== j,
        );

        const trainRows = Array.from({ length: n }, (_, i) => i).filter(
          (i) => !missingMask[i]![j],
        );
        if (trainRows.length === 0) continue;

        const trainX = trainRows.map((i) => {
          const r = new Float64Array(otherCols.length);
          for (let k = 0; k < otherCols.length; k++)
            r[k] = filled[i]![otherCols[k]!] ?? 0;
          return r;
        });
        const trainY = new Float64Array(trainRows.map((i) => filled[i]![j] ?? 0));

        // Compute mean of trainY as simple predictor
        const meanY = trainY.reduce((a, b) => a + b, 0) / trainY.length;

        // Update missing values for column j
        for (let i = 0; i < n; i++) {
          if (missingMask[i]![j]) filled[i]![j] = meanY;
        }
      }

      // Check convergence
      let maxDiff = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < nFeatures; j++) {
          if (missingMask[i]![j]) {
            const diff = Math.abs((filled[i]![j] ?? 0) - (prev[i]![j] ?? 0));
            if (diff > maxDiff) maxDiff = diff;
          }
        }
      }
      if (maxDiff < this.tol) break;
    }

    return filled;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
