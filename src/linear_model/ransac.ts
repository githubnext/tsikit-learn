/**
 * RANSAC (Random Sample Consensus) regressor.
 * Mirrors sklearn.linear_model.RANSACRegressor.
 */

import { NotFittedError } from "../exceptions.js";

export class RANSACRegressor {
  minSamples: number;
  residualThreshold: number;
  maxTrials: number;
  randomState: number;

  estimator_: { coef_: Float64Array; intercept_: number } | null = null;
  inlierMask_: Int8Array | null = null;
  nTrialsReached_: number = 0;

  constructor(
    options: {
      minSamples?: number;
      residualThreshold?: number;
      maxTrials?: number;
      randomState?: number;
    } = {},
  ) {
    this.minSamples = options.minSamples ?? 0.1;
    this.residualThreshold = options.residualThreshold ?? 1.0;
    this.maxTrials = options.maxTrials ?? 100;
    this.randomState = options.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const minSamples =
      this.minSamples < 1
        ? Math.max(2, Math.round(this.minSamples * n))
        : Math.round(this.minSamples);

    let bestInlierCount = 0;
    let bestCoef: Float64Array = new Float64Array(nFeatures);
    let bestIntercept = 0;
    let bestMask = new Int8Array(n);

    let rng = this.randomState;
    const nextRand = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return rng / 4294967296;
    };

    for (let trial = 0; trial < this.maxTrials; trial++) {
      // Random subset
      const indices: number[] = [];
      const pool = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < minSamples; i++) {
        const j = Math.floor(nextRand() * (pool.length - i)) + i;
        const tmp = pool[i] ?? 0;
        pool[i] = pool[j] ?? 0;
        pool[j] = tmp;
        indices.push(pool[i] ?? 0);
      }

      // Fit OLS on subset
      const { coef, intercept } = this._fitOLS(
        indices.map((i) => X[i] ?? new Float64Array(nFeatures)),
        indices.map((i) => y[i] ?? 0),
        nFeatures,
      );

      // Count inliers
      let inlierCount = 0;
      const mask = new Int8Array(n);
      for (let i = 0; i < n; i++) {
        const pred = this._predict(
          X[i] ?? new Float64Array(nFeatures),
          coef,
          intercept,
        );
        const residual = Math.abs((y[i] ?? 0) - pred);
        if (residual <= this.residualThreshold) {
          mask[i] = 1;
          inlierCount++;
        }
      }

      if (inlierCount > bestInlierCount) {
        bestInlierCount = inlierCount;
        bestCoef = coef;
        bestIntercept = intercept;
        bestMask = mask;
        this.nTrialsReached_ = trial + 1;
      }
    }

    // Refit on all inliers
    const inlierX: Float64Array[] = [];
    const inlierY: number[] = [];
    for (let i = 0; i < n; i++) {
      if (bestMask[i] === 1) {
        inlierX.push(X[i] ?? new Float64Array(nFeatures));
        inlierY.push(y[i] ?? 0);
      }
    }
    if (inlierX.length > 0) {
      const { coef, intercept } = this._fitOLS(inlierX, inlierY, nFeatures);
      bestCoef = coef;
      bestIntercept = intercept;
    }

    this.estimator_ = { coef_: bestCoef, intercept_: bestIntercept };
    this.inlierMask_ = bestMask;
    return this;
  }

  private _fitOLS(
    X: Float64Array[],
    y: number[],
    nFeatures: number,
  ): { coef: Float64Array; intercept: number } {
    const n = X.length;
    if (n === 0) return { coef: new Float64Array(nFeatures), intercept: 0 };

    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= n;

    const xMean = new Float64Array(nFeatures);
    for (const xi of X) {
      for (let j = 0; j < nFeatures; j++)
        xMean[j] = (xMean[j] ?? 0) + (xi[j] ?? 0);
    }
    for (let j = 0; j < nFeatures; j++) xMean[j] = (xMean[j] ?? 0) / n;

    // Simple single-feature OLS for now
    const coef = new Float64Array(nFeatures);
    if (nFeatures === 1) {
      let num = 0;
      let den = 0;
      for (let i = 0; i < n; i++) {
        const xc = (X[i]?.[0] ?? 0) - (xMean[0] ?? 0);
        const yc = (y[i] ?? 0) - yMean;
        num += xc * yc;
        den += xc * xc;
      }
      coef[0] = den !== 0 ? num / den : 0;
    }
    const intercept = yMean - this._dot(xMean, coef);
    return { coef, intercept };
  }

  private _dot(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
    return s;
  }

  private _predict(
    x: Float64Array,
    coef: Float64Array,
    intercept: number,
  ): number {
    return this._dot(x, coef) + intercept;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.estimator_)
      throw new NotFittedError("RANSACRegressor is not fitted");
    const { coef_, intercept_ } = this.estimator_;
    const result = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      result[i] = this._predict(X[i] ?? new Float64Array(0), coef_, intercept_);
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let yMean = 0;
    for (const yi of y) yMean += yi;
    yMean /= y.length;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
