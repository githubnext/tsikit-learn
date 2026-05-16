/**
 * Additional preprocessing utilities: KernelCenterer, MaxAbsScaler (if needed),
 * and other sklearn.preprocessing functions not yet ported.
 * Mirrors sklearn.preprocessing.KernelCenterer, add_dummy_feature, etc.
 */

import { BaseEstimator } from "../base.js";
import { NotFittedError } from "../exceptions.js";

/**
 * KernelCenterer: center a kernel matrix.
 * Mirrors sklearn.preprocessing.KernelCenterer.
 */
export class KernelCenterer extends BaseEstimator {
  kFitRows_: Float64Array | null = null;
  kFitAll_: number | null = null;
  nSamplesFit_: number | null = null;

  fit(K: Float64Array[]): this {
    const n = K.length;
    this.nSamplesFit_ = n;
    const rowMeans = new Float64Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
      let rowSum = 0;
      for (let j = 0; j < n; j++) rowSum += K[i]![j] ?? 0;
      rowMeans[i] = rowSum / n;
      total += rowSum;
    }
    this.kFitRows_ = rowMeans;
    this.kFitAll_ = total / (n * n);
    return this;
  }

  transform(K: Float64Array[]): Float64Array[] {
    if (this.kFitRows_ === null || this.kFitAll_ === null) {
      throw new NotFittedError("KernelCenterer");
    }
    const nTest = K.length;
    const nTrain = this.kFitRows_.length;
    const result: Float64Array[] = [];
    for (let i = 0; i < nTest; i++) {
      const row = new Float64Array(nTrain);
      let rowMean = 0;
      for (let j = 0; j < nTrain; j++) rowMean += K[i]![j] ?? 0;
      rowMean /= nTrain;
      for (let j = 0; j < nTrain; j++) {
        row[j] = (K[i]![j] ?? 0) - rowMean - (this.kFitRows_![j] ?? 0) + this.kFitAll_!;
      }
      result.push(row);
    }
    return result;
  }

  fitTransform(K: Float64Array[]): Float64Array[] {
    return this.fit(K).transform(K);
  }
}
