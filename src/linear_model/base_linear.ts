/**
 * Base classes and utilities for linear models — analogous to sklearn.linear_model._base.
 */

import { NotFittedError } from "../exceptions.js";

/** Result of _preprocessData: centered/scaled X and y, plus the offsets applied. */
export interface PreprocessDataResult {
  /** Preprocessed feature matrix (flat Float64Array, nSamples × nFeatures). */
  X: Float64Array;
  /** Preprocessed target vector. */
  y: Float64Array;
  /** Column means used to center X (or zeros if fitIntercept=false). */
  xMean: Float64Array;
  /** Column std-devs used to scale X (or ones if normalize=false). */
  xScale: Float64Array;
  /** Mean of y (or 0 if fitIntercept=false). */
  yMean: number;
}

/**
 * Centers (and optionally normalizes) X and y before fitting a linear model.
 * Returns copies; does not modify the input arrays.
 */
export function preprocessData(
  X: Float64Array,
  nSamples: number,
  nFeatures: number,
  y: Float64Array,
  fitIntercept: boolean,
  normalize = false,
): PreprocessDataResult {
  const xMean = new Float64Array(nFeatures);
  const xScale = new Float64Array(nFeatures).fill(1);
  let yMean = 0;

  const Xout = new Float64Array(X);
  const yOut = new Float64Array(y);

  if (!fitIntercept) {
    return { X: Xout, y: yOut, xMean, xScale, yMean };
  }

  // Compute column means
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++) xMean[j]! += Xout[i * nFeatures + j]!;
    yMean += yOut[i]!;
  }
  for (let j = 0; j < nFeatures; j++) xMean[j]! /= nSamples;
  yMean /= nSamples;

  // Center
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++) Xout[i * nFeatures + j]! -= xMean[j]!;
    yOut[i]! -= yMean;
  }

  if (normalize) {
    // Compute column L2 norms as scale
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const v = Xout[i * nFeatures + j]!;
        xScale[j]! += v * v;
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      const s = Math.sqrt(xScale[j]!);
      xScale[j] = s > 0 ? s : 1;
    }
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) Xout[i * nFeatures + j]! /= xScale[j]!;
    }
  }

  return { X: Xout, y: yOut, xMean, xScale, yMean };
}

/**
 * Sets the intercept_ from the precomputed means.
 * intercept = yMean - xMean · coef
 */
export function setIntercept(
  xMean: Float64Array,
  yMean: number,
  xScale: Float64Array,
  coef: Float64Array,
  fitIntercept: boolean,
): number {
  if (!fitIntercept) return 0;
  let intercept = yMean;
  for (let j = 0; j < coef.length; j++) {
    intercept -= (xMean[j]! / xScale[j]!) * coef[j]!;
  }
  return intercept;
}

/** Mixin providing LinearClassifierMixin.predict() from decision_function(). */
export abstract class LinearClassifierMixin {
  abstract classes_: Int32Array | undefined;

  abstract decisionFunction(X: Float64Array, nSamples: number, nFeatures: number): Float64Array;

  predict(X: Float64Array, nSamples: number, nFeatures: number): Int32Array {
    if (!this.classes_) throw new NotFittedError("LinearClassifierMixin is not fitted");
    const scores = this.decisionFunction(X, nSamples, nFeatures);
    const out = new Int32Array(nSamples);
    const nClasses = this.classes_.length;
    if (nClasses === 2) {
      for (let i = 0; i < nSamples; i++) out[i] = scores[i]! > 0 ? this.classes_[1]! : this.classes_[0]!;
    } else {
      for (let i = 0; i < nSamples; i++) {
        let best = 0;
        let bestScore = Number.NEGATIVE_INFINITY;
        for (let k = 0; k < nClasses; k++) {
          const s = scores[i * nClasses + k]!;
          if (s > bestScore) { bestScore = s; best = k; }
        }
        out[i] = this.classes_[best]!;
      }
    }
    return out;
  }
}

/** Mixin for sparse coefficient storage and sparsify()/densify(). */
export class SparseCoefMixin {
  coef_: Float64Array | undefined;
  /** Indices of non-zero coefficients (populated after sparsify()). */
  sparseIndices_: Int32Array | undefined;
  /** Values of non-zero coefficients (populated after sparsify()). */
  sparseValues_: Float64Array | undefined;

  /** Convert coef_ to sparse representation. */
  sparsify(): this {
    if (!this.coef_) throw new NotFittedError("SparseCoefMixin is not fitted");
    const indices: number[] = [];
    const values: number[] = [];
    for (let j = 0; j < this.coef_.length; j++) {
      const v = this.coef_[j]!;
      if (v !== 0) { indices.push(j); values.push(v); }
    }
    this.sparseIndices_ = new Int32Array(indices);
    this.sparseValues_ = new Float64Array(values);
    return this;
  }

  /** Restore dense coef_ from sparse representation. */
  densify(): this {
    if (!this.sparseIndices_ || !this.sparseValues_) return this;
    const n = this.coef_?.length ?? 0;
    const dense = new Float64Array(n);
    for (let k = 0; k < this.sparseIndices_.length; k++) {
      dense[this.sparseIndices_[k]!] = this.sparseValues_[k]!;
    }
    this.coef_ = dense;
    return this;
  }
}
