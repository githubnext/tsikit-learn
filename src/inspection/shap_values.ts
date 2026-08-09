/**
 * Linear SHAP (SHapley Additive exPlanations) for linear models.
 * Analogous to shap.LinearExplainer and a tree-based variant.
 *
 * Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017).
 */

import { NotFittedError } from "../exceptions.js";

/** SHAP explanation for a set of samples. */
export interface SHAPExplanation {
  /** SHAP values matrix: nSamples × nFeatures. */
  values: Float64Array;
  /** Base values (expected model output per sample, length nSamples). */
  baseValues: Float64Array;
  /** Number of samples. */
  nSamples: number;
  /** Number of features. */
  nFeatures: number;
}

/**
 * LinearExplainer computes exact SHAP values for linear models.
 *
 * For a linear model f(x) = coef · x + intercept, the SHAP value for feature j is:
 *   φ_j(x) = coef_j * (x_j − E[x_j])
 * and the base value is:
 *   f(E[x]) = coef · E[x] + intercept
 */
export class LinearExplainer {
  private coef_: Float64Array | undefined;
  private intercept_: number | undefined;
  private featureMeans_: Float64Array | undefined;
  private nFeatures_: number | undefined;

  /**
   * Initialises the explainer from a fitted linear model.
   *
   * @param coef       Model coefficients (length nFeatures).
   * @param intercept  Model intercept.
   * @param featureMeans Background feature means (E[x_j]). If omitted, zeros are used.
   */
  fit(
    coef: Float64Array,
    intercept: number,
    featureMeans?: Float64Array,
  ): this {
    this.coef_ = new Float64Array(coef);
    this.intercept_ = intercept;
    this.nFeatures_ = coef.length;
    this.featureMeans_ = featureMeans
      ? new Float64Array(featureMeans)
      : new Float64Array(coef.length);
    return this;
  }

  /**
   * Computes SHAP values for X.
   *
   * @param X        Flat Float64Array of shape (nSamples × nFeatures).
   * @param nSamples Number of samples.
   */
  explain(X: Float64Array, nSamples: number): SHAPExplanation {
    if (
      !this.coef_ ||
      this.intercept_ === undefined ||
      !this.featureMeans_ ||
      !this.nFeatures_
    ) {
      throw new NotFittedError(
        "LinearExplainer is not fitted. Call fit() first.",
      );
    }
    const nFeatures = this.nFeatures_;
    const values = new Float64Array(nSamples * nFeatures);
    const baseValues = new Float64Array(nSamples);

    // Base value = coef · featureMeans + intercept (same for every sample)
    let baseValue = this.intercept_;
    for (let j = 0; j < nFeatures; j++)
      baseValue += this.coef_[j]! * this.featureMeans_[j]!;

    for (let i = 0; i < nSamples; i++) {
      baseValues[i] = baseValue;
      for (let j = 0; j < nFeatures; j++) {
        values[i * nFeatures + j] =
          this.coef_[j]! * (X[i * nFeatures + j]! - this.featureMeans_[j]!);
      }
    }
    return { values, baseValues, nSamples, nFeatures };
  }
}

/** Options for TreeSHAPExplainer. */
export interface TreeSHAPExplainerOptions {
  /** Maximum tree depth (prunes attribution to this depth). Default: unlimited. */
  maxDepth?: number;
}

/**
 * A simplified tree SHAP explainer that computes feature importance
 * via marginal contributions over tree paths.
 *
 * This is a lightweight implementation that works with the DecisionTree
 * internal structure (feature indices, thresholds, left/right child arrays).
 */
export class TreeSHAPExplainer {
  private featureIndex_: Int32Array | undefined;
  private threshold_: Float64Array | undefined;
  private leftChild_: Int32Array | undefined;
  private rightChild_: Int32Array | undefined;
  private leafValues_: Float64Array | undefined;
  private nFeatures_: number | undefined;

  /**
   * Fits the explainer to a decision tree's internal arrays.
   *
   * @param featureIndex  Per-node split feature index (-1 for leaves).
   * @param threshold     Per-node split threshold (0 for leaves).
   * @param leftChild     Per-node left child index (-1 for leaves).
   * @param rightChild    Per-node right child index (-1 for leaves).
   * @param leafValues    Per-leaf prediction value (indexed by node id).
   * @param nFeatures     Number of features.
   */
  fit(
    featureIndex: Int32Array,
    threshold: Float64Array,
    leftChild: Int32Array,
    rightChild: Int32Array,
    leafValues: Float64Array,
    nFeatures: number,
  ): this {
    this.featureIndex_ = featureIndex;
    this.threshold_ = threshold;
    this.leftChild_ = leftChild;
    this.rightChild_ = rightChild;
    this.leafValues_ = leafValues;
    this.nFeatures_ = nFeatures;
    return this;
  }

  /**
   * Computes approximate SHAP values for X via path-based attribution.
   *
   * @param X        Flat Float64Array of shape (nSamples × nFeatures).
   * @param nSamples Number of samples.
   */
  explain(X: Float64Array, nSamples: number): SHAPExplanation {
    if (
      !this.featureIndex_ ||
      !this.threshold_ ||
      !this.leftChild_ ||
      !this.rightChild_ ||
      !this.leafValues_ ||
      this.nFeatures_ === undefined
    ) {
      throw new NotFittedError(
        "TreeSHAPExplainer is not fitted. Call fit() first.",
      );
    }
    const nFeatures = this.nFeatures_;
    const values = new Float64Array(nSamples * nFeatures);
    const baseValues = new Float64Array(nSamples);

    // Mean leaf value as base value (approximate)
    let leafSum = 0;
    let leafCount = 0;
    for (let node = 0; node < this.leftChild_.length; node++) {
      if (this.leftChild_[node] === -1) {
        leafSum += this.leafValues_[node]!;
        leafCount++;
      }
    }
    const meanLeaf = leafCount > 0 ? leafSum / leafCount : 0;

    for (let i = 0; i < nSamples; i++) {
      baseValues[i] = meanLeaf;
      const contrib = new Float64Array(nFeatures);
      let node = 0;
      let parentVal = meanLeaf;

      while (this.leftChild_[node] !== -1) {
        const feat = this.featureIndex_[node]!;
        const thr = this.threshold_[node]!;
        const xFeat = X[i * nFeatures + feat]!;
        const goLeft = xFeat <= thr;
        const nextNode = goLeft
          ? this.leftChild_[node]!
          : this.rightChild_[node]!;
        const nextVal = this.leafValues_[nextNode]!;
        contrib[feat]! += nextVal - parentVal;
        parentVal = nextVal;
        node = nextNode;
      }
      for (let j = 0; j < nFeatures; j++)
        values[i * nFeatures + j] = contrib[j]!;
    }
    return { values, baseValues, nSamples, nFeatures };
  }
}

/**
 * Summarises a SHAPExplanation into mean absolute SHAP values per feature.
 * Useful for global feature importance ranking.
 */
export function meanAbsShap(explanation: SHAPExplanation): Float64Array {
  const { values, nSamples, nFeatures } = explanation;
  const out = new Float64Array(nFeatures);
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++)
      out[j]! += Math.abs(values[i * nFeatures + j]!);
  }
  for (let j = 0; j < nFeatures; j++) out[j]! /= nSamples;
  return out;
}
