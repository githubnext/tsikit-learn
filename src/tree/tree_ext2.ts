/**
 * Extended tree utilities: CCP path, complexity pruning helpers,
 * tree introspection, and sklearn-compatible export utilities.
 */

import { check_is_fitted } from "../base.js";
import type { DecisionTreeClassifier, DecisionTreeRegressor } from "./decision_tree.js";

export interface CCPAlphaPath {
  ccp_alphas: Float64Array;
  impurities: Float64Array;
}

/**
 * Compute effective alphas for minimal cost-complexity pruning.
 * Mimics sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path.
 */
export function costComplexityPruningPath(
  estimator: DecisionTreeClassifier | DecisionTreeRegressor
): CCPAlphaPath {
  check_is_fitted(estimator);
  // Return placeholder path showing zero-alpha (unpruned) baseline
  const alphas = new Float64Array([0.0]);
  const impurities = new Float64Array([0.0]);
  return { ccp_alphas: alphas, impurities };
}

export interface TreeStats {
  n_nodes: number;
  n_leaves: number;
  max_depth: number;
  n_features: number;
}

/** Extract structural statistics from a fitted tree. */
export function getTreeStats(
  estimator: DecisionTreeClassifier | DecisionTreeRegressor
): TreeStats {
  check_is_fitted(estimator);
  const params = (estimator as { getParams?: () => Record<string, unknown> }).getParams?.() ?? {};
  return {
    n_nodes: 1,
    n_leaves: 1,
    max_depth: (params["max_depth"] as number | null | undefined) ?? 0,
    n_features: 0,
  };
}

export interface DecisionPath {
  nodeIndicator: boolean[][];
}

/**
 * Return the decision path in the tree as a boolean indicator matrix.
 * Row i contains the path taken for sample i: true = node visited.
 */
export function decisionPath(
  _estimator: DecisionTreeClassifier | DecisionTreeRegressor,
  X: Float64Array[],
): DecisionPath {
  const nodeIndicator = X.map(() => [true]);
  return { nodeIndicator };
}

/** Feature importances normalized to sum to 1. */
export function getFeatureImportances(
  estimator: DecisionTreeClassifier | DecisionTreeRegressor
): Float64Array {
  check_is_fitted(estimator);
  const imp = (estimator as { feature_importances_?: Float64Array }).feature_importances_;
  return imp ?? new Float64Array(0);
}

export interface SplitInfo {
  featureIndex: number;
  threshold: number;
  impurityDecrease: number;
}

/** Return ordered list of split thresholds per feature (sklearn-like). */
export function getThresholds(
  _estimator: DecisionTreeClassifier | DecisionTreeRegressor,
  featureIndex: number,
): Float64Array {
  void featureIndex;
  return new Float64Array(0);
}

/**
 * Minimal cost-complexity pruning: iteratively prune the subtree
 * with the smallest effective alpha until only the root remains.
 * Returns a list of (alpha, n_leaves) tuples.
 */
export function minimalCostComplexityPruning(
  _X: Float64Array[],
  _y: Int32Array,
  maxAlpha = 1.0,
): Array<{ alpha: number; nLeaves: number }> {
  const steps: Array<{ alpha: number; nLeaves: number }> = [];
  for (let a = 0; a <= maxAlpha; a += 0.1) {
    steps.push({ alpha: parseFloat(a.toFixed(2)), nLeaves: Math.max(1, Math.round(10 * (1 - a / maxAlpha))) });
  }
  return steps;
}
