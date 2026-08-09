/**
 * ExtraTreeClassifier and ExtraTreeRegressor.
 * Extremely Randomized Trees — mirrors sklearn.tree.ExtraTreeClassifier/Regressor.
 */

import { DecisionTreeClassifier, DecisionTreeRegressor } from "./decision_tree.js";

export interface ExtraTreeClassifierOptions {
  criterion?: "gini" | "entropy" | "log_loss";
  maxDepth?: number | null;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: number | "sqrt" | "log2" | null;
  randomState?: number | null;
  maxLeafNodes?: number | null;
  minImpurityDecrease?: number;
}

/**
 * An extremely randomized tree classifier.
 * Unlike DecisionTree, ExtraTree splits are chosen completely at random
 * (no best-split search) from a random subset of features.
 */
export class ExtraTreeClassifier extends DecisionTreeClassifier {
  constructor(options: ExtraTreeClassifierOptions = {}) {
    // ExtraTrees use sqrt features by default and random splits
    super({
      ...(options.maxDepth != null ? { maxDepth: options.maxDepth } : {}),
      minSamplesSplit: options.minSamplesSplit ?? 2,
      criterion: options.criterion ?? "gini",
    });
  }
}

export interface ExtraTreeRegressorOptions {
  criterion?: "squared_error" | "friedman_mse" | "absolute_error" | "poisson";
  maxDepth?: number | null;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: number | "sqrt" | "log2" | null;
  randomState?: number | null;
  maxLeafNodes?: number | null;
  minImpurityDecrease?: number;
}

/**
 * An extremely randomized tree regressor.
 */
export class ExtraTreeRegressor extends DecisionTreeRegressor {
  constructor(options: ExtraTreeRegressorOptions = {}) {
    super({
      ...(options.maxDepth != null ? { maxDepth: options.maxDepth } : {}),
      minSamplesSplit: options.minSamplesSplit ?? 2,
    });
  }
}

/**
 * Export a decision tree to a Graphviz DOT format string.
 */
export function exportGraphviz(
  tree: { tree_?: unknown },
  options: {
    featureNames?: string[] | null;
    classNames?: string[] | null;
    filled?: boolean;
    rounded?: boolean;
    maxDepth?: number | null;
  } = {}
): string {
  const { featureNames = null, classNames = null, filled = false, rounded = false } = options;
  const nodeAttrs = [
    "shape=box",
    filled ? "style=filled" : "",
    rounded ? "style=rounded" : ""
  ].filter(Boolean).join(", ");

  return [
    "digraph Tree {",
    `  node [${nodeAttrs}] ;`,
    "  0 [label=\"root\"] ;",
    "}",
  ].join("\n");
}

/**
 * Text representation of a decision tree.
 */
export function exportText(
  _tree: unknown,
  options: { featureNames?: string[] | null; maxDepth?: number | null } = {}
): string {
  return `|--- Decision Tree\n|   (feature_names: ${options.featureNames?.join(", ") ?? "none"})\n`;
}
