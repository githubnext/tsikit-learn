/**
 * Cost-complexity pruning for decision trees.
 * Mirrors scikit-learn's tree._classes cost_complexity_pruning_path.
 */

export interface PruningPathResult {
  /** Effective alphas (ccp_alphas) at which subtrees change */
  ccpAlphas: Float64Array;
  /** Impurity sums at each pruning step */
  impurities: Float64Array;
}

export interface PruningNode {
  impurity: number;
  nSamples: number;
  left: PruningNode | null;
  right: PruningNode | null;
}

function nodeCount(node: PruningNode): number {
  if (node.left === null && node.right === null) return 1;
  const l = node.left !== null ? nodeCount(node.left) : 0;
  const r = node.right !== null ? nodeCount(node.right) : 0;
  return 1 + l + r;
}

function leafImpuritySum(node: PruningNode): number {
  if (node.left === null && node.right === null) {
    return node.impurity * node.nSamples;
  }
  const l = node.left !== null ? leafImpuritySum(node.left) : 0;
  const r = node.right !== null ? leafImpuritySum(node.right) : 0;
  return l + r;
}

function leafCount(node: PruningNode): number {
  if (node.left === null && node.right === null) return 1;
  const l = node.left !== null ? leafCount(node.left) : 0;
  const r = node.right !== null ? leafCount(node.right) : 0;
  return l + r;
}

/**
 * Compute the cost-complexity pruning path.
 * Returns the effective alpha values and impurity sums at each pruning step.
 */
export function costComplexityPruningPath(
  tree: PruningNode,
): PruningPathResult {
  const alphas: number[] = [];
  const imps: number[] = [];

  const computeAlpha = (node: PruningNode): number => {
    if (node.left === null && node.right === null)
      return Number.POSITIVE_INFINITY;
    const subtreeImp = leafImpuritySum(node);
    const subtreeLeaves = leafCount(node);
    const nodeImp = node.impurity * node.nSamples;
    const alpha = (nodeImp - subtreeImp) / (subtreeLeaves - 1);
    const leftAlpha =
      node.left !== null ? computeAlpha(node.left) : Number.POSITIVE_INFINITY;
    const rightAlpha =
      node.right !== null ? computeAlpha(node.right) : Number.POSITIVE_INFINITY;
    return Math.min(alpha, leftAlpha, rightAlpha);
  };

  const prune = (node: PruningNode, alpha: number): PruningNode => {
    if (node.left === null && node.right === null) return node;
    const nodeImp = node.impurity * node.nSamples;
    const subtreeImp = leafImpuritySum(node);
    const subtreeLeaves = leafCount(node);
    const nodeAlpha = (nodeImp - subtreeImp) / (subtreeLeaves - 1);
    if (nodeAlpha <= alpha) {
      return {
        impurity: node.impurity,
        nSamples: node.nSamples,
        left: null,
        right: null,
      };
    }
    return {
      impurity: node.impurity,
      nSamples: node.nSamples,
      left: node.left !== null ? prune(node.left, alpha) : null,
      right: node.right !== null ? prune(node.right, alpha) : null,
    };
  };

  let current = tree;
  alphas.push(0);
  imps.push(leafImpuritySum(current));

  while (leafCount(current) > 1) {
    const alpha = computeAlpha(current);
    if (!Number.isFinite(alpha)) break;
    current = prune(current, alpha);
    alphas.push(alpha);
    imps.push(leafImpuritySum(current));
  }

  return {
    ccpAlphas: new Float64Array(alphas),
    impurities: new Float64Array(imps),
  };
}

/**
 * Minimal cost-complexity pruning — prune subtrees with alpha <= ccp_alpha.
 */
export function minimalCostComplexityPrune(
  tree: PruningNode,
  ccpAlpha: number,
): PruningNode {
  if (ccpAlpha < 0) throw new RangeError("ccpAlpha must be >= 0");
  if (ccpAlpha === 0) return tree;

  const prune = (node: PruningNode): PruningNode => {
    if (node.left === null && node.right === null) return node;
    const nodeImp = node.impurity * node.nSamples;
    const subtreeImp = leafImpuritySum(node);
    const subtreeLeaves = leafCount(node);
    const alpha = (nodeImp - subtreeImp) / (subtreeLeaves - 1);
    if (alpha <= ccpAlpha) {
      return {
        impurity: node.impurity,
        nSamples: node.nSamples,
        left: null,
        right: null,
      };
    }
    return {
      impurity: node.impurity,
      nSamples: node.nSamples,
      left: node.left !== null ? prune(node.left) : null,
      right: node.right !== null ? prune(node.right) : null,
    };
  };

  return prune(tree);
}
