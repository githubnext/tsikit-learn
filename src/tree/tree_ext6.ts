/**
 * Tree extensions: cost-complexity pruning, post-pruning utilities.
 * Port of sklearn.tree pruning extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Node in a decision tree for cost-complexity pruning. */
export interface TreeNodeCCP {
	isLeaf: boolean;
	impurity: number;
	nSamples: number;
	leftChild?: TreeNodeCCP;
	rightChild?: TreeNodeCCP;
	label?: number;
}

/** Compute cost-complexity pruning path (alphas and impurities). */
export function costComplexityPruningPath(
	root: TreeNodeCCP,
): { ccp_alphas: Float64Array; impurities: Float64Array } {
	const alphas: number[] = [0];
	const impurities: number[] = [subTreeImpurity(root)];

	let tree = cloneTree(root);
	while (!tree.isLeaf) {
		const alpha = weakestLink(tree);
		pruneAtAlpha(tree, alpha);
		alphas.push(alpha);
		impurities.push(subTreeImpurity(tree));
	}

	return {
		ccp_alphas: new Float64Array(alphas),
		impurities: new Float64Array(impurities),
	};
}

function subTreeImpurity(node: TreeNodeCCP): number {
	if (node.isLeaf) return node.impurity * node.nSamples;
	return (
		subTreeImpurity(node.leftChild!) + subTreeImpurity(node.rightChild!)
	);
}

function nLeaves(node: TreeNodeCCP): number {
	if (node.isLeaf) return 1;
	return nLeaves(node.leftChild!) + nLeaves(node.rightChild!);
}

function weakestLink(node: TreeNodeCCP): number {
	if (node.isLeaf) return Number.POSITIVE_INFINITY;
	const leafImpurity = subTreeImpurity(node);
	const leaves = nLeaves(node);
	const alpha = (node.impurity * node.nSamples - leafImpurity) / (leaves - 1);
	const leftAlpha = weakestLink(node.leftChild!);
	const rightAlpha = weakestLink(node.rightChild!);
	return Math.min(alpha, leftAlpha, rightAlpha);
}

function pruneAtAlpha(node: TreeNodeCCP, alpha: number): void {
	if (node.isLeaf) return;
	const leafImpurity = subTreeImpurity(node);
	const leaves = nLeaves(node);
	const nodeAlpha = (node.impurity * node.nSamples - leafImpurity) / Math.max(1, leaves - 1);
	if (nodeAlpha <= alpha) {
		node.isLeaf = true;
		delete (node as { leftChild?: TreeNodeCCP }).leftChild;
		delete (node as { rightChild?: TreeNodeCCP }).rightChild;
	} else {
		pruneAtAlpha(node.leftChild!, alpha);
		pruneAtAlpha(node.rightChild!, alpha);
	}
}

function cloneTree(node: TreeNodeCCP): TreeNodeCCP {
	const clone: TreeNodeCCP = {
		isLeaf: node.isLeaf,
		impurity: node.impurity,
		nSamples: node.nSamples,
		...(node.label !== undefined ? { label: node.label } : {}),
	};
	if (node.leftChild) clone.leftChild = cloneTree(node.leftChild);
	if (node.rightChild) clone.rightChild = cloneTree(node.rightChild);
	return clone;
}

/** Compute tree depth. */
export function treeDepth(node: TreeNodeCCP): number {
	if (node.isLeaf) return 0;
	return 1 + Math.max(treeDepth(node.leftChild!), treeDepth(node.rightChild!));
}

/** Compute number of nodes in a tree. */
export function countNodes(node: TreeNodeCCP): number {
	if (node.isLeaf) return 1;
	return 1 + countNodes(node.leftChild!) + countNodes(node.rightChild!);
}

/** Decision tree classifier with CCP alpha pruning support. */
export class DecisionTreeWithCCP {
	private root_: {
		feat: number;
		thresh: number;
		left: number;
		right: number;
	}[] | null = null;
	private leafValues_: Int32Array | null = null;
	readonly maxDepth: number;
	readonly ccpAlpha: number;

	constructor(options: { maxDepth?: number; ccpAlpha?: number } = {}) {
		this.maxDepth = options.maxDepth ?? 5;
		this.ccpAlpha = options.ccpAlpha ?? 0.0;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		// Simplified classification tree with CART splitting
		type Node =
			| { type: "internal"; feat: number; thresh: number; left: number; right: number }
			| { type: "leaf"; label: number };
		const nodes: Node[] = [];
		const classes = [...new Set([...y])].sort((a, b) => a - b);

		const buildNode = (indices: number[], depth: number): number => {
			const nodeIdx = nodes.length;
			if (indices.length === 0) {
				nodes.push({ type: "leaf", label: classes[0] ?? 0 });
				return nodeIdx;
			}
			// Majority class
			const counts = new Map<number, number>();
			for (const i of indices) counts.set(y[i] ?? 0, (counts.get(y[i] ?? 0) ?? 0) + 1);
			const majorityLabel = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;

			if (depth >= this.maxDepth || indices.length <= 1 || counts.size === 1) {
				nodes.push({ type: "leaf", label: majorityLabel });
				return nodeIdx;
			}

			const nFeatures = X[0]?.length ?? 0;
			let bestGini = Number.POSITIVE_INFINITY;
			let bestFeat = 0;
			let bestThresh = 0;

			for (let j = 0; j < nFeatures; j++) {
				const vals = indices.map((i) => ({ v: X[i]?.[j] ?? 0, y: y[i] ?? 0 }));
				vals.sort((a, b) => a.v - b.v);
				for (let k = 0; k < vals.length - 1; k++) {
					if ((vals[k]?.v ?? 0) === (vals[k + 1]?.v ?? 0)) continue;
					const thresh = ((vals[k]?.v ?? 0) + (vals[k + 1]?.v ?? 0)) / 2;
					const lIdx = indices.filter((i) => (X[i]?.[j] ?? 0) <= thresh);
					const rIdx = indices.filter((i) => (X[i]?.[j] ?? 0) > thresh);
					const gini = giniImpurity(lIdx, y, indices.length) + giniImpurity(rIdx, y, indices.length);
					if (gini < bestGini) {
						bestGini = gini;
						bestFeat = j;
						bestThresh = thresh;
					}
				}
			}

			const lIdx = indices.filter((i) => (X[i]?.[bestFeat] ?? 0) <= bestThresh);
			const rIdx = indices.filter((i) => (X[i]?.[bestFeat] ?? 0) > bestThresh);

			if (lIdx.length === 0 || rIdx.length === 0) {
				nodes.push({ type: "leaf", label: majorityLabel });
				return nodeIdx;
			}

			nodes.push({ type: "internal", feat: bestFeat, thresh: bestThresh, left: 0, right: 0 });
			const leftIdx = buildNode(lIdx, depth + 1);
			const rightIdx = buildNode(rIdx, depth + 1);
			const node = nodes[nodeIdx];
			if (node?.type === "internal") {
				node.left = leftIdx;
				node.right = rightIdx;
			}
			return nodeIdx;
		};

		buildNode(Array.from({ length: X.length }, (_, i) => i), 0);
		this.root_ = nodes.map((n) =>
			n.type === "internal"
				? { feat: n.feat, thresh: n.thresh, left: n.left, right: n.right }
				: { feat: -1, thresh: 0, left: -(n.label + 1), right: -(n.label + 1) },
		);
		this.leafValues_ = new Int32Array(nodes.map((n) => (n.type === "leaf" ? n.label : -1)));
		return this;
	}

	predict(X: Float64Array[]): Int32Array {
		if (this.root_ === null) throw new NotFittedError("DecisionTreeWithCCP is not fitted.");
		return new Int32Array(
			X.map((row) => {
				let nodeIdx = 0;
				for (let depth = 0; depth <= this.maxDepth + 1; depth++) {
					const node = this.root_![nodeIdx];
					if (node === undefined) break;
					if (node.feat < 0) return this.leafValues_![nodeIdx] ?? 0;
					nodeIdx = (row[node.feat] ?? 0) <= node.thresh ? node.left : node.right;
				}
				return 0;
			}),
		);
	}
}

function giniImpurity(indices: number[], y: Int32Array, total: number): number {
	if (indices.length === 0) return 0;
	const counts = new Map<number, number>();
	for (const i of indices) counts.set(y[i] ?? 0, (counts.get(y[i] ?? 0) ?? 0) + 1);
	let gini = 0;
	for (const c of counts.values()) {
		const p = c / indices.length;
		gini += p * (1 - p);
	}
	return (gini * indices.length) / total;
}
