/**
 * Isolation Forest extension: anomaly score analysis.
 * Port of sklearn.ensemble._iforest extensions
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Extended Isolation Forest with feature importance tracking.
 * Complements the main IForest class.
 */
export class IsolationForestExt {
	nEstimators: number;
	maxSamples: number | "auto";
	contamination: number | "auto";
	maxFeatures: number;
	randomState: number | null;

	estimators_?: IsoTree[];
	maxSamples_?: number;
	nFeatures_?: number;
	featureImportances_?: Float64Array;
	offset_?: number;

	constructor(params: {
		nEstimators?: number;
		maxSamples?: number | "auto";
		contamination?: number | "auto";
		maxFeatures?: number;
		randomState?: number | null;
	} = {}) {
		this.nEstimators = params.nEstimators ?? 100;
		this.maxSamples = params.maxSamples ?? "auto";
		this.contamination = params.contamination ?? "auto";
		this.maxFeatures = params.maxFeatures ?? 1.0;
		this.randomState = params.randomState ?? null;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const d = X[0]?.length ?? 0;
		this.nFeatures_ = d;
		this.maxSamples_ = this.maxSamples === "auto" ? Math.min(256, n) : this.maxSamples;

		let seed = this.randomState ?? 42;
		const rand = (): number => {
			seed = (seed * 1664525 + 1013904223) & 0xffffffff;
			return (seed >>> 0) / 0x100000000;
		};

		this.estimators_ = [];
		for (let t = 0; t < this.nEstimators; t++) {
			// Subsample
			const indices = Array.from({ length: this.maxSamples_ }, () => Math.floor(rand() * n));
			const Xsub = indices.map((i) => X[i]!);
			const tree = new IsoTree();
			tree.build(Xsub, Math.ceil(Math.log2(this.maxSamples_)), rand);
			this.estimators_.push(tree);
		}

		// Compute offset for scoring
		const scores = this._rawScores(X);
		if (this.contamination === "auto") {
			this.offset_ = -0.5;
		} else {
			const sorted = Float64Array.from(scores).sort();
			const idx = Math.floor((1 - this.contamination) * n);
			this.offset_ = sorted[Math.min(idx, n - 1)]!;
		}

		// Compute feature importances from split features
		this.featureImportances_ = new Float64Array(d);
		for (const tree of this.estimators_) {
			for (const feature of tree.splitFeatures) {
				this.featureImportances_![feature]++;
			}
		}
		const totalSplits = this.featureImportances_.reduce((s, v) => s + v, 0);
		if (totalSplits > 0) {
			for (let j = 0; j < d; j++) this.featureImportances_[j]! /= totalSplits;
		}

		return this;
	}

	private _rawScores(X: Float64Array[]): Float64Array {
		const n = X.length;
		const scores = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			let avgPathLength = 0;
			for (const tree of this.estimators_!) {
				avgPathLength += tree.pathLength(X[i]!);
			}
			avgPathLength /= this.estimators_!.length;
			const c = avgPathLen(this.maxSamples_!);
			scores[i] = -(2 ** (-avgPathLength / c));
		}
		return scores;
	}

	predict(X: Float64Array[]): Int32Array {
		if (!this.estimators_) throw new NotFittedError("IsolationForestExt");
		const scores = this._rawScores(X);
		return new Int32Array(scores.map((s) => s < this.offset_! ? -1 : 1));
	}

	decisionFunction(X: Float64Array[]): Float64Array {
		if (!this.estimators_) throw new NotFittedError("IsolationForestExt");
		const scores = this._rawScores(X);
		return new Float64Array(scores.map((s) => s - this.offset_!));
	}

	scoresSamples(X: Float64Array[]): Float64Array {
		if (!this.estimators_) throw new NotFittedError("IsolationForestExt");
		return this._rawScores(X);
	}
}

/** Average path length for a BST with n nodes */
function avgPathLen(n: number): number {
	if (n <= 1) return 1;
	const h = (n: number) => Math.log(n) + 0.5772156649;
	return 2 * h(n - 1) - 2 * (n - 1) / n;
}

interface IsoNode {
	featureIdx: number;
	threshold: number;
	left: IsoNode | null;
	right: IsoNode | null;
	size: number;
}

class IsoTree {
	private root: IsoNode | null = null;
	splitFeatures: number[] = [];

	build(X: Float64Array[], maxDepth: number, rand: () => number): void {
		this.splitFeatures = [];
		this.root = this._buildNode(X, 0, maxDepth, rand);
	}

	private _buildNode(X: Float64Array[], depth: number, maxDepth: number, rand: () => number): IsoNode {
		const n = X.length;
		const d = X[0]?.length ?? 0;

		if (n <= 1 || depth >= maxDepth) {
			return { featureIdx: 0, threshold: 0, left: null, right: null, size: n };
		}

		const featureIdx = Math.floor(rand() * d);
		this.splitFeatures.push(featureIdx);

		let minVal = Number.POSITIVE_INFINITY;
		let maxVal = Number.NEGATIVE_INFINITY;
		for (const x of X) {
			const v = x[featureIdx] ?? 0;
			if (v < minVal) minVal = v;
			if (v > maxVal) maxVal = v;
		}

		if (minVal === maxVal) {
			return { featureIdx, threshold: minVal, left: null, right: null, size: n };
		}

		const threshold = minVal + rand() * (maxVal - minVal);
		const left = X.filter((x) => (x[featureIdx] ?? 0) < threshold);
		const right = X.filter((x) => (x[featureIdx] ?? 0) >= threshold);

		return {
			featureIdx,
			threshold,
			left: this._buildNode(left, depth + 1, maxDepth, rand),
			right: this._buildNode(right, depth + 1, maxDepth, rand),
			size: n,
		};
	}

	pathLength(x: Float64Array): number {
		let node = this.root;
		let depth = 0;
		while (node !== null && node.left !== null && node.right !== null) {
			if ((x[node.featureIdx] ?? 0) < node.threshold) {
				node = node.left;
			} else {
				node = node.right;
			}
			depth++;
		}
		const size = node?.size ?? 1;
		return depth + avgPathLen(size);
	}
}
