/**
 * Feature selection extensions: VarianceThreshold, SelectFromModelExt.
 * Port of sklearn.feature_selection extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Remove features with low variance. */
export class VarianceThresholdFull {
	private variances_: Float64Array | null = null;
	private supportMask_: boolean[] | null = null;
	readonly threshold: number;

	constructor(options: { threshold?: number } = {}) {
		this.threshold = options.threshold ?? 0.0;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const nSamples = X.length;
		const means = new Float64Array(nFeatures);
		for (const row of X) {
			for (let j = 0; j < nFeatures; j++) means[j]! += row[j] ?? 0;
		}
		for (let j = 0; j < nFeatures; j++) means[j]! /= nSamples;
		const variances = new Float64Array(nFeatures);
		for (const row of X) {
			for (let j = 0; j < nFeatures; j++) {
				const d = (row[j] ?? 0) - (means[j] ?? 0);
				variances[j]! += d * d;
			}
		}
		for (let j = 0; j < nFeatures; j++) variances[j]! /= nSamples;
		this.variances_ = variances;
		this.supportMask_ = Array.from({ length: nFeatures }, (_, j) => (variances[j] ?? 0) > this.threshold);
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.supportMask_ === null) throw new NotFittedError("VarianceThresholdFull is not fitted.");
		const selectedCols = this.supportMask_
			.map((v, i) => (v ? i : -1))
			.filter((i) => i >= 0);
		return X.map((row) => {
			const out = new Float64Array(selectedCols.length);
			for (let k = 0; k < selectedCols.length; k++) out[k] = row[selectedCols[k]!] ?? 0;
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	getSupport(): boolean[] {
		if (this.supportMask_ === null) throw new NotFittedError("VarianceThresholdFull is not fitted.");
		return this.supportMask_;
	}

	get variances(): Float64Array {
		if (this.variances_ === null) throw new NotFittedError("VarianceThresholdFull is not fitted.");
		return this.variances_;
	}
}

/** Select features based on mutual information with target. */
export class SelectKBestMutualInfo {
	private scores_: Float64Array | null = null;
	private selectedIndices_: Int32Array | null = null;
	readonly k: number;

	constructor(options: { k?: number } = {}) {
		this.k = options.k ?? 10;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const nFeatures = X[0]?.length ?? 0;
		const scores = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) {
			scores[j] = estimateMutualInfoDiscrete(
				new Float64Array(X.map((row) => row[j] ?? 0)),
				y,
			);
		}
		this.scores_ = scores;
		const k = Math.min(this.k, nFeatures);
		const order = Array.from({ length: nFeatures }, (_, i) => i).sort(
			(a, b) => (scores[b] ?? 0) - (scores[a] ?? 0),
		);
		this.selectedIndices_ = new Int32Array(order.slice(0, k).sort((a, b) => a - b));
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.selectedIndices_ === null) throw new NotFittedError("SelectKBestMutualInfo is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(this.selectedIndices_!.length);
			for (let k = 0; k < this.selectedIndices_!.length; k++) {
				out[k] = row[this.selectedIndices_![k]!] ?? 0;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}

	get scores(): Float64Array {
		if (this.scores_ === null) throw new NotFittedError("SelectKBestMutualInfo is not fitted.");
		return this.scores_;
	}
}

/** Estimate mutual information between continuous feature and discrete target (discretize the feature first). */
function estimateMutualInfoDiscrete(x: Float64Array, y: Int32Array): number {
	const n = x.length;
	// Discretize x into 5 bins
	const sorted = Float64Array.from(x).sort();
	const nBins = 5;
	const binEdges: number[] = [];
	for (let k = 1; k < nBins; k++) {
		binEdges.push(sorted[Math.floor((k * n) / nBins)] ?? 0);
	}
	const xBins = new Int32Array(n);
	for (let i = 0; i < n; i++) {
		let bin = 0;
		for (const edge of binEdges) {
			if ((x[i] ?? 0) >= edge) bin++;
			else break;
		}
		xBins[i] = bin;
	}
	const classes = [...new Set([...y])];
	const xClasses = [...new Set([...xBins])];
	let mi = 0;
	for (const cx of xClasses) {
		for (const cy of classes) {
			let pxy = 0;
			let px = 0;
			let py = 0;
			for (let i = 0; i < n; i++) {
				if ((xBins[i] ?? 0) === cx) px++;
				if ((y[i] ?? 0) === cy) py++;
				if ((xBins[i] ?? 0) === cx && (y[i] ?? 0) === cy) pxy++;
			}
			pxy /= n;
			px /= n;
			py /= n;
			if (pxy > 0 && px > 0 && py > 0) {
				mi += pxy * Math.log(pxy / (px * py));
			}
		}
	}
	return Math.max(0, mi);
}

/** Select features with false discovery rate (Benjamini-Hochberg). */
export function selectFdrFeatures(
	pValues: Float64Array,
	alpha = 0.05,
): boolean[] {
	const n = pValues.length;
	const sorted = Array.from({ length: n }, (_, i) => i).sort(
		(a, b) => (pValues[a] ?? 0) - (pValues[b] ?? 0),
	);
	const selected = new Array(n).fill(false) as boolean[];
	let maxK = -1;
	for (let k = 0; k < n; k++) {
		const threshold = ((k + 1) * alpha) / n;
		if ((pValues[sorted[k]!] ?? 0) <= threshold) maxK = k;
	}
	if (maxK >= 0) {
		for (let k = 0; k <= maxK; k++) selected[sorted[k]!] = true;
	}
	return selected;
}
