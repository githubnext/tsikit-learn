/**
 * Ensemble extensions: ExtraTreesRegressor, gradient boosting extensions.
 * Port of sklearn.ensemble extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Gradient boosting residual step estimator (weak learner for regression). */
export class GradientBoostingResidualFitter {
	private leafValues_: Float64Array | null = null;
	private splitFeature_ = 0;
	private splitThreshold_ = 0;
	readonly maxDepth: number;
	readonly learningRate: number;

	constructor(options: { maxDepth?: number; learningRate?: number } = {}) {
		this.maxDepth = options.maxDepth ?? 3;
		this.learningRate = options.learningRate ?? 0.1;
	}

	fit(X: Float64Array[], residuals: Float64Array): this {
		// Simple stump: find best feature/threshold split
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		let bestMse = Number.POSITIVE_INFINITY;
		let bestFeat = 0;
		let bestThresh = 0;
		for (let j = 0; j < nFeatures; j++) {
			const vals = new Float64Array(n);
			for (let i = 0; i < n; i++) vals[i] = X[i]?.[j] ?? 0;
			const sorted = Float64Array.from(vals).sort();
			for (let k = 0; k < sorted.length - 1; k++) {
				const t = ((sorted[k] ?? 0) + (sorted[k + 1] ?? 0)) / 2;
				let l = 0;
				let lSum = 0;
				let rSum = 0;
				let r = 0;
				for (let i = 0; i < n; i++) {
					if ((X[i]?.[j] ?? 0) <= t) {
						l++;
						lSum += residuals[i] ?? 0;
					} else {
						r++;
						rSum += residuals[i] ?? 0;
					}
				}
				const lMean = l === 0 ? 0 : lSum / l;
				const rMean = r === 0 ? 0 : rSum / r;
				let mse = 0;
				for (let i = 0; i < n; i++) {
					const pred = (X[i]?.[j] ?? 0) <= t ? lMean : rMean;
					const d = (residuals[i] ?? 0) - pred;
					mse += d * d;
				}
				if (mse < bestMse) {
					bestMse = mse;
					bestFeat = j;
					bestThresh = t;
				}
			}
		}
		this.splitFeature_ = bestFeat;
		this.splitThreshold_ = bestThresh;
		let lSum = 0;
		let l = 0;
		let rSum = 0;
		let r = 0;
		for (let i = 0; i < n; i++) {
			if ((X[i]?.[bestFeat] ?? 0) <= bestThresh) {
				l++;
				lSum += residuals[i] ?? 0;
			} else {
				r++;
				rSum += residuals[i] ?? 0;
			}
		}
		this.leafValues_ = new Float64Array([
			l === 0 ? 0 : lSum / l,
			r === 0 ? 0 : rSum / r,
		]);
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.leafValues_ === null) throw new NotFittedError("GradientBoostingResidualFitter is not fitted.");
		return new Float64Array(
			X.map((row) =>
				(row[this.splitFeature_] ?? 0) <= this.splitThreshold_
					? (this.leafValues_![0] ?? 0) * this.learningRate
					: (this.leafValues_![1] ?? 0) * this.learningRate,
			),
		);
	}
}

/** ExtraTreesRegressor: ensemble of extremely randomized regression trees. */
export class ExtraTreesRegressorExt {
	private trees_: Array<{ feat: number; thresh: number; lVal: number; rVal: number }> | null = null;
	readonly nEstimators: number;
	readonly maxFeatures: number | "sqrt" | "log2";
	readonly randomState: number;

	constructor(
		options: {
			nEstimators?: number;
			maxFeatures?: number | "sqrt" | "log2";
			randomState?: number;
		} = {},
	) {
		this.nEstimators = options.nEstimators ?? 100;
		this.maxFeatures = options.maxFeatures ?? "sqrt";
		this.randomState = options.randomState ?? 0;
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		let mf: number;
		if (this.maxFeatures === "sqrt") mf = Math.max(1, Math.round(Math.sqrt(nFeatures)));
		else if (this.maxFeatures === "log2") mf = Math.max(1, Math.round(Math.log2(nFeatures)));
		else mf = this.maxFeatures as number;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		this.trees_ = Array.from({ length: this.nEstimators }, () => {
			// Random feature subset
			const featIdx = Array.from({ length: nFeatures }, (_, i) => i)
				.sort(() => rand() - 0.5)
				.slice(0, mf);
			// Bootstrap sample
			const sampleIdx = Array.from({ length: n }, () => Math.floor(rand() * n));
			// Random split
			const feat = featIdx[Math.floor(rand() * mf)] ?? 0;
			let minV = Number.POSITIVE_INFINITY;
			let maxV = Number.NEGATIVE_INFINITY;
			for (const si of sampleIdx) {
				const v = X[si]?.[feat] ?? 0;
				if (v < minV) minV = v;
				if (v > maxV) maxV = v;
			}
			const thresh = minV + rand() * (maxV - minV);
			let lSum = 0;
			let l = 0;
			let rSum = 0;
			let r = 0;
			for (const si of sampleIdx) {
				if ((X[si]?.[feat] ?? 0) <= thresh) {
					l++;
					lSum += y[si] ?? 0;
				} else {
					r++;
					rSum += y[si] ?? 0;
				}
			}
			return { feat, thresh, lVal: l === 0 ? 0 : lSum / l, rVal: r === 0 ? 0 : rSum / r };
		});
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.trees_ === null) throw new NotFittedError("ExtraTreesRegressorExt is not fitted.");
		return new Float64Array(
			X.map((row) => {
				let sum = 0;
				for (const tree of this.trees_!) {
					sum += (row[tree.feat] ?? 0) <= tree.thresh ? tree.lVal : tree.rVal;
				}
				return sum / (this.trees_?.length ?? 1);
			}),
		);
	}
}
