/**
 * Mixture model extensions: VariationalGaussianMixture, Bayesian extension.
 * Port of sklearn.mixture extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Dirichlet Process Gaussian Mixture Model (simplified variational approximation). */
export class DPGMMSimple {
	private means_: Float64Array[] | null = null;
	private weights_: Float64Array | null = null;
	private labels_: Int32Array | null = null;
	readonly maxComponents: number;
	readonly nIter: number;
	readonly randomState: number;

	constructor(
		options: {
			maxComponents?: number;
			nIter?: number;
			randomState?: number;
		} = {},
	) {
		this.maxComponents = options.maxComponents ?? 10;
		this.nIter = options.nIter ?? 100;
		this.randomState = options.randomState ?? 0;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const p = X[0]?.length ?? 0;
		const K = Math.min(this.maxComponents, n);
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		// Initialize responsibilities randomly
		const resp = Array.from({ length: n }, () => {
			const r = new Float64Array(K).map(() => rand());
			let s = 0;
			for (let k = 0; k < K; k++) s += r[k] ?? 0;
			for (let k = 0; k < K; k++) r[k]! /= s;
			return r;
		});
		const means: Float64Array[] = Array.from({ length: K }, () => new Float64Array(p));
		const weights = new Float64Array(K);

		for (let iter = 0; iter < this.nIter; iter++) {
			// M-step
			for (let k = 0; k < K; k++) weights[k] = 0;
			for (let k = 0; k < K; k++) means[k] = new Float64Array(p);
			for (let i = 0; i < n; i++) {
				for (let k = 0; k < K; k++) {
					const r = resp[i]?.[k] ?? 0;
					weights[k]! += r;
					for (let j = 0; j < p; j++) means[k]![j]! += r * (X[i]?.[j] ?? 0);
				}
			}
			for (let k = 0; k < K; k++) {
				const w = weights[k] ?? 0;
				if (w > 0) {
					for (let j = 0; j < p; j++) means[k]![j]! /= w;
				}
			}
			const totalW = weights.reduce((s, v) => s + v, 0);
			for (let k = 0; k < K; k++) weights[k]! /= totalW;

			// E-step
			for (let i = 0; i < n; i++) {
				const row = resp[i]!;
				let s = 0;
				for (let k = 0; k < K; k++) {
					let sq = 0;
					for (let j = 0; j < p; j++) {
						const diff = (X[i]?.[j] ?? 0) - (means[k]![j] ?? 0);
						sq += diff * diff;
					}
					row[k] = (weights[k] ?? 0) * Math.exp(-0.5 * sq);
					s += row[k] ?? 0;
				}
				for (let k = 0; k < K; k++) row[k]! = s === 0 ? 1 / K : (row[k] ?? 0) / s;
			}
		}
		this.means_ = means;
		this.weights_ = weights;
		this.labels_ = new Int32Array(n).map((_, i) => {
			let best = 0;
			let bestR = 0;
			const row = resp[i]!;
			for (let k = 0; k < K; k++) {
				if ((row[k] ?? 0) > bestR) {
					bestR = row[k] ?? 0;
					best = k;
				}
			}
			return best;
		});
		return this;
	}

	predict(X: Float64Array[]): Int32Array {
		if (this.means_ === null || this.weights_ === null) throw new NotFittedError("DPGMMSimple is not fitted.");
		const K = this.means_.length;
		return new Int32Array(
			X.map((xi) => {
				let best = 0;
				let bestScore = Number.NEGATIVE_INFINITY;
				for (let k = 0; k < K; k++) {
					let sq = 0;
					for (let j = 0; j < xi.length; j++) {
						const diff = (xi[j] ?? 0) - (this.means_![k]?.[j] ?? 0);
						sq += diff * diff;
					}
					const score = Math.log(this.weights_![k] ?? 1e-10) - 0.5 * sq;
					if (score > bestScore) {
						bestScore = score;
						best = k;
					}
				}
				return best;
			}),
		);
	}

	get labels(): Int32Array {
		if (this.labels_ === null) throw new NotFittedError("DPGMMSimple is not fitted.");
		return this.labels_;
	}

	get weights(): Float64Array {
		if (this.weights_ === null) throw new NotFittedError("DPGMMSimple is not fitted.");
		return this.weights_;
	}

	get means(): Float64Array[] {
		if (this.means_ === null) throw new NotFittedError("DPGMMSimple is not fitted.");
		return this.means_;
	}
}
