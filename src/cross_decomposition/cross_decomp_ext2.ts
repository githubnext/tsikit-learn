/**
 * Cross-decomposition extensions: PLSSVD, CCA extensions.
 * Port of sklearn.cross_decomposition extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** PLS Canonical (PLSC): symmetric variant of PLS. */
export class PLSCanonical {
	private xRotations_: Float64Array[] | null = null;
	private yRotations_: Float64Array[] | null = null;
	private xMean_: Float64Array | null = null;
	private yMean_: Float64Array | null = null;
	readonly nComponents: number;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			nComponents?: number;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.nComponents = options.nComponents ?? 2;
		this.maxIter = options.maxIter ?? 500;
		this.tol = options.tol ?? 1e-6;
	}

	fit(X: Float64Array[], Y: Float64Array[]): this {
		const n = X.length;
		const p = X[0]?.length ?? 0;
		const q = Y[0]?.length ?? 0;
		const k = Math.min(this.nComponents, Math.min(p, q));

		const xMean = new Float64Array(p);
		const yMean = new Float64Array(q);
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < p; j++) xMean[j]! += X[i]?.[j] ?? 0;
			for (let j = 0; j < q; j++) yMean[j]! += Y[i]?.[j] ?? 0;
		}
		for (let j = 0; j < p; j++) xMean[j]! /= n;
		for (let j = 0; j < q; j++) yMean[j]! /= n;
		this.xMean_ = xMean;
		this.yMean_ = yMean;

		const Xc = X.map((row) => new Float64Array(p).map((_, j) => (row[j] ?? 0) - (xMean[j] ?? 0)));
		const Yc = Y.map((row) => new Float64Array(q).map((_, j) => (row[j] ?? 0) - (yMean[j] ?? 0)));

		const xRotations: Float64Array[] = [];
		const yRotations: Float64Array[] = [];

		let XResid = Xc.map((r) => new Float64Array(r));
		let YResid = Yc.map((r) => new Float64Array(r));

		for (let comp = 0; comp < k; comp++) {
			// Compute X^T * Y covariance
			const Cxy = Array.from({ length: p }, (_, a) =>
				new Float64Array(q).map((_, b) => {
					let s = 0;
					for (let i = 0; i < n; i++) s += (XResid[i]?.[a] ?? 0) * (YResid[i]?.[b] ?? 0);
					return s;
				}),
			);
			// Power iteration for first SVD component
			let u = new Float64Array(p);
			u[0] = 1;
			let v = new Float64Array(q);
			for (let iter = 0; iter < this.maxIter; iter++) {
				// u = Cxy * v
				const newU = new Float64Array(p);
				for (let a = 0; a < p; a++) {
					for (let b = 0; b < q; b++) newU[a]! += (Cxy[a]?.[b] ?? 0) * (v[b] ?? 0);
				}
				let norm = 0;
				for (let a = 0; a < p; a++) norm += (newU[a] ?? 0) ** 2;
				norm = Math.sqrt(norm) || 1;
				for (let a = 0; a < p; a++) newU[a]! /= norm;
				// v = Cxy^T * u
				const newV = new Float64Array(q);
				for (let b = 0; b < q; b++) {
					for (let a = 0; a < p; a++) newV[b]! += (Cxy[a]?.[b] ?? 0) * (newU[a] ?? 0);
				}
				let normV = 0;
				for (let b = 0; b < q; b++) normV += (newV[b] ?? 0) ** 2;
				normV = Math.sqrt(normV) || 1;
				for (let b = 0; b < q; b++) newV[b]! /= normV;
				let diff = 0;
				for (let a = 0; a < p; a++) diff += ((newU[a] ?? 0) - (u[a] ?? 0)) ** 2;
				u = newU;
				v = newV;
				if (diff < this.tol) break;
			}
			xRotations.push(u);
			yRotations.push(v);
			// Deflate
			const xt = new Float64Array(n).map((_, i) => {
				let s = 0;
				for (let a = 0; a < p; a++) s += (XResid[i]?.[a] ?? 0) * (u[a] ?? 0);
				return s;
			});
			for (let i = 0; i < n; i++) {
				for (let a = 0; a < p; a++) XResid[i]![a]! -= (xt[i] ?? 0) * (u[a] ?? 0);
			}
			const yt = new Float64Array(n).map((_, i) => {
				let s = 0;
				for (let b = 0; b < q; b++) s += (YResid[i]?.[b] ?? 0) * (v[b] ?? 0);
				return s;
			});
			for (let i = 0; i < n; i++) {
				for (let b = 0; b < q; b++) YResid[i]![b]! -= (yt[i] ?? 0) * (v[b] ?? 0);
			}
		}
		this.xRotations_ = xRotations;
		this.yRotations_ = yRotations;
		return this;
	}

	transform(X: Float64Array[], Y?: Float64Array[]): { xScores: Float64Array[]; yScores?: Float64Array[] } {
		if (this.xRotations_ === null || this.xMean_ === null) throw new NotFittedError("PLSCanonical is not fitted.");
		const k = this.xRotations_.length;
		const xScores = X.map((row) => {
			const scores = new Float64Array(k);
			for (let c = 0; c < k; c++) {
				for (let j = 0; j < row.length; j++) {
					scores[c] += ((row[j] ?? 0) - (this.xMean_![j] ?? 0)) * (this.xRotations_![c]?.[j] ?? 0);
				}
			}
			return scores;
		});
		if (!Y || !this.yRotations_ || !this.yMean_) return { xScores };
		const yScores = Y.map((row) => {
			const scores = new Float64Array(k);
			for (let c = 0; c < k; c++) {
				for (let j = 0; j < row.length; j++) {
					scores[c] += ((row[j] ?? 0) - (this.yMean_![j] ?? 0)) * (this.yRotations_![c]?.[j] ?? 0);
				}
			}
			return scores;
		});
		return { xScores, yScores };
	}

	fitTransform(X: Float64Array[], Y: Float64Array[]): { xScores: Float64Array[]; yScores: Float64Array[] } {
		this.fit(X, Y);
		const result = this.transform(X, Y);
		return { xScores: result.xScores, yScores: result.yScores! };
	}
}
