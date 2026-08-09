/**
 * Decomposition extensions: factor analysis extensions, sparse decomposition.
 * Port of sklearn.decomposition extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Factor Analysis: finds latent factors explaining observed variable covariance. */
export class FactorAnalysisExt {
	private components_: Float64Array[] | null = null;
	private noiseLV_: Float64Array | null = null;
	private mean_: Float64Array | null = null;
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
		this.maxIter = options.maxIter ?? 1000;
		this.tol = options.tol ?? 1e-4;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const p = X[0]?.length ?? 0;
		const k = Math.min(this.nComponents, p);

		const mean = new Float64Array(p);
		for (const row of X) for (let j = 0; j < p; j++) mean[j]! += row[j] ?? 0;
		for (let j = 0; j < p; j++) mean[j]! /= n;
		this.mean_ = mean;

		// Center data
		const Xc = X.map((row) => {
			const centered = new Float64Array(p);
			for (let j = 0; j < p; j++) centered[j] = (row[j] ?? 0) - (mean[j] ?? 0);
			return centered;
		});

		// EM algorithm: init
		const W: Float64Array[] = Array.from({ length: k }, () => {
			const w = new Float64Array(p);
			for (let j = 0; j < p; j++) w[j] = (Math.random() - 0.5) * 0.01;
			return w;
		});
		const psi = new Float64Array(p).fill(1.0); // unique variances

		for (let iter = 0; iter < this.maxIter; iter++) {
			// E-step: compute posterior
			// M-step: update W and psi
			// Simplified: use first k right singular vectors
			// Compute covariance
			const cov: number[][] = Array.from({ length: p }, () => new Array(p).fill(0) as number[]);
			for (const row of Xc) {
				for (let a = 0; a < p; a++) {
					for (let b = 0; b < p; b++) {
						cov[a]![b]! += (row[a] ?? 0) * (row[b] ?? 0);
					}
				}
			}
			for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov[a]![b]! /= n;

			// Power iteration for top-k eigenvectors
			let maxChange = 0;
			for (let c = 0; c < k; c++) {
				const oldW = new Float64Array(W[c]!);
				// Deflated covariance application
				const newW = new Float64Array(p);
				for (let a = 0; a < p; a++) {
					for (let b = 0; b < p; b++) {
						newW[a]! += (cov[a]![b] ?? 0) * (W[c]![b] ?? 0);
					}
				}
				let norm = 0;
				for (let a = 0; a < p; a++) norm += (newW[a] ?? 0) * (newW[a] ?? 0);
				norm = Math.sqrt(norm);
				if (norm > 0) for (let a = 0; a < p; a++) newW[a]! /= norm;
				let chg = 0;
				for (let a = 0; a < p; a++) {
					const d = (newW[a] ?? 0) - (oldW[a] ?? 0);
					chg += d * d;
				}
				maxChange = Math.max(maxChange, chg);
				W[c] = newW;
			}
			// Update psi
			for (let j = 0; j < p; j++) {
				let explained = 0;
				for (let c = 0; c < k; c++) explained += (W[c]?.[j] ?? 0) * (W[c]?.[j] ?? 0);
				psi[j] = Math.max(0.005, (cov[j]?.[j] ?? 0) - explained);
			}
			if (maxChange < this.tol) break;
		}
		this.components_ = W;
		this.noiseLV_ = psi;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.components_ === null || this.mean_ === null) {
			throw new NotFittedError("FactorAnalysisExt is not fitted.");
		}
		const k = this.components_.length;
		return X.map((row) => {
			const out = new Float64Array(k);
			for (let c = 0; c < k; c++) {
				for (let j = 0; j < row.length; j++) {
					out[c]! += (this.components_![c]![j] ?? 0) * ((row[j] ?? 0) - (this.mean_![j] ?? 0));
				}
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	get components(): Float64Array[] {
		if (this.components_ === null) throw new NotFittedError("FactorAnalysisExt is not fitted.");
		return this.components_;
	}
}

/** Sparse random matrix for dimensionality reduction. */
export class SparseRandomProjectionExt {
	private randomMatrix_: Float64Array[] | null = null;
	readonly nComponents: number;
	readonly density: number;
	readonly randomState: number;

	constructor(
		options: {
			nComponents?: number;
			density?: number | "auto";
			randomState?: number;
		} = {},
	) {
		this.nComponents = options.nComponents ?? 10;
		this.density = typeof options.density === "number" ? options.density : 1 / 3;
		this.randomState = options.randomState ?? 0;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		const scale = Math.sqrt(1 / (this.density * this.nComponents));
		this.randomMatrix_ = Array.from({ length: nFeatures }, () => {
			const row = new Float64Array(this.nComponents);
			for (let c = 0; c < this.nComponents; c++) {
				const u = rand();
				if (u < this.density / 2) row[c] = -scale;
				else if (u < this.density) row[c] = scale;
				else row[c] = 0;
			}
			return row;
		});
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.randomMatrix_ === null) throw new NotFittedError("SparseRandomProjectionExt is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(this.nComponents);
			for (let j = 0; j < row.length; j++) {
				const rj = this.randomMatrix_![j];
				if (rj === undefined) continue;
				for (let c = 0; c < this.nComponents; c++) {
					out[c]! += (row[j] ?? 0) * (rj[c] ?? 0);
				}
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}
