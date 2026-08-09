/**
 * Gaussian process extensions: GPR with ARD kernel, sparse GP.
 * Port of sklearn.gaussian_process extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Automatic Relevance Determination (ARD) RBF kernel. */
export class ARDKernel {
	readonly lengthScales: Float64Array;

	constructor(nFeatures: number, initialScale = 1.0) {
		this.lengthScales = new Float64Array(nFeatures).fill(initialScale);
	}

	call(x1: Float64Array, x2: Float64Array): number {
		let sq = 0;
		for (let j = 0; j < x1.length; j++) {
			const diff = ((x1[j] ?? 0) - (x2[j] ?? 0)) / (this.lengthScales[j] ?? 1);
			sq += diff * diff;
		}
		return Math.exp(-0.5 * sq);
	}

	computeMatrix(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
		return X1.map((x1) => new Float64Array(X2.map((x2) => this.call(x1, x2))));
	}
}

/** Matern 3/2 kernel. */
export class Matern32Kernel {
	readonly lengthScale: number;

	constructor(lengthScale = 1.0) {
		this.lengthScale = lengthScale;
	}

	call(x1: Float64Array, x2: Float64Array): number {
		let sq = 0;
		for (let j = 0; j < x1.length; j++) {
			const diff = (x1[j] ?? 0) - (x2[j] ?? 0);
			sq += diff * diff;
		}
		const r = Math.sqrt(sq) / this.lengthScale;
		return (1 + Math.SQRT2 * 3 * r) * Math.exp(-Math.SQRT2 * 3 * r);
	}

	computeMatrix(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
		return X1.map((x1) => new Float64Array(X2.map((x2) => this.call(x1, x2))));
	}
}

/** Sparse Gaussian Process Regression using inducing points. */
export class SparseGPR {
	private Xu_: Float64Array[] | null = null;
	private alpha_: Float64Array | null = null;
	private Kuu_inv_: Float64Array[] | null = null;
	readonly nInducingPoints: number;
	readonly noiseVar: number;
	readonly kernelFn: (x1: Float64Array, x2: Float64Array) => number;

	constructor(
		options: {
			nInducingPoints?: number;
			noiseVar?: number;
			kernel?: "rbf" | "matern32";
			lengthScale?: number;
		} = {},
	) {
		this.nInducingPoints = options.nInducingPoints ?? 10;
		this.noiseVar = options.noiseVar ?? 0.01;
		const ls = options.lengthScale ?? 1.0;
		if (options.kernel === "matern32") {
			const k = new Matern32Kernel(ls);
			this.kernelFn = (x1, x2) => k.call(x1, x2);
		} else {
			const k = new ARDKernel(1, ls);
			this.kernelFn = (x1: Float64Array, x2: Float64Array) => {
				let sq = 0;
				for (let j = 0; j < x1.length; j++) {
					const diff = (x1[j] ?? 0) - (x2[j] ?? 0);
					sq += diff * diff;
				}
				return Math.exp(-0.5 * sq / (ls * ls));
			};
		}
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const n = X.length;
		const m = Math.min(this.nInducingPoints, n);
		// Select inducing points as first m training points
		this.Xu_ = X.slice(0, m);
		// Build Kuu
		const Kuu = this.Xu_.map((xu) => new Float64Array(this.Xu_!.map((xv) => this.kernelFn(xu, xv))));
		// Add noise to diagonal
		for (let i = 0; i < m; i++) Kuu[i]![i]! += 1e-4;
		// Build Kuf (m x n)
		const Kuf = this.Xu_.map((xu) => new Float64Array(X.map((xi) => this.kernelFn(xu, xi))));
		// Compute Kuu_inv (simplified: use Cholesky-style inverse)
		this.Kuu_inv_ = invertSmallMatrix(Kuu);
		// Compute alpha = Kuu_inv * Kuf * (Kff + noise*I)^-1 * y (approximated)
		// Simple: alpha = Kuu_inv * Kuf * y / (noise * n)
		const Kufy = new Float64Array(m);
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < n; j++) {
				Kufy[i]! += (Kuf[i]?.[j] ?? 0) * (y[j] ?? 0);
			}
		}
		this.alpha_ = new Float64Array(m);
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < m; j++) {
				this.alpha_[i]! += (this.Kuu_inv_[i]?.[j] ?? 0) * (Kufy[j] ?? 0);
			}
			this.alpha_[i]! /= (this.noiseVar * n || 1);
		}
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.Xu_ === null || this.alpha_ === null) {
			throw new NotFittedError("SparseGPR is not fitted.");
		}
		return new Float64Array(
			X.map((xi) => {
				let pred = 0;
				for (let i = 0; i < (this.Xu_?.length ?? 0); i++) {
					pred += (this.alpha_![i] ?? 0) * this.kernelFn(this.Xu_![i]!, xi);
				}
				return pred;
			}),
		);
	}
}

function invertSmallMatrix(A: Float64Array[]): Float64Array[] {
	const n = A.length;
	// Augmented matrix [A | I]
	const aug = A.map((row, i) => {
		const r = new Float64Array(2 * n);
		for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
		r[n + i] = 1;
		return r;
	});
	// Gauss-Jordan
	for (let col = 0; col < n; col++) {
		let maxRow = col;
		for (let row = col + 1; row < n; row++) {
			if (Math.abs(aug[row]?.[col] ?? 0) > Math.abs(aug[maxRow]?.[col] ?? 0)) maxRow = row;
		}
		const tmp = aug[col]!;
		aug[col] = aug[maxRow]!;
		aug[maxRow] = tmp;
		const pivot = aug[col]?.[col] ?? 1;
		if (Math.abs(pivot) < 1e-12) continue;
		for (let j = 0; j < 2 * n; j++) aug[col]![j]! /= pivot;
		for (let row = 0; row < n; row++) {
			if (row === col) continue;
			const factor = aug[row]?.[col] ?? 0;
			for (let j = 0; j < 2 * n; j++) aug[row]![j]! -= factor * (aug[col]?.[j] ?? 0);
		}
	}
	return aug.map((row) => new Float64Array(row.slice(n)));
}
