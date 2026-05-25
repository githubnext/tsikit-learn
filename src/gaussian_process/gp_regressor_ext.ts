/**
 * GP Regressor extensions and multi-output GP.
 * Port of sklearn.gaussian_process._gpr extensions
 */

import { NotFittedError } from "../exceptions.js";

/** Noise kernel for GP regression */
export class WhiteNoise {
	noiseLevel: number;

	constructor(noiseLevel = 1.0) {
		this.noiseLevel = noiseLevel;
	}

	evaluate(x1: Float64Array, x2: Float64Array): number {
		// White noise: only contributes on diagonal
		let same = x1.length === x2.length;
		if (same) for (let j = 0; j < x1.length; j++) if ((x1[j] ?? 0) !== (x2[j] ?? 0)) { same = false; break; }
		return same ? this.noiseLevel : 0;
	}
}

/** Dot product kernel */
export class DotProductKernel {
	sigma0: number;

	constructor(sigma0 = 1.0) {
		this.sigma0 = sigma0;
	}

	evaluate(x1: Float64Array, x2: Float64Array): number {
		let dot = this.sigma0 ** 2;
		for (let j = 0; j < x1.length; j++) dot += (x1[j] ?? 0) * (x2[j] ?? 0);
		return dot;
	}
}

/** Compute kernel matrix with a kernel function */
export function computeKernelMatrix(
	X1: Float64Array[],
	X2: Float64Array[],
	kernelFn: (x1: Float64Array, x2: Float64Array) => number,
): Float64Array[] {
	const n1 = X1.length;
	const n2 = X2.length;
	return Array.from({ length: n1 }, (_, i) => {
		const row = new Float64Array(n2);
		for (let j = 0; j < n2; j++) row[j] = kernelFn(X1[i]!, X2[j]!);
		return row;
	});
}

/** Cholesky decomposition of positive definite matrix */
function cholesky(A: Float64Array[]): Float64Array[] {
	const n = A.length;
	const L: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
	for (let i = 0; i < n; i++) {
		for (let j = 0; j <= i; j++) {
			let sum = A[i]?.[j] ?? 0;
			for (let k = 0; k < j; k++) sum -= (L[i]?.[k] ?? 0) * (L[j]?.[k] ?? 0);
			L[i]![j] = i === j ? Math.sqrt(Math.max(sum, 0)) : sum / (L[j]![j] ?? 1);
		}
	}
	return L;
}

/** Solve L x = b (forward substitution) */
function solveLower(L: Float64Array[], b: Float64Array): Float64Array {
	const n = b.length;
	const x = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let sum = b[i]!;
		for (let j = 0; j < i; j++) sum -= (L[i]?.[j] ?? 0) * x[j]!;
		x[i] = sum / (L[i]![i] ?? 1);
	}
	return x;
}

/** Solve L^T x = b (backward substitution) */
function solveUpper(L: Float64Array[], b: Float64Array): Float64Array {
	const n = b.length;
	const x = new Float64Array(n);
	for (let i = n - 1; i >= 0; i--) {
		let sum = b[i]!;
		for (let j = i + 1; j < n; j++) sum -= (L[j]?.[i] ?? 0) * x[j]!;
		x[i] = sum / (L[i]![i] ?? 1);
	}
	return x;
}

export interface GPRegressorExtParams {
	kernelFn?: (x1: Float64Array, x2: Float64Array) => number;
	alpha?: number;
	normalizeY?: boolean;
}

/**
 * GP Regressor with explicit kernel function.
 * Complementary to the main GaussianProcessRegressor.
 */
export class GPRegressorExt {
	kernelFn: (x1: Float64Array, x2: Float64Array) => number;
	alpha: number;
	normalizeY: boolean;

	XTrain_?: Float64Array[];
	yTrain_?: Float64Array;
	L_?: Float64Array[];
	alpha_?: Float64Array;
	yMean_?: number;
	yStd_?: number;

	constructor(params: GPRegressorExtParams = {}) {
		// Default: RBF kernel
		this.kernelFn = params.kernelFn ?? ((x1, x2) => {
			let distSq = 0;
			for (let j = 0; j < x1.length; j++) distSq += ((x1[j] ?? 0) - (x2[j] ?? 0)) ** 2;
			return Math.exp(-0.5 * distSq);
		});
		this.alpha = params.alpha ?? 1e-10;
		this.normalizeY = params.normalizeY ?? false;
	}

	fit(X: Float64Array[], y: Float64Array): this {
		this.XTrain_ = X;
		const n = X.length;

		let trainY = y.slice();
		if (this.normalizeY) {
			this.yMean_ = trainY.reduce((s, v) => s + v, 0) / n;
			this.yStd_ = Math.sqrt(trainY.reduce((s, v) => s + (v - this.yMean_!) ** 2, 0) / n) || 1;
			trainY = new Float64Array(Array.from(trainY, (v) => (v - this.yMean_!) / this.yStd_!));
		}
		this.yTrain_ = trainY;

		// Kernel matrix + noise
		const K = computeKernelMatrix(X, X, this.kernelFn);
		for (let i = 0; i < n; i++) K[i]![i] += this.alpha;

		// Cholesky decomposition
		this.L_ = cholesky(K);
		const v = solveLower(this.L_, trainY);
		this.alpha_ = solveUpper(this.L_, v);
		return this;
	}

	predict(X: Float64Array[], returnStd = false): { mean: Float64Array; std?: Float64Array } {
		if (!this.XTrain_) throw new NotFittedError("GPRegressorExt");
		const n = X.length;
		const KStar = computeKernelMatrix(X, this.XTrain_, this.kernelFn);
		const mean = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < this.alpha_!.length; j++) {
				mean[i] += (KStar[i]?.[j] ?? 0) * this.alpha_![j]!;
			}
		}

		if (this.normalizeY && this.yMean_ !== undefined && this.yStd_ !== undefined) {
			for (let i = 0; i < n; i++) mean[i] = mean[i]! * this.yStd_! + this.yMean_!;
		}

		if (!returnStd) return { mean };

		const std = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			const kStarI = KStar[i]!;
			const v = solveLower(this.L_!, kStarI);
			let varI = this.kernelFn(X[i]!, X[i]!);
			for (let j = 0; j < v.length; j++) varI -= v[j]! * v[j]!;
			std[i] = Math.sqrt(Math.max(0, varI)) * (this.yStd_ ?? 1);
		}
		return { mean, std };
	}

	score(X: Float64Array[], y: Float64Array): number {
		const { mean } = this.predict(X);
		const yMean = y.reduce((s, v) => s + v, 0) / y.length;
		let ss_res = 0;
		let ss_tot = 0;
		for (let i = 0; i < y.length; i++) {
			ss_res += ((y[i] ?? 0) - (mean[i] ?? 0)) ** 2;
			ss_tot += ((y[i] ?? 0) - yMean) ** 2;
		}
		return 1 - ss_res / (ss_tot || 1);
	}
}
