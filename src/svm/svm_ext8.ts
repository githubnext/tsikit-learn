/**
 * SVM extensions: kernel SVMs with different kernels.
 * Port of sklearn.svm extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute polynomial kernel matrix. */
export function polynomialKernel(
	X: Float64Array[],
	Y: Float64Array[],
	degree = 3,
	gamma = 1.0,
	coef0 = 1.0,
): Float64Array[] {
	return X.map((xi) =>
		new Float64Array(
			Y.map((yj) => {
				let dot = 0;
				for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
				return Math.pow(gamma * dot + coef0, degree);
			}),
		),
	);
}

/** Compute sigmoid kernel matrix. */
export function sigmoidKernel(
	X: Float64Array[],
	Y: Float64Array[],
	gamma = 1.0,
	coef0 = 0.0,
): Float64Array[] {
	return X.map((xi) =>
		new Float64Array(
			Y.map((yj) => {
				let dot = 0;
				for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
				return Math.tanh(gamma * dot + coef0);
			}),
		),
	);
}

/** Compute chi-squared kernel (for histograms). */
export function chiSquaredKernel(
	X: Float64Array[],
	Y: Float64Array[],
	gamma = 1.0,
): Float64Array[] {
	return X.map((xi) =>
		new Float64Array(
			Y.map((yj) => {
				let k = 0;
				for (let d = 0; d < xi.length; d++) {
					const a = xi[d] ?? 0;
					const b = yj[d] ?? 0;
					const s = a + b;
					if (s > 0) k += ((a - b) * (a - b)) / s;
				}
				return Math.exp(-gamma * k);
			}),
		),
	);
}

/** SMO (Sequential Minimal Optimization) for binary SVM classification. */
export class SVCKernelSMO {
	private alphas_: Float64Array | null = null;
	private bias_ = 0;
	private supportVectors_: Float64Array[] | null = null;
	private svAlphas_: Float64Array | null = null;
	private svLabels_: Int32Array | null = null;
	readonly C: number;
	readonly kernelFn: (xi: Float64Array, xj: Float64Array) => number;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			C?: number;
			kernel?: "rbf" | "linear" | "poly";
			gamma?: number;
			degree?: number;
			coef0?: number;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.C = options.C ?? 1.0;
		this.maxIter = options.maxIter ?? 100;
		this.tol = options.tol ?? 1e-3;
		const gamma = options.gamma ?? 1.0;
		const degree = options.degree ?? 3;
		const coef0 = options.coef0 ?? 0.0;
		const kernel = options.kernel ?? "rbf";
		this.kernelFn = (xi: Float64Array, xj: Float64Array): number => {
			let dot = 0;
			for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (xj[k] ?? 0);
			if (kernel === "linear") return dot;
			if (kernel === "poly") return Math.pow(gamma * dot + coef0, degree);
			// RBF
			let sq = 0;
			for (let k = 0; k < xi.length; k++) {
				const d = (xi[k] ?? 0) - (xj[k] ?? 0);
				sq += d * d;
			}
			return Math.exp(-gamma * sq);
		};
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const n = X.length;
		const alphas = new Float64Array(n);
		let bias = 0;
		// Simplified SMO: iterate over all pairs
		for (let iter = 0; iter < this.maxIter; iter++) {
			let numChanged = 0;
			for (let i = 0; i < n; i++) {
				const yi = (y[i] ?? 0) === 1 ? 1 : -1;
				let fi = bias;
				for (let k = 0; k < n; k++) {
					fi += (alphas[k] ?? 0) * ((y[k] ?? 0) === 1 ? 1 : -1) * this.kernelFn(X[k]!, X[i]!);
				}
				const ei = fi - yi;
				if ((yi * ei < -this.tol && (alphas[i] ?? 0) < this.C) ||
					(yi * ei > this.tol && (alphas[i] ?? 0) > 0)) {
					// Pick j != i randomly (simplified: take i+1 mod n)
					const j = (i + 1) % n;
					const yj = (y[j] ?? 0) === 1 ? 1 : -1;
					let fj = bias;
					for (let k = 0; k < n; k++) {
						fj += (alphas[k] ?? 0) * ((y[k] ?? 0) === 1 ? 1 : -1) * this.kernelFn(X[k]!, X[j]!);
					}
					const ej = fj - yj;
					const kii = this.kernelFn(X[i]!, X[i]!);
					const kjj = this.kernelFn(X[j]!, X[j]!);
					const kij = this.kernelFn(X[i]!, X[j]!);
					const eta = 2 * kij - kii - kjj;
					if (eta >= 0) continue;
					const aiOld = alphas[i] ?? 0;
					const ajOld = alphas[j] ?? 0;
					let ajNew = ajOld - (yj * (ei - ej)) / eta;
					let L: number;
					let H: number;
					if (yi === yj) {
						L = Math.max(0, aiOld + ajOld - this.C);
						H = Math.min(this.C, aiOld + ajOld);
					} else {
						L = Math.max(0, ajOld - aiOld);
						H = Math.min(this.C, this.C + ajOld - aiOld);
					}
					ajNew = Math.min(H, Math.max(L, ajNew));
					if (Math.abs(ajNew - ajOld) < 1e-5) continue;
					const aiNew = aiOld + yi * yj * (ajOld - ajNew);
					alphas[i] = aiNew;
					alphas[j] = ajNew;
					// Update bias
					const b1 = bias - ei - yi * (aiNew - aiOld) * kii - yj * (ajNew - ajOld) * kij;
					const b2 = bias - ej - yi * (aiNew - aiOld) * kij - yj * (ajNew - ajOld) * kjj;
					if (0 < aiNew && aiNew < this.C) bias = b1;
					else if (0 < ajNew && ajNew < this.C) bias = b2;
					else bias = (b1 + b2) / 2;
					numChanged++;
				}
			}
			if (numChanged === 0) break;
		}
		this.alphas_ = alphas;
		this.bias_ = bias;
		const svIdx = Array.from({ length: n }, (_, i) => i).filter((i) => (alphas[i] ?? 0) > 1e-5);
		this.supportVectors_ = svIdx.map((i) => X[i]!);
		this.svAlphas_ = new Float64Array(svIdx.map((i) => alphas[i] ?? 0));
		this.svLabels_ = new Int32Array(svIdx.map((i) => ((y[i] ?? 0) === 1 ? 1 : -1)));
		return this;
	}

	decision_function(X: Float64Array[]): Float64Array {
		if (this.supportVectors_ === null || this.svAlphas_ === null || this.svLabels_ === null) {
			throw new NotFittedError("SVCKernelSMO is not fitted.");
		}
		return new Float64Array(
			X.map((xi) => {
				let val = this.bias_;
				for (let k = 0; k < (this.supportVectors_?.length ?? 0); k++) {
					val += (this.svAlphas_![k] ?? 0) * (this.svLabels_![k] ?? 0) * this.kernelFn(this.supportVectors_![k]!, xi);
				}
				return val;
			}),
		);
	}

	predict(X: Float64Array[]): Int32Array {
		const df = this.decision_function(X);
		return new Int32Array(df.map((v) => (v >= 0 ? 1 : -1)));
	}
}
