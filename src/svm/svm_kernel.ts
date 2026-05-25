/**
 * SVM kernel utilities and kernel matrix computation.
 * Port of sklearn.svm._base kernel utilities
 */

export type KernelType = "linear" | "poly" | "rbf" | "sigmoid" | "precomputed";

export interface KernelParams {
	kernel?: KernelType;
	degree?: number;
	gamma?: number | "scale" | "auto";
	coef0?: number;
}

/** Compute linear kernel between two vectors */
export function linearKernel(x: Float64Array, y: Float64Array): number {
	let dot = 0;
	for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (y[j] ?? 0);
	return dot;
}

/** Compute polynomial kernel */
export function polyKernel(x: Float64Array, y: Float64Array, degree = 3, gamma = 1.0, coef0 = 1.0): number {
	let dot = 0;
	for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (y[j] ?? 0);
	return (gamma * dot + coef0) ** degree;
}

/** Compute RBF kernel */
export function rbfKernel(x: Float64Array, y: Float64Array, gamma = 1.0): number {
	let distSq = 0;
	for (let j = 0; j < x.length; j++) distSq += ((x[j] ?? 0) - (y[j] ?? 0)) ** 2;
	return Math.exp(-gamma * distSq);
}

/** Compute sigmoid kernel */
export function sigmoidKernel(x: Float64Array, y: Float64Array, gamma = 0.01, coef0 = 0.0): number {
	let dot = 0;
	for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (y[j] ?? 0);
	return Math.tanh(gamma * dot + coef0);
}

/** Compute chi2 kernel */
export function chi2Kernel(x: Float64Array, y: Float64Array, gamma = 1.0): number {
	let result = 0;
	for (let j = 0; j < x.length; j++) {
		const sum = (x[j] ?? 0) + (y[j] ?? 0);
		if (sum > 0) result += ((x[j] ?? 0) - (y[j] ?? 0)) ** 2 / sum;
	}
	return Math.exp(-gamma * result);
}

/** Compute Laplacian kernel */
export function laplacianKernel(x: Float64Array, y: Float64Array, gamma = 1.0): number {
	let dist = 0;
	for (let j = 0; j < x.length; j++) dist += Math.abs((x[j] ?? 0) - (y[j] ?? 0));
	return Math.exp(-gamma * dist);
}

/** Compute cosine similarity kernel */
export function cosineKernel(x: Float64Array, y: Float64Array): number {
	let dot = 0;
	let normX = 0;
	let normY = 0;
	for (let j = 0; j < x.length; j++) {
		dot += (x[j] ?? 0) * (y[j] ?? 0);
		normX += (x[j] ?? 0) ** 2;
		normY += (y[j] ?? 0) ** 2;
	}
	return dot / (Math.sqrt(normX) * Math.sqrt(normY) + 1e-10);
}

/** Compute kernel matrix between X and Y */
export function kernelMatrix(
	X: Float64Array[],
	Y: Float64Array[],
	params: KernelParams = {},
): Float64Array[] {
	const n = X.length;
	const m = Y.length;
	const kernel = params.kernel ?? "rbf";
	const degree = params.degree ?? 3;
	const coef0 = params.coef0 ?? 1.0;

	// Compute gamma
	let gamma: number;
	if (params.gamma === undefined || params.gamma === "scale") {
		const nFeatures = X[0]?.length ?? 1;
		// Estimate variance of X
		let varX = 0;
		for (const x of X) for (const v of x) varX += v * v;
		varX /= (n * (X[0]?.length ?? 1));
		gamma = 1 / (nFeatures * (varX || 1));
	} else if (params.gamma === "auto") {
		gamma = 1 / (X[0]?.length ?? 1);
	} else {
		gamma = params.gamma;
	}

	return Array.from({ length: n }, (_, i) => {
		const row = new Float64Array(m);
		for (let j = 0; j < m; j++) {
			const x = X[i]!;
			const y = Y[j]!;
			switch (kernel) {
				case "linear": row[j] = linearKernel(x, y); break;
				case "poly": row[j] = polyKernel(x, y, degree, gamma, coef0); break;
				case "rbf": row[j] = rbfKernel(x, y, gamma); break;
				case "sigmoid": row[j] = sigmoidKernel(x, y, gamma, coef0); break;
				default: row[j] = rbfKernel(x, y, gamma); break;
			}
		}
		return row;
	});
}

/** Compute kernel diagonal */
export function kernelDiag(X: Float64Array[], params: KernelParams = {}): Float64Array {
	const n = X.length;
	const K = kernelMatrix(X, X, params);
	const diag = new Float64Array(n);
	for (let i = 0; i < n; i++) diag[i] = K[i]?.[i] ?? 0;
	return diag;
}

/**
 * SVM dual coefficient utilities.
 */
export interface SVMDualCoeffs {
	dualCoef: Float64Array[];
	intercept: Float64Array;
	supportVectors: Float64Array[];
	supportVectorIndices: Int32Array;
	nSupportPerClass: Int32Array;
}

/** Compute decision function from kernel evaluations */
export function decisionFunction(
	X: Float64Array[],
	supportVectors: Float64Array[],
	dualCoef: Float64Array[],
	intercept: Float64Array,
	params: KernelParams = {},
): Float64Array[] {
	const n = X.length;
	const K = kernelMatrix(X, supportVectors, params);
	const nClasses = dualCoef.length + 1;
	return Array.from({ length: n }, (_, i) => {
		const scores = new Float64Array(dualCoef.length);
		for (let j = 0; j < dualCoef.length; j++) {
			let score = intercept[j] ?? 0;
			for (let k = 0; k < supportVectors.length; k++) {
				score += (dualCoef[j]?.[k] ?? 0) * (K[i]?.[k] ?? 0);
			}
			scores[j] = score;
		}
		return scores;
	});
}
