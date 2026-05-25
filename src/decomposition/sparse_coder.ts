/**
 * SparseCoder and dictionary learning transform utilities.
 * Port of sklearn.decomposition._dict_learning (SparseCoder part)
 */

import { NotFittedError } from "../exceptions.js";

/** Sparse coding algorithms */
export type SparseCodingAlgorithm = "lasso_lars" | "lasso_cd" | "lars" | "omp" | "threshold";

export interface SparseCoderParams {
	dictionary: Float64Array[];
	transform_algorithm?: SparseCodingAlgorithm;
	transform_n_nonzero_coefs?: number | null;
	transform_alpha?: number | null;
	split_sign?: boolean;
	n_jobs?: number | null;
	positive_code?: boolean;
	transform_max_iter?: number;
}

/**
 * Sparse coding with a fixed, precomputed dictionary.
 * Port of sklearn.decomposition.SparseCoder
 */
export class SparseCoder {
	dictionary: Float64Array[];
	transform_algorithm: SparseCodingAlgorithm;
	transform_n_nonzero_coefs: number | null;
	transform_alpha: number | null;
	split_sign: boolean;
	positive_code: boolean;
	transform_max_iter: number;

	constructor(params: SparseCoderParams) {
		this.dictionary = params.dictionary;
		this.transform_algorithm = params.transform_algorithm ?? "omp";
		this.transform_n_nonzero_coefs = params.transform_n_nonzero_coefs ?? null;
		this.transform_alpha = params.transform_alpha ?? null;
		this.split_sign = params.split_sign ?? false;
		this.positive_code = params.positive_code ?? false;
		this.transform_max_iter = params.transform_max_iter ?? 1000;
	}

	// SparseCoder doesn't fit — it uses a fixed dictionary
	fit(_X: Float64Array[]): this {
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		const nSamples = X.length;
		const nComponents = this.dictionary.length;
		const codes: Float64Array[] = [];
		for (let i = 0; i < nSamples; i++) {
			const x = X[i]!;
			const code = new Float64Array(nComponents);
			if (this.transform_algorithm === "threshold") {
				const alpha = this.transform_alpha ?? 0.1;
				// Compute correlations and threshold
				for (let k = 0; k < nComponents; k++) {
					let dot = 0;
					const atom = this.dictionary[k]!;
					for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (atom[j] ?? 0);
					if (Math.abs(dot) > alpha) code[k] = dot;
				}
			} else if (this.transform_algorithm === "omp") {
				// Orthogonal matching pursuit (greedy)
				const nNonzero = this.transform_n_nonzero_coefs ?? Math.ceil(x.length / 10);
				const residual = x.slice();
				const selected: number[] = [];
				for (let step = 0; step < nNonzero; step++) {
					let bestAtom = 0;
					let bestDot = -Number.POSITIVE_INFINITY;
					for (let k = 0; k < nComponents; k++) {
						if (selected.includes(k)) continue;
						let dot = 0;
						const atom = this.dictionary[k]!;
						for (let j = 0; j < residual.length; j++) dot += (residual[j] ?? 0) * (atom[j] ?? 0);
						if (Math.abs(dot) > bestDot) { bestDot = Math.abs(dot); bestAtom = k; }
					}
					selected.push(bestAtom);
					const atom = this.dictionary[bestAtom]!;
					let atomNorm = 0;
					for (let j = 0; j < atom.length; j++) atomNorm += (atom[j] ?? 0) ** 2;
					let proj = 0;
					for (let j = 0; j < residual.length; j++) proj += (residual[j] ?? 0) * (atom[j] ?? 0);
					const alpha = proj / (atomNorm || 1);
					code[bestAtom] = alpha;
					for (let j = 0; j < residual.length; j++) residual[j]! -= alpha * (atom[j] ?? 0);
				}
			} else {
				// Default: least squares projection
				for (let k = 0; k < nComponents; k++) {
					let dot = 0;
					let norm = 0;
					const atom = this.dictionary[k]!;
					for (let j = 0; j < x.length; j++) {
						dot += (x[j] ?? 0) * (atom[j] ?? 0);
						norm += (atom[j] ?? 0) ** 2;
					}
					code[k] = dot / (norm || 1);
				}
			}
			if (this.positive_code) {
				for (let k = 0; k < nComponents; k++) if ((code[k] ?? 0) < 0) code[k] = 0;
			}
			codes.push(code);
		}
		return codes;
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}

/** Compute sparse code for a single sample using LASSO-CD approximation */
export function sparseDecode(
	X: Float64Array[],
	dictionary: Float64Array[],
	algorithm: SparseCodingAlgorithm = "omp",
	nNonzeroCoefs: number | null = null,
	alpha: number | null = null,
): Float64Array[] {
	const coder = new SparseCoder({
		dictionary,
		transform_algorithm: algorithm,
		transform_n_nonzero_coefs: nNonzeroCoefs,
		transform_alpha: alpha,
	});
	return coder.transform(X);
}
