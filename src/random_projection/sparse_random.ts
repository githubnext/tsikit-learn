/**
 * Sparse random projection transformer.
 * Port of sklearn.random_projection (SparseRandomProjection, GaussianRandomProjection extensions)
 */

import { NotFittedError } from "../exceptions.js";

export interface SparseRandomProjectionParams {
	nComponents?: number | "auto";
	density?: number | "auto";
	epsSparse?: number;
	denseOutput?: boolean;
	randomState?: number | null;
}

/**
 * Reduce dimensionality through sparse random projection.
 * Uses the Achlioptas random sparse matrix.
 * Port of sklearn.random_projection.SparseRandomProjection
 */
export class SparseRandomProjectionBase {
	nComponents: number | "auto";
	density: number | "auto";
	epsSparse: number;
	denseOutput: boolean;
	randomState: number | null;

	nComponents_?: number;
	density_?: number;
	components_?: Int8Array[];
	nInputFeatures_?: number;

	constructor(params: SparseRandomProjectionParams = {}) {
		this.nComponents = params.nComponents ?? "auto";
		this.density = params.density ?? "auto";
		this.epsSparse = params.epsSparse ?? 0.1;
		this.denseOutput = params.denseOutput ?? false;
		this.randomState = params.randomState ?? null;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const d = X[0]?.length ?? 0;
		this.nInputFeatures_ = d;

		// Johnson-Lindenstrauss lemma for nComponents
		if (this.nComponents === "auto") {
			const eps = this.epsSparse;
			this.nComponents_ = Math.max(1, Math.ceil(4 * Math.log(n) / (eps ** 2 / 2 - eps ** 3 / 3)));
		} else {
			this.nComponents_ = this.nComponents;
		}

		// Achlioptas sparse matrix density
		if (this.density === "auto") {
			this.density_ = Math.min(1.0, 1 / Math.sqrt(d));
		} else {
			this.density_ = this.density;
		}

		// Generate sparse random matrix
		let seed = this.randomState ?? 42;
		const rand = (): number => {
			seed = (seed * 1664525 + 1013904223) & 0xffffffff;
			return (seed >>> 0) / 0x100000000;
		};

		const k = this.nComponents_;
		this.components_ = Array.from({ length: k }, () => {
			const row = new Int8Array(d);
			for (let j = 0; j < d; j++) {
				const r = rand();
				if (r < this.density_! / 2) row[j] = 1;
				else if (r < this.density_!) row[j] = -1;
				else row[j] = 0;
			}
			return row;
		});
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.components_) throw new NotFittedError("SparseRandomProjection");
		const k = this.nComponents_!;
		const d = this.nInputFeatures_!;
		const scale = 1 / Math.sqrt(this.density_! * d);
		return X.map((x) => {
			const result = new Float64Array(k);
			for (let c = 0; c < k; c++) {
				const comp = this.components_![c]!;
				let dot = 0;
				for (let j = 0; j < d; j++) dot += (comp[j] ?? 0) * (x[j] ?? 0);
				result[c] = dot * scale;
			}
			return result;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}

export interface GaussianRandomProjectionParams {
	nComponents?: number | "auto";
	eps?: number;
	randomState?: number | null;
}

/**
 * Reduce dimensionality through Gaussian random projection.
 * Port of sklearn.random_projection.GaussianRandomProjection
 */
export class GaussianRandomProjectionBase {
	nComponents: number | "auto";
	eps: number;
	randomState: number | null;

	nComponents_?: number;
	components_?: Float64Array[];
	nInputFeatures_?: number;

	constructor(params: GaussianRandomProjectionParams = {}) {
		this.nComponents = params.nComponents ?? "auto";
		this.eps = params.eps ?? 0.1;
		this.randomState = params.randomState ?? null;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const d = X[0]?.length ?? 0;
		this.nInputFeatures_ = d;

		if (this.nComponents === "auto") {
			const eps = this.eps;
			this.nComponents_ = Math.max(1, Math.ceil(4 * Math.log(n) / (eps ** 2 / 2 - eps ** 3 / 3)));
		} else {
			this.nComponents_ = this.nComponents;
		}

		let seed = this.randomState ?? 42;
		const randn = (): number => {
			// Box-Muller
			seed = (seed * 1664525 + 1013904223) & 0xffffffff;
			const u1 = ((seed >>> 0) + 1) / 0x100000001;
			seed = (seed * 1664525 + 1013904223) & 0xffffffff;
			const u2 = ((seed >>> 0) + 1) / 0x100000001;
			return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
		};

		const k = this.nComponents_;
		const scale = 1 / Math.sqrt(k);
		this.components_ = Array.from({ length: k }, () => {
			const row = new Float64Array(d);
			for (let j = 0; j < d; j++) row[j] = randn() * scale;
			return row;
		});
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.components_) throw new NotFittedError("GaussianRandomProjection");
		const k = this.nComponents_!;
		const d = this.nInputFeatures_!;
		return X.map((x) => {
			const result = new Float64Array(k);
			for (let c = 0; c < k; c++) {
				const comp = this.components_![c]!;
				let dot = 0;
				for (let j = 0; j < d; j++) dot += (comp[j] ?? 0) * (x[j] ?? 0);
				result[c] = dot;
			}
			return result;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}

/**
 * Johnson-Lindenstrauss lemma: minimum number of components.
 * Port of sklearn.random_projection.johnson_lindenstrauss_min_dim
 */
export function johnsonLindenstraussMinDimBase(nSamples: number, eps = 0.1): number {
	const denominator = eps ** 2 / 2 - eps ** 3 / 3;
	if (denominator <= 0) throw new Error("eps must be in (0, 1)");
	return Math.ceil(4 * Math.log(nSamples) / denominator);
}
