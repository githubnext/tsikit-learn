/**
 * Preprocessing extensions: MaxAbsScaler, RobustScaler extensions.
 * Port of sklearn.preprocessing extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Scale features to a specified range per feature. */
export class MaxAbsScalerFull {
	private maxAbsVals_: Float64Array | null = null;
	private scaleVals_: Float64Array | null = null;

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const maxAbs = new Float64Array(nFeatures).fill(0);
		for (const row of X) {
			for (let j = 0; j < nFeatures; j++) {
				const v = Math.abs(row[j] ?? 0);
				if (v > (maxAbs[j] ?? 0)) maxAbs[j] = v;
			}
		}
		this.maxAbsVals_ = maxAbs;
		this.scaleVals_ = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) {
			const m = maxAbs[j] ?? 0;
			this.scaleVals_[j] = m === 0 ? 1 : 1 / m;
		}
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.scaleVals_ === null) throw new NotFittedError("MaxAbsScalerFull is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(row.length);
			for (let j = 0; j < row.length; j++) {
				out[j] = (row[j] ?? 0) * (this.scaleVals_![j] ?? 1);
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	inverseTransform(X: Float64Array[]): Float64Array[] {
		if (this.maxAbsVals_ === null) throw new NotFittedError("MaxAbsScalerFull is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(row.length);
			for (let j = 0; j < row.length; j++) {
				out[j] = (row[j] ?? 0) * (this.maxAbsVals_![j] ?? 1);
			}
			return out;
		});
	}
}

/** Quantile-based feature discretization (equal-frequency binning). */
export class EqualFrequencyBinner {
	private binEdges_: Float64Array[] | null = null;
	readonly nBins: number;

	constructor(options: { nBins?: number } = {}) {
		this.nBins = options.nBins ?? 5;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const nSamples = X.length;
		this.binEdges_ = Array.from({ length: nFeatures }, (_, j) => {
			const vals = new Float64Array(nSamples);
			for (let i = 0; i < nSamples; i++) vals[i] = X[i]?.[j] ?? 0;
			vals.sort();
			const edges = new Float64Array(this.nBins + 1);
			edges[0] = vals[0] ?? 0;
			edges[this.nBins] = vals[nSamples - 1] ?? 0;
			for (let k = 1; k < this.nBins; k++) {
				const idx = (k * nSamples) / this.nBins;
				const lo = Math.floor(idx);
				const hi = Math.ceil(idx);
				edges[k] =
					lo === hi
						? vals[lo] ?? 0
						: ((vals[lo] ?? 0) + (vals[hi] ?? 0)) / 2;
			}
			return edges;
		});
		return this;
	}

	transform(X: Float64Array[]): Int32Array[] {
		if (this.binEdges_ === null) throw new NotFittedError("EqualFrequencyBinner is not fitted.");
		return X.map((row) => {
			const out = new Int32Array(row.length);
			for (let j = 0; j < row.length; j++) {
				const edges = this.binEdges_![j]!;
				const v = row[j] ?? 0;
				let bin = this.nBins - 1;
				for (let k = 1; k < edges.length; k++) {
					if (v < (edges[k] ?? Number.POSITIVE_INFINITY)) {
						bin = k - 1;
						break;
					}
				}
				out[j] = bin;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Int32Array[] {
		return this.fit(X).transform(X);
	}
}

/** Label propagation binarizer (extend multi-label binarization). */
export class MultiOutputBinarizer {
	private classes_: Int32Array | null = null;

	fit(y: Int32Array[]): this {
		const allClasses = new Set<number>();
		for (const row of y) for (let i = 0; i < row.length; i++) allClasses.add(row[i] ?? 0);
		const sorted = [...allClasses].sort((a, b) => a - b);
		this.classes_ = new Int32Array(sorted);
		return this;
	}

	transform(y: Int32Array[]): Int32Array[] {
		if (this.classes_ === null) throw new NotFittedError("MultiOutputBinarizer is not fitted.");
		const nClasses = this.classes_.length;
		const classIdx = new Map<number, number>();
		for (let k = 0; k < nClasses; k++) classIdx.set(this.classes_[k]!, k);
		return y.map((row) => {
			const out = new Int32Array(nClasses);
			for (let i = 0; i < row.length; i++) {
				const idx = classIdx.get(row[i] ?? 0);
				if (idx !== undefined) out[idx] = 1;
			}
			return out;
		});
	}

	inverseTransform(Y: Int32Array[]): Int32Array[] {
		if (this.classes_ === null) throw new NotFittedError("MultiOutputBinarizer is not fitted.");
		return Y.map((row) => {
			const active: number[] = [];
			for (let k = 0; k < row.length; k++) {
				if ((row[k] ?? 0) === 1) active.push(this.classes_![k]!);
			}
			return new Int32Array(active);
		});
	}
}

/** Add polynomial interaction features (degree-2 interactions only). */
export function addInteractionFeatures(X: Float64Array[]): Float64Array[] {
	const nFeatures = X[0]?.length ?? 0;
	const nInteractions = (nFeatures * (nFeatures + 1)) / 2;
	return X.map((row) => {
		const out = new Float64Array(nFeatures + nInteractions);
		let idx = 0;
		for (let j = 0; j < nFeatures; j++) out[idx++] = row[j] ?? 0;
		for (let j = 0; j < nFeatures; j++) {
			for (let k = j; k < nFeatures; k++) {
				out[idx++] = (row[j] ?? 0) * (row[k] ?? 0);
			}
		}
		return out;
	});
}
