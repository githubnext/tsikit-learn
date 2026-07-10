/**
 * Preprocessing helper utilities.
 * Port of sklearn.preprocessing._base helpers and utilities
 */

/**
 * Add dummy feature (bias column) to a data matrix.
 * Port of sklearn.preprocessing.add_dummy_feature
 */
export function addDummyFeature(X: Float64Array[], value = 1.0): Float64Array[] {
	return X.map((row) => {
		const newRow = new Float64Array(row.length + 1);
		newRow[0] = value;
		for (let j = 0; j < row.length; j++) newRow[j + 1] = row[j] ?? 0;
		return newRow;
	});
}

/**
 * Compute column-wise statistics.
 */
export interface ColumnStats {
	mean: Float64Array;
	std: Float64Array;
	min: Float64Array;
	max: Float64Array;
	nSamples: number;
}

export function computeColumnStats(X: Float64Array[]): ColumnStats {
	const n = X.length;
	const d = X[0]?.length ?? 0;
	const mean = new Float64Array(d);
	const min = new Float64Array(d).fill(Number.POSITIVE_INFINITY);
	const max = new Float64Array(d).fill(Number.NEGATIVE_INFINITY);

	for (const x of X) {
		for (let j = 0; j < d; j++) {
			const v = x[j] ?? 0;
			mean[j] = (mean[j] ?? 0) + v / n;
			if (v < min[j]!) min[j] = v;
			if (v > max[j]!) max[j] = v;
		}
	}

	const std = new Float64Array(d);
	for (const x of X) {
		for (let j = 0; j < d; j++) {
			std[j] = (std[j] ?? 0) + ((x[j] ?? 0) - mean[j]!) ** 2 / n;
		}
	}
	for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]!);

	return { mean, std, min, max, nSamples: n };
}

/**
 * Check and validate sample weights.
 */
export function validateSampleWeight(
	sampleWeight: Float64Array | null,
	nSamples: number,
	dtype = "float64",
): Float64Array {
	void dtype;
	if (sampleWeight === null) {
		return new Float64Array(nSamples).fill(1.0);
	}
	if (sampleWeight.length !== nSamples) {
		throw new Error(`sampleWeight has ${sampleWeight.length} samples, expected ${nSamples}`);
	}
	return sampleWeight;
}

/**
 * Scale input to unit norm (inplace version for a single vector).
 */
export function normalizeVector(x: Float64Array, norm: "l1" | "l2" | "max" = "l2"): Float64Array {
	let scale: number;
	if (norm === "l1") {
		scale = 0;
		for (const v of x) scale += Math.abs(v);
	} else if (norm === "max") {
		scale = 0;
		for (const v of x) if (Math.abs(v) > scale) scale = Math.abs(v);
	} else {
		scale = 0;
		for (const v of x) scale += v * v;
		scale = Math.sqrt(scale);
	}
	if (scale === 0) return x.slice();
	const result = new Float64Array(x.length);
	for (let i = 0; i < x.length; i++) result[i] = (x[i] ?? 0) / scale;
	return result;
}

/**
 * Binarize a data matrix (threshold each element).
 */
export function binarize(X: Float64Array[], threshold = 0.0): Float64Array[] {
	return X.map((row) => {
		const result = new Float64Array(row.length);
		for (let j = 0; j < row.length; j++) result[j] = (row[j] ?? 0) > threshold ? 1 : 0;
		return result;
	});
}

/**
 * Apply column-wise centering and scaling.
 */
export function scaleMatrix(
	X: Float64Array[],
	mean: Float64Array | null,
	scale: Float64Array | null,
	copy = true,
): Float64Array[] {
	const result = copy ? X.map((r) => r.slice()) : X;
	const d = X[0]?.length ?? 0;
	for (const row of result) {
		for (let j = 0; j < d; j++) {
			if (mean) row[j]! -= mean[j] ?? 0;
			if (scale) row[j]! /= (scale[j] ?? 1) || 1;
		}
	}
	return result;
}

/**
 * Compute mean and standard deviation for scaling.
 */
export function meanAndStd(
	X: Float64Array[],
	withMean = true,
	withStd = true,
): { mean: Float64Array | null; std: Float64Array | null } {
	const d = X[0]?.length ?? 0;
	const mean = withMean ? new Float64Array(d) : null;
	const std = withStd ? new Float64Array(d) : null;
	const n = X.length;

	if (withMean && mean) {
		for (const x of X) for (let j = 0; j < d; j++) mean[j] = (mean[j] ?? 0) + (x[j] ?? 0) / n;
	}

	if (withStd && std) {
		for (const x of X) {
			for (let j = 0; j < d; j++) {
				const v = (x[j] ?? 0) - (mean ? mean[j]! : 0);
				std[j] = (std[j] ?? 0) + v * v / n;
			}
		}
		for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]!);
	}

	return { mean, std };
}

/**
 * Compute min and max for min-max scaling.
 */
export function minMaxCompute(X: Float64Array[]): { min: Float64Array; max: Float64Array } {
	const d = X[0]?.length ?? 0;
	const min = new Float64Array(d).fill(Number.POSITIVE_INFINITY);
	const max = new Float64Array(d).fill(Number.NEGATIVE_INFINITY);
	for (const x of X) {
		for (let j = 0; j < d; j++) {
			const v = x[j] ?? 0;
			if (v < min[j]!) min[j] = v;
			if (v > max[j]!) max[j] = v;
		}
	}
	return { min, max };
}
