/**
 * Extended utils: sample weight utilities, multioutput helpers.
 * Port of sklearn.utils extensions.
 */

/** Compute effective number of samples given sample weights. */
export function computeEffectiveN(sampleWeight: Float64Array): number {
	let sumW = 0;
	let sumW2 = 0;
	for (let i = 0; i < sampleWeight.length; i++) {
		const w = sampleWeight[i] ?? 0;
		sumW += w;
		sumW2 += w * w;
	}
	return sumW2 === 0 ? 0 : (sumW * sumW) / sumW2;
}

/** Normalize sample weights to sum to nSamples. */
export function normalizeSampleWeight(
	sampleWeight: Float64Array | null,
	nSamples: number,
): Float64Array {
	if (sampleWeight === null) return new Float64Array(nSamples).fill(1);
	let total = 0;
	for (let i = 0; i < sampleWeight.length; i++) total += sampleWeight[i] ?? 0;
	const scale = total === 0 ? 1 : nSamples / total;
	const result = new Float64Array(sampleWeight.length);
	for (let i = 0; i < sampleWeight.length; i++) result[i] = (sampleWeight[i] ?? 0) * scale;
	return result;
}

/** Compute indices that sort an array (argsort). */
export function argsort(arr: Float64Array, ascending = true): Int32Array {
	const idx = Int32Array.from({ length: arr.length }, (_, i) => i);
	idx.sort((a, b) =>
		ascending ? (arr[a] ?? 0) - (arr[b] ?? 0) : (arr[b] ?? 0) - (arr[a] ?? 0),
	);
	return idx;
}

/** Compute the unique values in an array along with their counts. */
export function uniqueWithCounts(
	arr: Int32Array,
): { values: Int32Array; counts: Int32Array } {
	const map = new Map<number, number>();
	for (let i = 0; i < arr.length; i++) {
		const v = arr[i] ?? 0;
		map.set(v, (map.get(v) ?? 0) + 1);
	}
	const sorted = [...map.entries()].sort((a, b) => a[0] - b[0]);
	const values = new Int32Array(sorted.map(([v]) => v));
	const counts = new Int32Array(sorted.map(([, c]) => c));
	return { values, counts };
}

/** Compute column-wise means and standard deviations of a 2D array. */
export function colStats(
	X: Float64Array[],
): { means: Float64Array; stds: Float64Array } {
	const nSamples = X.length;
	const nFeatures = X[0]?.length ?? 0;
	const means = new Float64Array(nFeatures);
	const stds = new Float64Array(nFeatures);
	for (let j = 0; j < nFeatures; j++) {
		let s = 0;
		for (let i = 0; i < nSamples; i++) s += X[i]?.[j] ?? 0;
		means[j] = s / nSamples;
	}
	for (let j = 0; j < nFeatures; j++) {
		let v = 0;
		for (let i = 0; i < nSamples; i++) {
			const d = (X[i]?.[j] ?? 0) - (means[j] ?? 0);
			v += d * d;
		}
		stds[j] = Math.sqrt(v / nSamples);
	}
	return { means, stds };
}

/** Compute pairwise squared Euclidean distances. */
export function pairwiseSquaredDistances(
	X: Float64Array[],
	Y?: Float64Array[],
): Float64Array[] {
	const A = X;
	const B = Y ?? X;
	const m = A.length;
	const n = B.length;
	const D: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n));
	for (let i = 0; i < m; i++) {
		for (let j = 0; j < n; j++) {
			let d = 0;
			const len = A[i]?.length ?? 0;
			for (let k = 0; k < len; k++) {
				const diff = (A[i]?.[k] ?? 0) - (B[j]?.[k] ?? 0);
				d += diff * diff;
			}
			D[i]![j] = d;
		}
	}
	return D;
}

/** Column-wise min/max for arrays. */
export function colMinMax(
	X: Float64Array[],
): { mins: Float64Array; maxs: Float64Array } {
	const nFeatures = X[0]?.length ?? 0;
	const mins = new Float64Array(nFeatures).fill(Number.POSITIVE_INFINITY);
	const maxs = new Float64Array(nFeatures).fill(Number.NEGATIVE_INFINITY);
	for (const row of X) {
		for (let j = 0; j < nFeatures; j++) {
			const v = row[j] ?? 0;
			if (v < (mins[j] ?? Number.POSITIVE_INFINITY)) mins[j] = v;
			if (v > (maxs[j] ?? Number.NEGATIVE_INFINITY)) maxs[j] = v;
		}
	}
	return { mins, maxs };
}

/** Check if all values in an array are finite. */
export function allFinite(X: Float64Array[]): boolean {
	for (const row of X) {
		for (let j = 0; j < row.length; j++) {
			const v = row[j] ?? 0;
			if (!Number.isFinite(v)) return false;
		}
	}
	return true;
}

/** Shuffle indices in place using Fisher-Yates. */
export function shuffleIndices(indices: Int32Array, seed = 0): void {
	let rng = seed;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	for (let i = indices.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		const tmp = indices[i]!;
		indices[i] = indices[j]!;
		indices[j] = tmp;
	}
}
