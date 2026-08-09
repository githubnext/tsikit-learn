/**
 * Extended isotonic regression utilities.
 * Port of sklearn.isotonic extensions.
 */

/** Check if an array is monotone increasing. */
export function checkIncreasing(x: Float64Array, y: Float64Array): boolean {
	if (x.length !== y.length) throw new Error("x and y must have the same length");
	// Use Spearman correlation sign
	const n = x.length;
	if (n < 2) return true;
	const xRanks = rankData(x);
	const yRanks = rankData(y);
	let cov = 0;
	for (let i = 0; i < n; i++) {
		cov += ((xRanks[i] ?? 0) - (n + 1) / 2) * ((yRanks[i] ?? 0) - (n + 1) / 2);
	}
	return cov >= 0;
}

function rankData(arr: Float64Array): Float64Array {
	const n = arr.length;
	const idx = Array.from({ length: n }, (_, i) => i).sort(
		(a, b) => (arr[a] ?? 0) - (arr[b] ?? 0),
	);
	const ranks = new Float64Array(n);
	let i = 0;
	while (i < n) {
		let j = i;
		while (j < n - 1 && (arr[idx[j]!] ?? 0) === (arr[idx[j + 1]!] ?? 0)) j++;
		const rank = (i + j) / 2 + 1;
		for (let k = i; k <= j; k++) ranks[idx[k]!] = rank;
		i = j + 1;
	}
	return ranks;
}

/** Pool Adjacent Violators (PAV) algorithm for isotonic regression. */
export function pavAlgorithm(
	y: Float64Array,
	weights: Float64Array | null = null,
): Float64Array {
	const n = y.length;
	const result = new Float64Array(n);
	const w = weights ?? new Float64Array(n).fill(1);
	// Pool adjacent violators
	const poolY: number[] = [];
	const poolW: number[] = [];
	for (let i = 0; i < n; i++) {
		poolY.push(y[i] ?? 0);
		poolW.push(w[i] ?? 1);
		while (poolY.length >= 2) {
			const last = poolY.length - 1;
			if ((poolY[last] ?? 0) < (poolY[last - 1] ?? 0)) {
				const wLast = poolW[last] ?? 1;
				const wPrev = poolW[last - 1] ?? 1;
				const mergedVal =
					((poolY[last] ?? 0) * wLast + (poolY[last - 1] ?? 0) * wPrev) /
					(wLast + wPrev);
				poolY.pop();
				poolW.pop();
				poolY[last - 1] = mergedVal;
				poolW[last - 1] = wLast + wPrev;
			} else {
				break;
			}
		}
	}
	// Expand pools back
	let idx = 0;
	for (let i = 0; i < poolY.length; i++) {
		const cnt = Math.round(poolW[i] ?? 1);
		for (let k = 0; k < cnt && idx < n; k++) {
			result[idx++] = poolY[i] ?? 0;
		}
	}
	return result;
}

/** Compute the isotonic regression using weighted least squares with monotone constraints. */
export class IsotonicRegressionFull {
	private yThresholds_: Float64Array | null = null;
	private xThresholds_: Float64Array | null = null;
	readonly increasing: boolean;
	readonly outOfBounds: "nan" | "clip" | "raise";

	constructor(
		options: {
			increasing?: boolean;
			outOfBounds?: "nan" | "clip" | "raise";
		} = {},
	) {
		this.increasing = options.increasing ?? true;
		this.outOfBounds = options.outOfBounds ?? "nan";
	}

	fit(X: Float64Array, y: Float64Array, sampleWeight?: Float64Array): this {
		const n = X.length;
		// Sort by X
		const order = Array.from({ length: n }, (_, i) => i).sort(
			(a, b) => (X[a] ?? 0) - (X[b] ?? 0),
		);
		const sortedX = new Float64Array(n);
		const sortedY = new Float64Array(n);
		const sortedW = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			sortedX[i] = X[order[i]!] ?? 0;
			sortedY[i] = y[order[i]!] ?? 0;
			sortedW[i] = sampleWeight?.[order[i]!] ?? 1;
		}
		const isotonic = this.increasing ? pavAlgorithm(sortedY, sortedW) : pavAlgorithmDecreasing(sortedY, sortedW);
		this.xThresholds_ = sortedX;
		this.yThresholds_ = isotonic;
		return this;
	}

	predict(X: Float64Array): Float64Array {
		if (this.xThresholds_ === null || this.yThresholds_ === null) {
			throw new Error("IsotonicRegressionFull is not fitted.");
		}
		const result = new Float64Array(X.length);
		const xMin = this.xThresholds_[0] ?? 0;
		const xMax = this.xThresholds_[this.xThresholds_.length - 1] ?? 0;
		for (let i = 0; i < X.length; i++) {
			const xi = X[i] ?? 0;
			if (xi < xMin) {
				if (this.outOfBounds === "clip") result[i] = this.yThresholds_[0] ?? 0;
				else if (this.outOfBounds === "raise") throw new Error(`Out of bounds: ${xi}`);
				else result[i] = Number.NaN;
			} else if (xi > xMax) {
				if (this.outOfBounds === "clip")
					result[i] = this.yThresholds_[this.yThresholds_.length - 1] ?? 0;
				else if (this.outOfBounds === "raise") throw new Error(`Out of bounds: ${xi}`);
				else result[i] = Number.NaN;
			} else {
				result[i] = interpolate(xi, this.xThresholds_, this.yThresholds_);
			}
		}
		return result;
	}

	score(X: Float64Array, y: Float64Array): number {
		const yPred = this.predict(X);
		let ssTot = 0;
		let ssRes = 0;
		let yMean = 0;
		for (let i = 0; i < y.length; i++) yMean += y[i] ?? 0;
		yMean /= y.length;
		for (let i = 0; i < y.length; i++) {
			const d = (y[i] ?? 0) - yMean;
			ssTot += d * d;
			const e = (y[i] ?? 0) - (yPred[i] ?? 0);
			ssRes += e * e;
		}
		return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
	}
}

function pavAlgorithmDecreasing(y: Float64Array, w: Float64Array): Float64Array {
	const reversed = new Float64Array(y.length);
	for (let i = 0; i < y.length; i++) reversed[i] = y[y.length - 1 - i] ?? 0;
	const revW = new Float64Array(w.length);
	for (let i = 0; i < w.length; i++) revW[i] = w[w.length - 1 - i] ?? 0;
	const result = pavAlgorithm(reversed, revW);
	const out = new Float64Array(result.length);
	for (let i = 0; i < result.length; i++) out[i] = result[result.length - 1 - i] ?? 0;
	return out;
}

function interpolate(x: number, xs: Float64Array, ys: Float64Array): number {
	let lo = 0;
	let hi = xs.length - 1;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if ((xs[mid] ?? 0) < x) lo = mid + 1;
		else hi = mid;
	}
	if (lo === 0) return ys[0] ?? 0;
	const x0 = xs[lo - 1] ?? 0;
	const x1 = xs[lo] ?? 0;
	const y0 = ys[lo - 1] ?? 0;
	const y1 = ys[lo] ?? 0;
	if (x1 === x0) return y0;
	return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
}
