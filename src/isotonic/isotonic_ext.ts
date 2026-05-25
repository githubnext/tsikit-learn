/**
 * Isotonic regression extensions.
 * Port of sklearn.isotonic extensions (_isotonic_regression, check_increasing)
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Check if a relationship is monotonically increasing or decreasing.
 * Port of sklearn.isotonic.check_increasing
 */
export function checkIncreasing(x: Float64Array, y: Float64Array): boolean {
	// Use Spearman correlation sign
	const n = x.length;
	let concordant = 0;
	let discordant = 0;
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			const dx = (x[i]! - x[j]!);
			const dy = (y[i]! - y[j]!);
			if (dx * dy > 0) concordant++;
			else if (dx * dy < 0) discordant++;
		}
	}
	return concordant >= discordant;
}

/** PAVA (Pool Adjacent Violators Algorithm) for isotonic regression */
export function isoReg(y: Float64Array, increasing = true): Float64Array {
	const n = y.length;
	const result = y.slice();

	if (n <= 1) return result;

	// Pool Adjacent Violators
	const poolSums: number[] = Array.from(y);
	const poolCounts: number[] = new Array(n).fill(1);
	let i = 0;

	while (i < poolSums.length - 1) {
		const shouldViolate = increasing
			? (poolSums[i]! / poolCounts[i]!) > (poolSums[i + 1]! / poolCounts[i + 1]!)
			: (poolSums[i]! / poolCounts[i]!) < (poolSums[i + 1]! / poolCounts[i + 1]!);

		if (shouldViolate) {
			poolSums[i] = poolSums[i]! + poolSums[i + 1]!;
			poolCounts[i] = poolCounts[i]! + poolCounts[i + 1]!;
			poolSums.splice(i + 1, 1);
			poolCounts.splice(i + 1, 1);
			if (i > 0) i--;
		} else {
			i++;
		}
	}

	// Expand back
	let idx = 0;
	for (let p = 0; p < poolSums.length; p++) {
		const mean = poolSums[p]! / poolCounts[p]!;
		for (let j = 0; j < poolCounts[p]!; j++) {
			result[idx++] = mean;
		}
	}
	return result;
}

/**
 * Extended isotonic regression with weights.
 * Port of sklearn.isotonic.isotonic_regression (weighted version)
 */
export function weightedIsoReg(
	y: Float64Array,
	sampleWeight: Float64Array | null = null,
	increasing = true,
): Float64Array {
	const n = y.length;
	const w = sampleWeight ?? new Float64Array(n).fill(1.0);
	const result = y.slice();

	const poolSums: number[] = Array.from(y).map((v, i) => v * (w[i] ?? 1));
	const poolWeights: number[] = Array.from(w);
	let i = 0;

	while (i < poolSums.length - 1) {
		const mean1 = poolSums[i]! / poolWeights[i]!;
		const mean2 = poolSums[i + 1]! / poolWeights[i + 1]!;
		const shouldViolate = increasing ? mean1 > mean2 : mean1 < mean2;

		if (shouldViolate) {
			poolSums[i] = poolSums[i]! + poolSums[i + 1]!;
			poolWeights[i] = poolWeights[i]! + poolWeights[i + 1]!;
			poolSums.splice(i + 1, 1);
			poolWeights.splice(i + 1, 1);
			if (i > 0) i--;
		} else {
			i++;
		}
	}

	let idx = 0;
	for (let p = 0; p < poolSums.length; p++) {
		const mean = poolSums[p]! / poolWeights[p]!;
		const cnt = Math.round(poolWeights[p]! / (w[idx] ?? 1));
		for (let j = 0; j < Math.max(1, cnt) && idx < n; j++) {
			result[idx++] = mean;
		}
	}
	return result;
}

/**
 * 2D isotonic regression (block model).
 * Port of sklearn.isotonic._isotonic_regression_2d
 */
export function isoReg2D(
	y: Float64Array[],
	increasing: [boolean, boolean] = [true, true],
): Float64Array[] {
	const nRows = y.length;
	const nCols = y[0]?.length ?? 0;

	// Apply 1D isotonic regression along rows
	let result = y.map((row) => isoReg(row, increasing[1]));

	// Apply 1D isotonic regression along columns
	for (let j = 0; j < nCols; j++) {
		const col = new Float64Array(nRows);
		for (let i = 0; i < nRows; i++) col[i] = result[i]?.[j] ?? 0;
		const isoCol = isoReg(col, increasing[0]);
		for (let i = 0; i < nRows; i++) result[i]![j] = isoCol[i]!;
	}

	return result;
}

/**
 * Extended IsotonicRegression with out-of-bounds behavior.
 * Port of sklearn.isotonic.IsotonicRegression (extensions)
 */
export class IsotonicRegressionExt {
	increasing: boolean | "auto";
	yMin: number | null;
	yMax: number | null;
	outOfBounds: "nan" | "clip" | "raise";

	increasing_?: boolean;
	xThresholds_?: Float64Array;
	yThresholds_?: Float64Array;

	constructor(params: {
		increasing?: boolean | "auto";
		yMin?: number | null;
		yMax?: number | null;
		outOfBounds?: "nan" | "clip" | "raise";
	} = {}) {
		this.increasing = params.increasing ?? true;
		this.yMin = params.yMin ?? null;
		this.yMax = params.yMax ?? null;
		this.outOfBounds = params.outOfBounds ?? "nan";
	}

	fit(x: Float64Array, y: Float64Array, sampleWeight: Float64Array | null = null): this {
		const n = x.length;

		// Sort by x
		const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0));
		const sortedX = new Float64Array(sortedIdx.map((i) => x[i]!));
		const sortedY = new Float64Array(sortedIdx.map((i) => y[i]!));
		const sortedW = sampleWeight ? new Float64Array(sortedIdx.map((i) => sampleWeight[i]!)) : null;

		// Determine direction
		if (this.increasing === "auto") {
			this.increasing_ = checkIncreasing(sortedX, sortedY);
		} else {
			this.increasing_ = this.increasing;
		}

		// Apply PAVA
		const fitted = weightedIsoReg(sortedY, sortedW, this.increasing_);

		// Clip to [yMin, yMax]
		if (this.yMin !== null || this.yMax !== null) {
			for (let i = 0; i < n; i++) {
				if (this.yMin !== null && fitted[i]! < this.yMin) fitted[i] = this.yMin;
				if (this.yMax !== null && fitted[i]! > this.yMax) fitted[i] = this.yMax;
			}
		}

		this.xThresholds_ = sortedX;
		this.yThresholds_ = fitted;
		return this;
	}

	predict(T: Float64Array): Float64Array {
		if (!this.xThresholds_) throw new NotFittedError("IsotonicRegressionExt");
		const n = T.length;
		const result = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			const t = T[i]!;
			const xMin = this.xThresholds_[0]!;
			const xMax = this.xThresholds_[this.xThresholds_.length - 1]!;
			if (t < xMin || t > xMax) {
				if (this.outOfBounds === "clip") {
					result[i] = t < xMin ? this.yThresholds_![0]! : this.yThresholds_![this.yThresholds_.length - 1]!;
				} else {
					result[i] = Number.NaN;
				}
				continue;
			}
			// Linear interpolation
			let lo = 0;
			let hi = this.xThresholds_.length - 1;
			while (lo < hi - 1) {
				const mid = (lo + hi) >> 1;
				if (this.xThresholds_[mid]! <= t) lo = mid;
				else hi = mid;
			}
			const x0 = this.xThresholds_[lo]!;
			const x1 = this.xThresholds_[hi]!;
			const y0 = this.yThresholds_![lo]!;
			const y1 = this.yThresholds_![hi]!;
			const alpha = x1 === x0 ? 0 : (t - x0) / (x1 - x0);
			result[i] = y0 + alpha * (y1 - y0);
		}
		return result;
	}

	score(x: Float64Array, y: Float64Array): number {
		const pred = this.predict(x);
		const yMean = y.reduce((s, v) => s + v, 0) / y.length;
		let ssRes = 0;
		let ssTot = 0;
		for (let i = 0; i < y.length; i++) {
			const p = pred[i]!;
			if (!Number.isNaN(p)) ssRes += (y[i]! - p) ** 2;
			ssTot += (y[i]! - yMean) ** 2;
		}
		return 1 - ssRes / (ssTot || 1);
	}
}
