/**
 * Imputation extensions: IterativeImputer extensions, KNN imputation utilities.
 * Port of sklearn.impute extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Median imputer for robust imputation. */
export class MedianImputer {
	private medians_: Float64Array | null = null;
	readonly missingValues: number;

	constructor(options: { missingValues?: number } = {}) {
		this.missingValues = options.missingValues ?? Number.NaN;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const medians = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) {
			const vals: number[] = [];
			for (const row of X) {
				const v = row[j] ?? 0;
				if (!this.isMissing(v)) vals.push(v);
			}
			vals.sort((a, b) => a - b);
			if (vals.length === 0) {
				medians[j] = 0;
			} else if (vals.length % 2 === 0) {
				medians[j] = ((vals[vals.length / 2 - 1] ?? 0) + (vals[vals.length / 2] ?? 0)) / 2;
			} else {
				medians[j] = vals[Math.floor(vals.length / 2)] ?? 0;
			}
		}
		this.medians_ = medians;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.medians_ === null) throw new NotFittedError("MedianImputer is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(row.length);
			for (let j = 0; j < row.length; j++) {
				const v = row[j] ?? 0;
				out[j] = this.isMissing(v) ? (this.medians_![j] ?? 0) : v;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	private isMissing(v: number): boolean {
		return Number.isNaN(this.missingValues) ? Number.isNaN(v) : v === this.missingValues;
	}
}

/** Most frequent value imputer for categorical features. */
export class MostFrequentImputer {
	private mostFrequent_: Float64Array | null = null;
	readonly missingValues: number;

	constructor(options: { missingValues?: number } = {}) {
		this.missingValues = options.missingValues ?? Number.NaN;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const mostFrequent = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) {
			const counts = new Map<number, number>();
			for (const row of X) {
				const v = row[j] ?? 0;
				if (!this.isMissing(v)) counts.set(v, (counts.get(v) ?? 0) + 1);
			}
			let bestVal = 0;
			let bestCount = 0;
			for (const [val, count] of counts) {
				if (count > bestCount) {
					bestCount = count;
					bestVal = val;
				}
			}
			mostFrequent[j] = bestVal;
		}
		this.mostFrequent_ = mostFrequent;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.mostFrequent_ === null) throw new NotFittedError("MostFrequentImputer is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(row.length);
			for (let j = 0; j < row.length; j++) {
				const v = row[j] ?? 0;
				out[j] = this.isMissing(v) ? (this.mostFrequent_![j] ?? 0) : v;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	private isMissing(v: number): boolean {
		return Number.isNaN(this.missingValues) ? Number.isNaN(v) : v === this.missingValues;
	}
}

/** Constant imputer (fill missing values with a constant). */
export class ConstantImputer {
	private fitted_ = false;
	readonly fillValue: number;
	readonly missingValues: number;

	constructor(options: { fillValue?: number; missingValues?: number } = {}) {
		this.fillValue = options.fillValue ?? 0;
		this.missingValues = options.missingValues ?? Number.NaN;
	}

	fit(_X: Float64Array[]): this {
		this.fitted_ = true;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.fitted_) throw new NotFittedError("ConstantImputer is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(row.length);
			for (let j = 0; j < row.length; j++) {
				const v = row[j] ?? 0;
				out[j] = this.isMissing(v) ? this.fillValue : v;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	private isMissing(v: number): boolean {
		return Number.isNaN(this.missingValues) ? Number.isNaN(v) : v === this.missingValues;
	}
}

/** Add missing value indicator columns. */
export class MissingIndicatorExt {
	private featureIndices_: Int32Array | null = null;
	readonly features: "all" | "missing-only";
	readonly missingValues: number;

	constructor(
		options: {
			features?: "all" | "missing-only";
			missingValues?: number;
		} = {},
	) {
		this.features = options.features ?? "missing-only";
		this.missingValues = options.missingValues ?? Number.NaN;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		if (this.features === "all") {
			this.featureIndices_ = new Int32Array(Array.from({ length: nFeatures }, (_, i) => i));
		} else {
			const hasMissing: number[] = [];
			for (let j = 0; j < nFeatures; j++) {
				for (const row of X) {
					if (this.isMissing(row[j] ?? 0)) {
						hasMissing.push(j);
						break;
					}
				}
			}
			this.featureIndices_ = new Int32Array(hasMissing);
		}
		return this;
	}

	transform(X: Float64Array[]): Int32Array[] {
		if (this.featureIndices_ === null) throw new NotFittedError("MissingIndicatorExt is not fitted.");
		return X.map((row) => {
			const out = new Int32Array(this.featureIndices_!.length);
			for (let k = 0; k < this.featureIndices_!.length; k++) {
				out[k] = this.isMissing(row[this.featureIndices_![k]!] ?? 0) ? 1 : 0;
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[]): Int32Array[] {
		return this.fit(X).transform(X);
	}

	private isMissing(v: number): boolean {
		return Number.isNaN(this.missingValues) ? Number.isNaN(v) : v === this.missingValues;
	}
}
