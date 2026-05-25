/**
 * Extended imputer utilities.
 * Port of sklearn.impute extensions (_iterative, experimental)
 */

import { NotFittedError } from "../exceptions.js";

/** Missing value strategies */
export type ImputeStrategy = "mean" | "median" | "most_frequent" | "constant";

/**
 * Imputer with column statistics — extends SimpleImputer.
 * Computes column statistics for later use.
 */
export class StatisticsImputer {
	strategy: ImputeStrategy;
	fillValue: number | string;
	missingValues: number;
	statistics_?: Float64Array;

	constructor(params: {
		strategy?: ImputeStrategy;
		fillValue?: number | string;
		missingValues?: number;
	} = {}) {
		this.strategy = params.strategy ?? "mean";
		this.fillValue = params.fillValue ?? 0;
		this.missingValues = params.missingValues ?? Number.NaN;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const d = X[0]?.length ?? 0;
		this.statistics_ = new Float64Array(d);

		for (let j = 0; j < d; j++) {
			const col = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? Number.NaN)
				.filter((v) => !Number.isNaN(v));

			if (col.length === 0) {
				this.statistics_[j] = typeof this.fillValue === "number" ? this.fillValue : 0;
				continue;
			}

			if (this.strategy === "mean") {
				this.statistics_[j] = col.reduce((s, v) => s + v, 0) / col.length;
			} else if (this.strategy === "median") {
				col.sort((a, b) => a - b);
				const mid = Math.floor(col.length / 2);
				this.statistics_[j] = col.length % 2 === 0 ? ((col[mid - 1]! + col[mid]!) / 2) : col[mid]!;
			} else if (this.strategy === "most_frequent") {
				const counts = new Map<number, number>();
				for (const v of col) counts.set(v, (counts.get(v) ?? 0) + 1);
				let best = col[0]!;
				let bestCnt = 0;
				for (const [v, cnt] of counts) if (cnt > bestCnt) { bestCnt = cnt; best = v; }
				this.statistics_[j] = best;
			} else {
				this.statistics_[j] = typeof this.fillValue === "number" ? this.fillValue : 0;
			}
		}
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.statistics_) throw new NotFittedError("StatisticsImputer");
		const d = this.statistics_.length;
		return X.map((row) => {
			const result = new Float64Array(d);
			for (let j = 0; j < d; j++) {
				const v = row[j] ?? Number.NaN;
				result[j] = (Number.isNaN(v) || v === this.missingValues) ? this.statistics_![j]! : v;
			}
			return result;
		});
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}

/**
 * Experimental iterative imputer (simplified MICE/MissForest variant).
 * Port of sklearn.impute.IterativeImputer (experimental)
 */
export class IterativeImputer {
	maxIter: number;
	tol: number;
	initialStrategy: ImputeStrategy;
	missingValues: number;
	statistics_?: Float64Array;
	imputedMeans_?: Float64Array[][];

	constructor(params: {
		maxIter?: number;
		tol?: number;
		initialStrategy?: ImputeStrategy;
		missingValues?: number;
	} = {}) {
		this.maxIter = params.maxIter ?? 10;
		this.tol = params.tol ?? 1e-3;
		this.initialStrategy = params.initialStrategy ?? "mean";
		this.missingValues = params.missingValues ?? Number.NaN;
	}

	fit(X: Float64Array[]): this {
		this.fitTransform(X);
		return this;
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		const n = X.length;
		const d = X[0]?.length ?? 0;

		// Initial imputation with mean
		const colMeans = new Float64Array(d);
		const counts = new Float64Array(d);
		for (const row of X) {
			for (let j = 0; j < d; j++) {
				const v = row[j] ?? Number.NaN;
				if (!Number.isNaN(v) && v !== this.missingValues) {
					colMeans[j] += v;
					counts[j]++;
				}
			}
		}
		for (let j = 0; j < d; j++) colMeans[j] /= (counts[j] ?? 1) || 1;
		this.statistics_ = colMeans;

		// Initialize imputed matrix
		let imputed = X.map((row) => {
			const result = new Float64Array(d);
			for (let j = 0; j < d; j++) {
				const v = row[j] ?? Number.NaN;
				result[j] = (Number.isNaN(v) || v === this.missingValues) ? colMeans[j]! : v;
			}
			return result;
		});

		// Iterative refinement
		for (let iter = 0; iter < this.maxIter; iter++) {
			const prevImputed = imputed.map((r) => r.slice());
			for (let j = 0; j < d; j++) {
				// Find rows missing feature j
				const missingRows = Array.from({ length: n }, (_, i) => {
					const v = X[i]?.[j] ?? Number.NaN;
					return Number.isNaN(v) || v === this.missingValues;
				});

				if (!missingRows.some(Boolean)) continue;

				// Use other features to predict j via simple linear regression
				const trainIdx = Array.from({ length: n }, (_, i) => i).filter((i) => !missingRows[i]);
				const testIdx = Array.from({ length: n }, (_, i) => i).filter((i) => missingRows[i]);

				if (trainIdx.length === 0) continue;

				// Simple mean prediction using correlated features
				for (const ti of testIdx) {
					let pred = 0;
					let totalWeight = 0;
					for (const si of trainIdx) {
						let sim = 0;
						for (let k = 0; k < d; k++) {
							if (k !== j) sim += Math.abs((imputed[ti]?.[k] ?? 0) - (imputed[si]?.[k] ?? 0));
						}
						const w = 1 / (sim + 1);
						pred += w * (imputed[si]?.[j] ?? 0);
						totalWeight += w;
					}
					imputed[ti]![j] = pred / (totalWeight || 1);
				}
			}

			// Check convergence
			let maxChange = 0;
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < d; j++) {
					maxChange = Math.max(maxChange, Math.abs((imputed[i]?.[j] ?? 0) - (prevImputed[i]?.[j] ?? 0)));
				}
			}
			if (maxChange < this.tol) break;
		}

		return imputed;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.statistics_) throw new NotFittedError("IterativeImputer");
		// Use fitted statistics for transform
		const d = this.statistics_.length;
		return X.map((row) => {
			const result = new Float64Array(d);
			for (let j = 0; j < d; j++) {
				const v = row[j] ?? Number.NaN;
				result[j] = (Number.isNaN(v) || v === this.missingValues) ? this.statistics_![j]! : v;
			}
			return result;
		});
	}
}
