/**
 * FDR/FPR-based feature selection.
 * Port of sklearn.feature_selection._univariate_selection (fdr, fpr, fwe)
 */

import { NotFittedError } from "../exceptions.js";

function betaInc(a: number, b: number, x: number): number {
	// Simple regularized incomplete beta function approximation
	if (x <= 0) return 0;
	if (x >= 1) return 1;
	// Continued fraction approximation (Numerical Recipes)
	const lnBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
	const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnBeta) / a;
	// Lentz's algorithm
	let result = 0;
	for (let i = 0; i < 200; i++) {
		const m = i >> 1;
		let d: number;
		if (i === 0) d = 1;
		else if (i % 2 === 0) d = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
		else d = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
		result += d;
	}
	return front * result;
}

function lgamma(z: number): number {
	// Stirling approximation
	const g = 7;
	const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
		-176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
	if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - lgamma(1 - z);
	z -= 1;
	let x = c[0]!;
	for (let i = 1; i < g + 2; i++) x += c[i]! / (z + i);
	const t = z + g + 0.5;
	return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function fPValue(f: number, dfNum: number, dfDen: number): number {
	if (f <= 0) return 1;
	// P(F > f) using regularized incomplete beta
	const x = dfDen / (dfDen + dfNum * f);
	return betaInc(dfDen / 2, dfNum / 2, x);
}

/** Compute F-statistic and p-values for classification */
export function fClassif(
	X: Float64Array[],
	y: Int32Array,
): { fStats: Float64Array; pValues: Float64Array } {
	const n = X.length;
	const nFeatures = X[0]?.length ?? 0;
	const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
	const k = classes.length;

	const fStats = new Float64Array(nFeatures);
	const pValues = new Float64Array(nFeatures);

	for (let j = 0; j < nFeatures; j++) {
		const globalMean = Array.from(X).reduce((s, x) => s + (x[j] ?? 0), 0) / n;
		let ssBetween = 0;
		let ssWithin = 0;

		for (const c of classes) {
			const classX = X.filter((_, i) => y[i] === c).map((x) => x[j] ?? 0);
			const nc = classX.length;
			const classMean = classX.reduce((s, v) => s + v, 0) / nc;
			ssBetween += nc * (classMean - globalMean) ** 2;
			ssWithin += classX.reduce((s, v) => s + (v - classMean) ** 2, 0);
		}

		const dfBetween = k - 1;
		const dfWithin = n - k;
		const f = (ssBetween / dfBetween) / (ssWithin / dfWithin + 1e-10);
		fStats[j] = f;
		pValues[j] = fPValue(f, dfBetween, dfWithin);
	}

	return { fStats, pValues };
}

/** Select features based on False Discovery Rate (FDR) */
export class SelectFdr {
	alpha: number;
	scoreFunc: "f_classif" | "chi2";
	supportMask_?: boolean[];
	pValues_?: Float64Array;
	scores_?: Float64Array;

	constructor(alpha = 0.05, scoreFunc: "f_classif" | "chi2" = "f_classif") {
		this.alpha = alpha;
		this.scoreFunc = scoreFunc;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const nFeatures = X[0]?.length ?? 0;
		const { fStats, pValues } = fClassif(X, y);
		this.scores_ = fStats;
		this.pValues_ = pValues;

		// Benjamini-Hochberg FDR procedure
		const sortedIdx = Array.from({ length: nFeatures }, (_, i) => i)
			.sort((a, b) => (pValues[a] ?? 1) - (pValues[b] ?? 1));

		let threshold = 0;
		for (let k = 0; k < nFeatures; k++) {
			const fdrThreshold = this.alpha * (k + 1) / nFeatures;
			if ((pValues[sortedIdx[k]!] ?? 1) <= fdrThreshold) {
				threshold = pValues[sortedIdx[k]!]!;
			}
		}

		this.supportMask_ = Array.from({ length: nFeatures }, (_, j) => (pValues[j] ?? 1) <= threshold);
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.supportMask_) throw new NotFittedError("SelectFdr");
		return X.map((x) => {
			const result: number[] = [];
			for (let j = 0; j < x.length; j++) if (this.supportMask_![j]) result.push(x[j] ?? 0);
			return new Float64Array(result);
		});
	}

	fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}

	get support_(): boolean[] {
		if (!this.supportMask_) throw new NotFittedError("SelectFdr");
		return this.supportMask_;
	}
}

/** Select features based on False Positive Rate (FPR) */
export class SelectFpr {
	alpha: number;
	scoreFunc: "f_classif" | "chi2";
	supportMask_?: boolean[];
	pValues_?: Float64Array;
	scores_?: Float64Array;

	constructor(alpha = 0.05, scoreFunc: "f_classif" | "chi2" = "f_classif") {
		this.alpha = alpha;
		this.scoreFunc = scoreFunc;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const nFeatures = X[0]?.length ?? 0;
		const { fStats, pValues } = fClassif(X, y);
		this.scores_ = fStats;
		this.pValues_ = pValues;
		this.supportMask_ = Array.from({ length: nFeatures }, (_, j) => (pValues[j] ?? 1) <= this.alpha);
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.supportMask_) throw new NotFittedError("SelectFpr");
		return X.map((x) => {
			const result: number[] = [];
			for (let j = 0; j < x.length; j++) if (this.supportMask_![j]) result.push(x[j] ?? 0);
			return new Float64Array(result);
		});
	}

	fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}
}

/** Select features based on Family-Wise Error Rate (FWE) — Bonferroni correction */
export class SelectFwe {
	alpha: number;
	scoreFunc: "f_classif" | "chi2";
	supportMask_?: boolean[];
	pValues_?: Float64Array;
	scores_?: Float64Array;

	constructor(alpha = 0.05, scoreFunc: "f_classif" | "chi2" = "f_classif") {
		this.alpha = alpha;
		this.scoreFunc = scoreFunc;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const nFeatures = X[0]?.length ?? 0;
		const { fStats, pValues } = fClassif(X, y);
		this.scores_ = fStats;
		this.pValues_ = pValues;
		// Bonferroni correction
		const bonferroniAlpha = this.alpha / nFeatures;
		this.supportMask_ = Array.from({ length: nFeatures }, (_, j) => (pValues[j] ?? 1) <= bonferroniAlpha);
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.supportMask_) throw new NotFittedError("SelectFwe");
		return X.map((x) => {
			const result: number[] = [];
			for (let j = 0; j < x.length; j++) if (this.supportMask_![j]) result.push(x[j] ?? 0);
			return new Float64Array(result);
		});
	}

	fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}
}
