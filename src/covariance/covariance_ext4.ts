/**
 * Covariance extensions: OAS (Oracle Approximating Shrinkage), POET.
 * Port of sklearn.covariance extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Oracle Approximating Shrinkage (OAS) estimator. */
export class OASShrinkage {
	private covariance_: Float64Array[] | null = null;
	private precision_: Float64Array[] | null = null;
	private shrinkage_: number | null = null;

	fit(X: Float64Array[]): this {
		const n = X.length;
		const p = X[0]?.length ?? 0;

		const mean = new Float64Array(p);
		for (const row of X) for (let j = 0; j < p; j++) mean[j]! += row[j] ?? 0;
		for (let j = 0; j < p; j++) mean[j]! /= n;

		const S = Array.from({ length: p }, () => new Float64Array(p));
		for (const row of X) {
			for (let a = 0; a < p; a++) {
				for (let b = 0; b < p; b++) {
					S[a]![b]! += ((row[a] ?? 0) - (mean[a] ?? 0)) * ((row[b] ?? 0) - (mean[b] ?? 0));
				}
			}
		}
		for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) S[a]![b]! /= n;

		// Trace and Frobenius norm
		let trS = 0;
		let trS2 = 0;
		for (let a = 0; a < p; a++) trS += S[a]![a] ?? 0;
		for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) trS2 += (S[a]![b] ?? 0) ** 2;

		// OAS shrinkage coefficient
		const num = (1 - 2 / p) * trS2 + trS * trS;
		const den = (n + 1 - 2 / p) * (trS2 - (trS * trS) / p);
		const rho = den === 0 ? 1 : Math.min(1, num / den);
		this.shrinkage_ = rho;

		const mu = trS / p;
		this.covariance_ = Array.from({ length: p }, (_, a) => {
			const row = new Float64Array(p);
			for (let b = 0; b < p; b++) {
				row[b] = (1 - rho) * (S[a]![b] ?? 0) + (a === b ? rho * mu : 0);
			}
			return row;
		});
		this.precision_ = invertMatrix(this.covariance_);
		return this;
	}

	get covariance(): Float64Array[] {
		if (this.covariance_ === null) throw new NotFittedError("OASShrinkage is not fitted.");
		return this.covariance_;
	}

	get precision(): Float64Array[] {
		if (this.precision_ === null) throw new NotFittedError("OASShrinkage is not fitted.");
		return this.precision_;
	}

	get shrinkage(): number {
		if (this.shrinkage_ === null) throw new NotFittedError("OASShrinkage is not fitted.");
		return this.shrinkage_;
	}
}

/** Compute log-likelihood of data under a covariance model. */
export function gaussianLogLikelihood(
	X: Float64Array[],
	mean: Float64Array,
	precision: Float64Array[],
): number {
	const n = X.length;
	const p = mean.length;
	// log det via Cholesky (simplified: use product of diagonal after LU)
	let logDet = 0;
	for (let j = 0; j < p; j++) logDet += Math.log(Math.abs(precision[j]?.[j] ?? 1));
	let logLik = (n * (logDet - p * Math.log(2 * Math.PI))) / 2;
	for (const row of X) {
		const diff = new Float64Array(p).map((_, j) => (row[j] ?? 0) - (mean[j] ?? 0));
		let quad = 0;
		for (let a = 0; a < p; a++) {
			let pda = 0;
			for (let b = 0; b < p; b++) pda += (precision[a]?.[b] ?? 0) * (diff[b] ?? 0);
			quad += (diff[a] ?? 0) * pda;
		}
		logLik -= quad / 2;
	}
	return logLik;
}

/** Covariance matrix cross-validation scoring (log-likelihood based). */
export function covarianceCVScore(
	X: Float64Array[],
	estimator: { fit: (X: Float64Array[]) => unknown; covariance: Float64Array[] },
	nFolds = 5,
): number {
	const n = X.length;
	const p = X[0]?.length ?? 0;
	const foldSize = Math.floor(n / nFolds);
	let totalScore = 0;
	for (let fold = 0; fold < nFolds; fold++) {
		const testStart = fold * foldSize;
		const testEnd = fold === nFolds - 1 ? n : testStart + foldSize;
		const trainX = X.filter((_, i) => i < testStart || i >= testEnd);
		const testX = X.slice(testStart, testEnd);
		estimator.fit(trainX);
		const cov = estimator.covariance;
		const mean = new Float64Array(p);
		for (const row of trainX) for (let j = 0; j < p; j++) mean[j]! += row[j] ?? 0;
		for (let j = 0; j < p; j++) mean[j]! /= trainX.length;
		// Score: negative log-likelihood
		let score = 0;
		for (const row of testX) {
			let quadForm = 0;
			for (let a = 0; a < p; a++) {
				let covDotDiff = 0;
				for (let b = 0; b < p; b++) {
					covDotDiff += (cov[a]?.[b] ?? 0) * ((row[b] ?? 0) - (mean[b] ?? 0));
				}
				quadForm += ((row[a] ?? 0) - (mean[a] ?? 0)) * covDotDiff;
			}
			score -= quadForm;
		}
		totalScore += score / testX.length;
	}
	return totalScore / nFolds;
}

function invertMatrix(A: Float64Array[]): Float64Array[] {
	const n = A.length;
	const aug = A.map((row, i) => {
		const r = new Float64Array(2 * n);
		for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
		r[n + i] = 1;
		return r;
	});
	for (let col = 0; col < n; col++) {
		let maxRow = col;
		for (let row = col + 1; row < n; row++) {
			if (Math.abs(aug[row]?.[col] ?? 0) > Math.abs(aug[maxRow]?.[col] ?? 0)) maxRow = row;
		}
		const tmp = aug[col]!;
		aug[col] = aug[maxRow]!;
		aug[maxRow] = tmp;
		const pivot = aug[col]?.[col] ?? 1;
		if (Math.abs(pivot) < 1e-12) continue;
		for (let j = 0; j < 2 * n; j++) aug[col]![j]! /= pivot;
		for (let row = 0; row < n; row++) {
			if (row === col) continue;
			const f = aug[row]?.[col] ?? 0;
			for (let j = 0; j < 2 * n; j++) aug[row]![j]! -= f * (aug[col]?.[j] ?? 0);
		}
	}
	return aug.map((row) => new Float64Array(row.slice(n)));
}
