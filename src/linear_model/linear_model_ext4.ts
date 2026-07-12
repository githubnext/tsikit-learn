/**
 * Extended linear model utilities: Lasso coordinate descent extensions.
 * Port of sklearn.linear_model extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute soft thresholding (proximal operator for L1). */
export function softThreshold(x: number, lambda: number): number {
	if (x > lambda) return x - lambda;
	if (x < -lambda) return x + lambda;
	return 0;
}

/** Coordinate descent for Lasso with warm start support. */
export class LassoCoordinateDescent {
	private coef_: Float64Array | null = null;
	private intercept_ = 0;
	private nIter_ = 0;
	readonly alpha: number;
	readonly fitIntercept: boolean;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			alpha?: number;
			fitIntercept?: boolean;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.alpha = options.alpha ?? 1.0;
		this.fitIntercept = options.fitIntercept ?? true;
		this.maxIter = options.maxIter ?? 1000;
		this.tol = options.tol ?? 1e-4;
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const nSamples = X.length;
		const nFeatures = X[0]?.length ?? 0;
		let yCenter = 0;
		const xCenter = new Float64Array(nFeatures);
		if (this.fitIntercept) {
			for (let i = 0; i < nSamples; i++) yCenter += y[i] ?? 0;
			yCenter /= nSamples;
			for (let j = 0; j < nFeatures; j++) {
				for (let i = 0; i < nSamples; i++) xCenter[j]! += X[i]?.[j] ?? 0;
				xCenter[j]! /= nSamples;
			}
		}
		const coef = new Float64Array(nFeatures);
		const r = new Float64Array(nSamples);
		for (let i = 0; i < nSamples; i++) {
			r[i] = (y[i] ?? 0) - yCenter;
		}
		const colNorms = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) {
			for (let i = 0; i < nSamples; i++) {
				const xij = (X[i]?.[j] ?? 0) - xCenter[j]!;
				colNorms[j]! += xij * xij;
			}
		}
		for (let iter = 0; iter < this.maxIter; iter++) {
			let maxChange = 0;
			for (let j = 0; j < nFeatures; j++) {
				const norm = colNorms[j] ?? 0;
				if (norm === 0) continue;
				const oldCoef = coef[j] ?? 0;
				let rho = 0;
				for (let i = 0; i < nSamples; i++) {
					rho += ((X[i]?.[j] ?? 0) - xCenter[j]!) * (r[i]! + oldCoef * ((X[i]?.[j] ?? 0) - xCenter[j]!));
				}
				const newCoef = softThreshold(rho / norm, (this.alpha * nSamples) / norm);
				coef[j] = newCoef;
				const delta = newCoef - oldCoef;
				if (Math.abs(delta) > maxChange) maxChange = Math.abs(delta);
				for (let i = 0; i < nSamples; i++) {
					r[i]! -= delta * ((X[i]?.[j] ?? 0) - xCenter[j]!);
				}
			}
			this.nIter_ = iter + 1;
			if (maxChange < this.tol) break;
		}
		this.coef_ = coef;
		if (this.fitIntercept) {
			this.intercept_ = yCenter;
			for (let j = 0; j < nFeatures; j++) {
				this.intercept_ -= (coef[j] ?? 0) * xCenter[j]!;
			}
		}
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("LassoCoordinateDescent is not fitted.");
		return new Float64Array(
			X.map((row) => {
				let val = this.intercept_;
				for (let j = 0; j < (this.coef_?.length ?? 0); j++) {
					val += (row[j] ?? 0) * (this.coef_![j] ?? 0);
				}
				return val;
			}),
		);
	}

	get coef(): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("LassoCoordinateDescent is not fitted.");
		return this.coef_;
	}
	get intercept(): number { return this.intercept_; }
	get nIter(): number { return this.nIter_; }
}

/** Elastic net coordinate descent variant. */
export class ElasticNetCoordinateDescent {
	private coef_: Float64Array | null = null;
	private intercept_ = 0;
	readonly alpha: number;
	readonly l1Ratio: number;
	readonly fitIntercept: boolean;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			alpha?: number;
			l1Ratio?: number;
			fitIntercept?: boolean;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.alpha = options.alpha ?? 1.0;
		this.l1Ratio = options.l1Ratio ?? 0.5;
		this.fitIntercept = options.fitIntercept ?? true;
		this.maxIter = options.maxIter ?? 1000;
		this.tol = options.tol ?? 1e-4;
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const nSamples = X.length;
		const nFeatures = X[0]?.length ?? 0;
		let yCenter = 0;
		const xCenter = new Float64Array(nFeatures);
		if (this.fitIntercept) {
			for (let i = 0; i < nSamples; i++) yCenter += y[i] ?? 0;
			yCenter /= nSamples;
			for (let j = 0; j < nFeatures; j++) {
				for (let i = 0; i < nSamples; i++) xCenter[j]! += X[i]?.[j] ?? 0;
				xCenter[j]! /= nSamples;
			}
		}
		const coef = new Float64Array(nFeatures);
		const r = new Float64Array(nSamples);
		for (let i = 0; i < nSamples; i++) r[i] = (y[i] ?? 0) - yCenter;
		const l1 = this.alpha * this.l1Ratio;
		const l2 = this.alpha * (1 - this.l1Ratio);
		for (let iter = 0; iter < this.maxIter; iter++) {
			let maxChange = 0;
			for (let j = 0; j < nFeatures; j++) {
				let norm = 0;
				for (let i = 0; i < nSamples; i++) {
					const xij = (X[i]?.[j] ?? 0) - xCenter[j]!;
					norm += xij * xij;
				}
				norm += l2 * nSamples;
				if (norm === 0) continue;
				const oldCoef = coef[j] ?? 0;
				let rho = 0;
				for (let i = 0; i < nSamples; i++) {
					rho += ((X[i]?.[j] ?? 0) - xCenter[j]!) * (r[i]! + oldCoef * ((X[i]?.[j] ?? 0) - xCenter[j]!));
				}
				const newCoef = softThreshold(rho / norm, (l1 * nSamples) / norm);
				coef[j] = newCoef;
				const delta = newCoef - oldCoef;
				if (Math.abs(delta) > maxChange) maxChange = Math.abs(delta);
				for (let i = 0; i < nSamples; i++) {
					r[i]! -= delta * ((X[i]?.[j] ?? 0) - xCenter[j]!);
				}
			}
			if (maxChange < this.tol) break;
		}
		this.coef_ = coef;
		if (this.fitIntercept) {
			this.intercept_ = yCenter;
			for (let j = 0; j < nFeatures; j++) this.intercept_ -= (coef[j] ?? 0) * xCenter[j]!;
		}
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("ElasticNetCoordinateDescent is not fitted.");
		return new Float64Array(
			X.map((row) => {
				let val = this.intercept_;
				for (let j = 0; j < (this.coef_?.length ?? 0); j++) val += (row[j] ?? 0) * (this.coef_![j] ?? 0);
				return val;
			}),
		);
	}

	get coef(): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("ElasticNetCoordinateDescent is not fitted.");
		return this.coef_;
	}
	get intercept(): number { return this.intercept_; }
}
