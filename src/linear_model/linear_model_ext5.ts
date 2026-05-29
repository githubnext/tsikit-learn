/**
 * Linear model extensions: Tweedie regressor, Huber-like extensions.
 * Port of sklearn.linear_model extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Tweedie Regressor using iteratively reweighted least squares. */
export class TweedieRegressorExt {
	private coef_: Float64Array | null = null;
	private intercept_ = 0;
	readonly power: number;
	readonly alpha: number;
	readonly fitIntercept: boolean;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			power?: number;
			alpha?: number;
			fitIntercept?: boolean;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.power = options.power ?? 0; // 0=Gaussian, 1=Poisson, 2=Gamma
		this.alpha = options.alpha ?? 1.0;
		this.fitIntercept = options.fitIntercept ?? true;
		this.maxIter = options.maxIter ?? 100;
		this.tol = options.tol ?? 1e-4;
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		const weights = new Float64Array(nFeatures);
		let intercept = 0;
		const lr = 0.01;
		for (let iter = 0; iter < this.maxIter; iter++) {
			let maxChange = 0;
			const grad = new Float64Array(nFeatures);
			let biasGrad = 0;
			for (let i = 0; i < n; i++) {
				let eta = intercept;
				for (let j = 0; j < nFeatures; j++) eta += (weights[j] ?? 0) * (X[i]?.[j] ?? 0);
				// Link: log for Poisson/Gamma, identity for Gaussian
				const mu = this.power === 0 ? eta : Math.exp(eta);
				const yTrue = y[i] ?? 0;
				// Gradient of deviance
				let gradFactor: number;
				if (this.power === 0) gradFactor = mu - yTrue; // Gaussian
				else if (this.power === 1) gradFactor = 1 - yTrue / (mu || 1e-10); // Poisson
				else gradFactor = 1 - yTrue / (mu * mu || 1e-10); // Gamma
				for (let j = 0; j < nFeatures; j++) grad[j]! += gradFactor * (X[i]?.[j] ?? 0);
				if (this.fitIntercept) biasGrad += gradFactor;
			}
			// Add L2 regularization
			for (let j = 0; j < nFeatures; j++) {
				const step = lr * ((grad[j] ?? 0) / n + this.alpha * (weights[j] ?? 0) / n);
				maxChange = Math.max(maxChange, Math.abs(step));
				weights[j]! -= step;
			}
			if (this.fitIntercept) intercept -= lr * biasGrad / n;
			if (maxChange < this.tol) break;
		}
		this.coef_ = weights;
		this.intercept_ = intercept;
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("TweedieRegressorExt is not fitted.");
		return new Float64Array(
			X.map((row) => {
				let eta = this.intercept_;
				for (let j = 0; j < (this.coef_?.length ?? 0); j++) eta += (this.coef_![j] ?? 0) * (row[j] ?? 0);
				return this.power === 0 ? eta : Math.exp(eta);
			}),
		);
	}

	get coef(): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("TweedieRegressorExt is not fitted.");
		return this.coef_;
	}
	get intercept(): number { return this.intercept_; }
}

/** Generalized Linear Model base with custom link and distribution. */
export type LinkFunction = "identity" | "log" | "logit" | "probit";

export class GeneralizedLinearModelExt {
	private coef_: Float64Array | null = null;
	private intercept_ = 0;
	readonly link: LinkFunction;
	readonly alpha: number;
	readonly maxIter: number;
	readonly tol: number;

	constructor(
		options: {
			link?: LinkFunction;
			alpha?: number;
			maxIter?: number;
			tol?: number;
		} = {},
	) {
		this.link = options.link ?? "identity";
		this.alpha = options.alpha ?? 0.0;
		this.maxIter = options.maxIter ?? 200;
		this.tol = options.tol ?? 1e-4;
	}

	private applyLink(eta: number): number {
		switch (this.link) {
			case "log": return Math.exp(eta);
			case "logit": return 1 / (1 + Math.exp(-eta));
			case "probit": return normalCDF(eta);
			default: return eta;
		}
	}

	private linkDerivative(mu: number, eta: number): number {
		switch (this.link) {
			case "log": return mu;
			case "logit": return mu * (1 - mu);
			case "probit": return normalPDF(eta);
			default: return 1;
		}
	}

	fit(X: Float64Array[], y: Float64Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		const weights = new Float64Array(nFeatures);
		let intercept = 0;
		const lr = 0.01;
		for (let iter = 0; iter < this.maxIter; iter++) {
			let maxChange = 0;
			const grad = new Float64Array(nFeatures);
			let biasGrad = 0;
			for (let i = 0; i < n; i++) {
				let eta = intercept;
				for (let j = 0; j < nFeatures; j++) eta += (weights[j] ?? 0) * (X[i]?.[j] ?? 0);
				const mu = this.applyLink(eta);
				const dLinkDeta = this.linkDerivative(mu, eta);
				const err = (mu - (y[i] ?? 0)) * dLinkDeta;
				for (let j = 0; j < nFeatures; j++) grad[j]! += err * (X[i]?.[j] ?? 0);
				biasGrad += err;
			}
			for (let j = 0; j < nFeatures; j++) {
				const step = lr * ((grad[j] ?? 0) / n + this.alpha * (weights[j] ?? 0));
				maxChange = Math.max(maxChange, Math.abs(step));
				weights[j]! -= step;
			}
			intercept -= lr * biasGrad / n;
			if (maxChange < this.tol) break;
		}
		this.coef_ = weights;
		this.intercept_ = intercept;
		return this;
	}

	predict(X: Float64Array[]): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("GeneralizedLinearModelExt is not fitted.");
		return new Float64Array(
			X.map((row) => {
				let eta = this.intercept_;
				for (let j = 0; j < (this.coef_?.length ?? 0); j++) eta += (this.coef_![j] ?? 0) * (row[j] ?? 0);
				return this.applyLink(eta);
			}),
		);
	}

	get coef(): Float64Array {
		if (this.coef_ === null) throw new NotFittedError("GeneralizedLinearModelExt is not fitted.");
		return this.coef_;
	}
}

function normalCDF(x: number): number {
	return 0.5 * (1 + erf(x / Math.SQRT2));
}

function normalPDF(x: number): number {
	return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function erf(x: number): number {
	const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
		a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
	const t = 1 / (1 + p * Math.abs(x));
	const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
	return Math.sign(x) * y;
}
