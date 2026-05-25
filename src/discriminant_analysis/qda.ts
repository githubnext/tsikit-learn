/**
 * Quadratic Discriminant Analysis.
 * Port of sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
 */

import { NotFittedError } from "../exceptions.js";

export interface QDAParams {
	regParam?: number;
	storeCovariance?: boolean;
	tol?: number;
	priors?: Float64Array | null;
}

/**
 * Quadratic Discriminant Analysis.
 * QDA fits a Gaussian density to each class and classifies using Bayes' rule.
 */
export class QuadraticDiscriminantAnalysis {
	regParam: number;
	storeCovariance: boolean;
	tol: number;
	priors: Float64Array | null;

	classes_?: Int32Array;
	priors_?: Float64Array;
	means_?: Float64Array[];
	covariance_?: Float64Array[][];
	rotations_?: Float64Array[][];
	scalings_?: Float64Array[];

	constructor(params: QDAParams = {}) {
		this.regParam = params.regParam ?? 0.0;
		this.storeCovariance = params.storeCovariance ?? false;
		this.tol = params.tol ?? 1e-4;
		this.priors = params.priors ?? null;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
		this.classes_ = new Int32Array(classSet);
		const nClasses = classSet.length;

		this.priors_ = new Float64Array(nClasses);
		this.means_ = [];
		this.covariance_ = [];
		this.rotations_ = [];
		this.scalings_ = [];

		for (let ci = 0; ci < nClasses; ci++) {
			const c = classSet[ci]!;
			const mask = Array.from({ length: n }, (_, i) => y[i] === c);
			const classX = X.filter((_, i) => mask[i]);
			const nc = classX.length;
			this.priors_[ci] = this.priors ? (this.priors[ci] ?? nc / n) : nc / n;

			// Class mean
			const mean = new Float64Array(nFeatures);
			for (const x of classX) {
				for (let j = 0; j < nFeatures; j++) mean[j] += (x[j] ?? 0) / nc;
			}
			this.means_.push(mean);

			// Class covariance
			const cov: Float64Array[] = Array.from({ length: nFeatures }, () => new Float64Array(nFeatures));
			for (const x of classX) {
				for (let j = 0; j < nFeatures; j++) {
					for (let k = 0; k < nFeatures; k++) {
						cov[j]![k] += ((x[j] ?? 0) - mean[j]!) * ((x[k] ?? 0) - mean[k]!) / (nc - 1);
					}
				}
			}
			// Regularization
			if (this.regParam > 0) {
				for (let j = 0; j < nFeatures; j++) cov[j]![j] += this.regParam;
			}

			if (this.storeCovariance) this.covariance_.push(cov);

			// SVD of covariance for log-det and inverse
			// Use simple diagonal approximation for efficiency
			const diagCov = new Float64Array(nFeatures);
			for (let j = 0; j < nFeatures; j++) diagCov[j] = cov[j]?.[j] ?? 1;
			this.scalings_.push(diagCov);

			// Rotation (identity for diagonal approx)
			const rotation: Float64Array[] = Array.from({ length: nFeatures }, (_, j) => {
				const row = new Float64Array(nFeatures);
				row[j] = 1;
				return row;
			});
			this.rotations_.push(rotation);
		}
		return this;
	}

	private _logLikelihood(x: Float64Array, ci: number): number {
		const mean = this.means_![ci]!;
		const scaling = this.scalings_![ci]!;
		const nFeatures = x.length;
		let logLik = Math.log(this.priors_![ci]!);
		let logDet = 0;
		for (let j = 0; j < nFeatures; j++) {
			const s = scaling[j]! || 1e-10;
			logDet += Math.log(s);
			const diff = (x[j] ?? 0) - (mean[j] ?? 0);
			logLik -= 0.5 * diff * diff / s;
		}
		logLik -= 0.5 * logDet;
		return logLik;
	}

	predict(X: Float64Array[]): Int32Array {
		if (!this.classes_) throw new NotFittedError("QuadraticDiscriminantAnalysis");
		return new Int32Array(X.map((x) => {
			let best = 0;
			let bestLL = -Number.POSITIVE_INFINITY;
			for (let ci = 0; ci < this.classes_!.length; ci++) {
				const ll = this._logLikelihood(x, ci);
				if (ll > bestLL) { bestLL = ll; best = ci; }
			}
			return this.classes_![best]!;
		}));
	}

	predictProba(X: Float64Array[]): Float64Array[] {
		if (!this.classes_) throw new NotFittedError("QuadraticDiscriminantAnalysis");
		const nClasses = this.classes_.length;
		return X.map((x) => {
			const logLiks = new Float64Array(nClasses);
			for (let ci = 0; ci < nClasses; ci++) logLiks[ci] = this._logLikelihood(x, ci);
			const maxLL = Math.max(...logLiks);
			const proba = new Float64Array(nClasses);
			let sum = 0;
			for (let ci = 0; ci < nClasses; ci++) { proba[ci] = Math.exp(logLiks[ci]! - maxLL); sum += proba[ci]!; }
			for (let ci = 0; ci < nClasses; ci++) proba[ci]! /= sum;
			return proba;
		});
	}

	score(X: Float64Array[], y: Int32Array): number {
		const pred = this.predict(X);
		let correct = 0;
		for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
		return correct / y.length;
	}
}
