/**
 * Extended Naive Bayes classifiers.
 * Port of sklearn.naive_bayes (ComplementNB, CategoricalNB extensions)
 */

import { NotFittedError } from "../exceptions.js";

export interface ComplementNBParams {
	alpha?: number;
	fitPrior?: boolean;
	classPrior?: Float64Array | null;
	norm?: boolean;
}

/**
 * Complement Naive Bayes classifier.
 * Port of sklearn.naive_bayes.ComplementNB
 * Better for imbalanced datasets than MultinomialNB.
 */
export class ComplementNB {
	alpha: number;
	fitPrior: boolean;
	classPrior: Float64Array | null;
	norm: boolean;

	classes_?: Int32Array;
	classPrior_?: Float64Array;
	classCount_?: Float64Array;
	featureCount_?: Float64Array[];
	featureLogProb_?: Float64Array[];

	constructor(params: ComplementNBParams = {}) {
		this.alpha = params.alpha ?? 1.0;
		this.fitPrior = params.fitPrior ?? true;
		this.classPrior = params.classPrior ?? null;
		this.norm = params.norm ?? false;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
		this.classes_ = new Int32Array(classSet);
		const nClasses = classSet.length;

		this.classCount_ = new Float64Array(nClasses);
		this.featureCount_ = Array.from({ length: nClasses }, () => new Float64Array(nFeatures));

		for (let i = 0; i < n; i++) {
			const ci = classSet.indexOf(y[i]!);
			this.classCount_[ci]++;
			for (let j = 0; j < nFeatures; j++) {
				this.featureCount_[ci]![j] += X[i]?.[j] ?? 0;
			}
		}

		// Complement counts: sum over all OTHER classes
		const totalFeatureCount = new Float64Array(nFeatures);
		for (let ci = 0; ci < nClasses; ci++) {
			for (let j = 0; j < nFeatures; j++) totalFeatureCount[j] += this.featureCount_[ci]![j] ?? 0;
		}

		this.featureLogProb_ = Array.from({ length: nClasses }, (_, ci) => {
			const complementCount = new Float64Array(nFeatures);
			for (let j = 0; j < nFeatures; j++) {
				complementCount[j] = totalFeatureCount[j]! - (this.featureCount_![ci]?.[j] ?? 0) + this.alpha;
			}
			let complementSum = 0;
			for (const c of complementCount) complementSum += c;
			const logProb = new Float64Array(nFeatures);
			for (let j = 0; j < nFeatures; j++) {
				logProb[j] = Math.log(complementCount[j]! / complementSum);
			}
			if (this.norm) {
				let norm = 0;
				for (const lp of logProb) norm += lp ** 2;
				norm = Math.sqrt(norm);
				for (let j = 0; j < nFeatures; j++) logProb[j]! /= norm || 1;
			}
			return logProb;
		});

		if (this.classPrior) {
			this.classPrior_ = this.classPrior;
		} else if (this.fitPrior) {
			this.classPrior_ = new Float64Array(nClasses);
			for (let ci = 0; ci < nClasses; ci++) {
				this.classPrior_[ci] = Math.log((this.classCount_[ci] ?? 0) / n);
			}
		} else {
			this.classPrior_ = new Float64Array(nClasses).fill(-Math.log(nClasses));
		}
		return this;
	}

	predictLogProba(X: Float64Array[]): Float64Array[] {
		if (!this.classes_) throw new NotFittedError("ComplementNB");
		const nClasses = this.classes_.length;
		return X.map((x) => {
			const logLiks = new Float64Array(nClasses);
			for (let ci = 0; ci < nClasses; ci++) {
				let ll = this.classPrior_![ci]!;
				const logProb = this.featureLogProb_![ci]!;
				for (let j = 0; j < x.length; j++) {
					// Complement NB negates the class-specific log probs
					ll -= (x[j] ?? 0) * (logProb[j] ?? 0);
				}
				logLiks[ci] = ll;
			}
			// Normalize
			const maxLL = Math.max(...logLiks);
			const expLLs = logLiks.map((ll) => Math.exp(ll - maxLL));
			const sum = expLLs.reduce((s, v) => s + v, 0);
			return new Float64Array(expLLs.map((v) => Math.log(v / sum)));
		});
	}

	predict(X: Float64Array[]): Int32Array {
		const logProbas = this.predictLogProba(X);
		return new Int32Array(logProbas.map((lp) => {
			let best = 0;
			for (let ci = 1; ci < lp.length; ci++) {
				if ((lp[ci] ?? -Infinity) > (lp[best] ?? -Infinity)) best = ci;
			}
			return this.classes_![best]!;
		}));
	}

	score(X: Float64Array[], y: Int32Array): number {
		const pred = this.predict(X);
		let correct = 0;
		for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
		return correct / y.length;
	}
}

/**
 * Naive Bayes for multivariate Bernoulli models.
 * Port of sklearn.naive_bayes.BernoulliNB
 */
export class BernoulliNB {
	alpha: number;
	fitPrior: boolean;
	classPrior: Float64Array | null;
	binarize: number | null;

	classes_?: Int32Array;
	classPrior_?: Float64Array;
	classCount_?: Float64Array;
	featureCount_?: Float64Array[];
	featureLogProb_?: Float64Array[];
	featureLogNegProb_?: Float64Array[];

	constructor(params: { alpha?: number; fitPrior?: boolean; classPrior?: Float64Array | null; binarize?: number | null } = {}) {
		this.alpha = params.alpha ?? 1.0;
		this.fitPrior = params.fitPrior ?? true;
		this.classPrior = params.classPrior ?? null;
		this.binarize = params.binarize ?? 0.0;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
		this.classes_ = new Int32Array(classSet);
		const nClasses = classSet.length;

		this.classCount_ = new Float64Array(nClasses);
		this.featureCount_ = Array.from({ length: nClasses }, () => new Float64Array(nFeatures));

		const threshold = this.binarize;
		for (let i = 0; i < n; i++) {
			const ci = classSet.indexOf(y[i]!);
			this.classCount_[ci]++;
			for (let j = 0; j < nFeatures; j++) {
				const val = threshold !== null ? ((X[i]?.[j] ?? 0) > threshold ? 1 : 0) : (X[i]?.[j] ?? 0);
				this.featureCount_[ci]![j] += val;
			}
		}

		this.featureLogProb_ = [];
		this.featureLogNegProb_ = [];
		for (let ci = 0; ci < nClasses; ci++) {
			const cnt = this.classCount_[ci]!;
			const fp = new Float64Array(nFeatures);
			const fnp = new Float64Array(nFeatures);
			for (let j = 0; j < nFeatures; j++) {
				const c = (this.featureCount_[ci]?.[j] ?? 0) + this.alpha;
				const total = cnt + 2 * this.alpha;
				fp[j] = Math.log(c / total);
				fnp[j] = Math.log((total - c) / total);
			}
			this.featureLogProb_.push(fp);
			this.featureLogNegProb_.push(fnp);
		}

		if (this.classPrior) {
			this.classPrior_ = this.classPrior;
		} else if (this.fitPrior) {
			this.classPrior_ = new Float64Array(nClasses);
			for (let ci = 0; ci < nClasses; ci++) {
				this.classPrior_[ci] = Math.log((this.classCount_[ci] ?? 0) / n);
			}
		} else {
			this.classPrior_ = new Float64Array(nClasses).fill(-Math.log(nClasses));
		}
		return this;
	}

	predict(X: Float64Array[]): Int32Array {
		if (!this.classes_) throw new NotFittedError("BernoulliNB");
		const nClasses = this.classes_.length;
		return new Int32Array(X.map((x) => {
			let bestCi = 0;
			let bestLL = -Number.POSITIVE_INFINITY;
			for (let ci = 0; ci < nClasses; ci++) {
				let ll = this.classPrior_![ci]!;
				const fp = this.featureLogProb_![ci]!;
				const fnp = this.featureLogNegProb_![ci]!;
				for (let j = 0; j < x.length; j++) {
					const threshold = this.binarize;
					const val = threshold !== null ? ((x[j] ?? 0) > threshold ? 1 : 0) : (x[j] ?? 0);
					ll += val * (fp[j] ?? 0) + (1 - val) * (fnp[j] ?? 0);
				}
				if (ll > bestLL) { bestLL = ll; bestCi = ci; }
			}
			return this.classes_![bestCi]!;
		}));
	}

	score(X: Float64Array[], y: Int32Array): number {
		const pred = this.predict(X);
		let correct = 0;
		for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
		return correct / y.length;
	}
}
