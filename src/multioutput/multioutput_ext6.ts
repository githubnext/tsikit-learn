/**
 * Multi-output extensions: cross-output regressor and structured prediction.
 * Port of sklearn.multioutput extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Multi-label indicator matrix utilities. */
export function makeMultilabelClassificationData(
	nSamples: number,
	nFeatures: number,
	nClasses: number,
	density = 0.2,
	randomState = 0,
): { X: Float64Array[]; Y: Int32Array[] } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const X: Float64Array[] = Array.from({ length: nSamples }, () => {
		const row = new Float64Array(nFeatures);
		for (let j = 0; j < nFeatures; j++) row[j] = rand() * 2 - 1;
		return row;
	});
	const Y: Int32Array[] = Array.from({ length: nSamples }, () => {
		const labels = new Int32Array(nClasses);
		for (let k = 0; k < nClasses; k++) labels[k] = rand() < density ? 1 : 0;
		return labels;
	});
	return { X, Y };
}

/** Compute example-based F1 score for multi-label classification. */
export function exampleBasedF1(
	yTrue: Int32Array[],
	yPred: Int32Array[],
): number {
	let total = 0;
	for (let i = 0; i < yTrue.length; i++) {
		const t = yTrue[i]!;
		const p = yPred[i]!;
		let tp = 0;
		let tTotal = 0;
		let pTotal = 0;
		for (let k = 0; k < t.length; k++) {
			if ((t[k] ?? 0) === 1) tTotal++;
			if ((p[k] ?? 0) === 1) pTotal++;
			if ((t[k] ?? 0) === 1 && (p[k] ?? 0) === 1) tp++;
		}
		const prec = pTotal === 0 ? 0 : tp / pTotal;
		const rec = tTotal === 0 ? 0 : tp / tTotal;
		total += prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
	}
	return yTrue.length === 0 ? 0 : total / yTrue.length;
}

/** Compute subset accuracy for multi-label classification (exact match). */
export function subsetAccuracy(yTrue: Int32Array[], yPred: Int32Array[]): number {
	let exact = 0;
	for (let i = 0; i < yTrue.length; i++) {
		let match = true;
		const t = yTrue[i]!;
		const p = yPred[i]!;
		for (let k = 0; k < t.length; k++) {
			if ((t[k] ?? 0) !== (p[k] ?? 0)) {
				match = false;
				break;
			}
		}
		if (match) exact++;
	}
	return yTrue.length === 0 ? 0 : exact / yTrue.length;
}

/** Multi-output gradient boosting stub — one tree per output per iteration. */
export class MultiOutputGradientBoostingRegressor {
	private trees_: Array<Array<{ feat: number; thresh: number; lVal: number; rVal: number }>> | null = null;
	private initialPreds_: Float64Array | null = null;
	readonly nEstimators: number;
	readonly learningRate: number;
	readonly nOutputs: number;

	constructor(options: {
		nEstimators?: number;
		learningRate?: number;
		nOutputs: number;
	}) {
		this.nEstimators = options.nEstimators ?? 10;
		this.learningRate = options.learningRate ?? 0.1;
		this.nOutputs = options.nOutputs;
	}

	fit(X: Float64Array[], Y: Float64Array[]): this {
		const n = X.length;
		const nFeatures = X[0]?.length ?? 0;
		this.initialPreds_ = new Float64Array(this.nOutputs);
		for (let out = 0; out < this.nOutputs; out++) {
			let s = 0;
			for (let i = 0; i < n; i++) s += Y[i]?.[out] ?? 0;
			this.initialPreds_[out] = s / n;
		}
		const preds: Float64Array[] = Array.from({ length: n }, () => new Float64Array(this.nOutputs));
		for (let i = 0; i < n; i++) {
			for (let out = 0; out < this.nOutputs; out++) {
				preds[i]![out] = this.initialPreds_[out] ?? 0;
			}
		}
		this.trees_ = [];
		for (let m = 0; m < this.nEstimators; m++) {
			const treesThisIter: Array<{ feat: number; thresh: number; lVal: number; rVal: number }> = [];
			for (let out = 0; out < this.nOutputs; out++) {
				const residuals = new Float64Array(n).map((_, i) => (Y[i]?.[out] ?? 0) - (preds[i]?.[out] ?? 0));
				// Fit stump
				let bestMse = Number.POSITIVE_INFINITY;
				let bestFeat = 0;
				let bestThresh = 0;
				for (let j = 0; j < nFeatures; j++) {
					const vals = Float64Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0);
					const sorted = Float64Array.from(vals).sort();
					for (let k = 0; k < sorted.length - 1; k++) {
						const thresh = ((sorted[k] ?? 0) + (sorted[k + 1] ?? 0)) / 2;
						let lSum = 0;
						let l = 0;
						let rSum = 0;
						let r = 0;
						for (let i = 0; i < n; i++) {
							if ((X[i]?.[j] ?? 0) <= thresh) { l++; lSum += residuals[i] ?? 0; }
							else { r++; rSum += residuals[i] ?? 0; }
						}
						const lMean = l === 0 ? 0 : lSum / l;
						const rMean = r === 0 ? 0 : rSum / r;
						let mse = 0;
						for (let i = 0; i < n; i++) {
							const pred2 = (X[i]?.[j] ?? 0) <= thresh ? lMean : rMean;
							const d = (residuals[i] ?? 0) - pred2;
							mse += d * d;
						}
						if (mse < bestMse) { bestMse = mse; bestFeat = j; bestThresh = thresh; }
					}
				}
				let lSum = 0; let l = 0; let rSum = 0; let r = 0;
				for (let i = 0; i < n; i++) {
					if ((X[i]?.[bestFeat] ?? 0) <= bestThresh) { l++; lSum += residuals[i] ?? 0; }
					else { r++; rSum += residuals[i] ?? 0; }
				}
				const tree = { feat: bestFeat, thresh: bestThresh, lVal: l === 0 ? 0 : lSum / l, rVal: r === 0 ? 0 : rSum / r };
				treesThisIter.push(tree);
				for (let i = 0; i < n; i++) {
					preds[i]![out] += this.learningRate * ((X[i]?.[bestFeat] ?? 0) <= bestThresh ? tree.lVal : tree.rVal);
				}
			}
			this.trees_.push(treesThisIter);
		}
		return this;
	}

	predict(X: Float64Array[]): Float64Array[] {
		if (this.trees_ === null || this.initialPreds_ === null) throw new NotFittedError("MultiOutputGradientBoostingRegressor is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(this.nOutputs);
			for (let o = 0; o < this.nOutputs; o++) out[o] = this.initialPreds_![o] ?? 0;
			for (const treesIter of this.trees_!) {
				for (let o = 0; o < this.nOutputs; o++) {
					const tree = treesIter[o]!;
					out[o] = (out[o] ?? 0) + this.learningRate * ((row[tree.feat] ?? 0) <= tree.thresh ? tree.lVal : tree.rVal);
				}
			}
			return out;
		});
	}
}
