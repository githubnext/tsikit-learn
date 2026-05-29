/**
 * Semi-supervised extensions: TSVM (Transductive SVM), label spreading extensions.
 * Port of sklearn.semi_supervised extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Confidence-based self-training for unlabeled data. */
export class ConfidenceSelfTraining {
	private weights_: Float64Array | null = null;
	private bias_ = 0;
	readonly threshold: number;
	readonly maxIter: number;

	constructor(options: { threshold?: number; maxIter?: number } = {}) {
		this.threshold = options.threshold ?? 0.75;
		this.maxIter = options.maxIter ?? 10;
	}

	fit(
		XLabeled: Float64Array[],
		yLabeled: Int32Array,
		XUnlabeled: Float64Array[],
	): this {
		const nFeatures = XLabeled[0]?.length ?? 0;
		const weights = new Float64Array(nFeatures);
		let bias = 0;

		// Initial training on labeled data
		const trainX = [...XLabeled];
		const trainY = [...yLabeled];

		for (let iter = 0; iter < this.maxIter; iter++) {
			// Train logistic regression
			const lr = 0.01;
			for (let step = 0; step < 200; step++) {
				const grad = new Float64Array(nFeatures);
				let biasGrad = 0;
				for (let i = 0; i < trainX.length; i++) {
					let logit = bias;
					for (let j = 0; j < nFeatures; j++) logit += (weights[j] ?? 0) * (trainX[i]?.[j] ?? 0);
					const pred = 1 / (1 + Math.exp(-logit));
					const err = pred - ((trainY[i] ?? 0) === 1 ? 1 : 0);
					for (let j = 0; j < nFeatures; j++) grad[j]! += err * (trainX[i]?.[j] ?? 0);
					biasGrad += err;
				}
				for (let j = 0; j < nFeatures; j++) weights[j]! -= lr * (grad[j] ?? 0) / trainX.length;
				bias -= lr * biasGrad / trainX.length;
			}
			// Add confident unlabeled predictions
			let added = 0;
			for (const xu of XUnlabeled) {
				let logit = bias;
				for (let j = 0; j < nFeatures; j++) logit += (weights[j] ?? 0) * (xu[j] ?? 0);
				const prob = 1 / (1 + Math.exp(-logit));
				if (prob >= this.threshold) {
					trainX.push(xu);
					trainY.push(1);
					added++;
				} else if (prob <= 1 - this.threshold) {
					trainX.push(xu);
					trainY.push(0);
					added++;
				}
			}
			if (added === 0) break;
		}
		this.weights_ = weights;
		this.bias_ = bias;
		return this;
	}

	predict(X: Float64Array[]): Int32Array {
		if (this.weights_ === null) throw new NotFittedError("ConfidenceSelfTraining is not fitted.");
		return new Int32Array(
			X.map((row) => {
				let logit = this.bias_;
				for (let j = 0; j < (this.weights_?.length ?? 0); j++) {
					logit += (this.weights_![j] ?? 0) * (row[j] ?? 0);
				}
				return logit >= 0 ? 1 : 0;
			}),
		);
	}

	predictProba(X: Float64Array[]): Float64Array[] {
		if (this.weights_ === null) throw new NotFittedError("ConfidenceSelfTraining is not fitted.");
		return X.map((row) => {
			let logit = this.bias_;
			for (let j = 0; j < (this.weights_?.length ?? 0); j++) {
				logit += (this.weights_![j] ?? 0) * (row[j] ?? 0);
			}
			const p = 1 / (1 + Math.exp(-logit));
			return new Float64Array([1 - p, p]);
		});
	}
}

/** Compute graph Laplacian for semi-supervised learning. */
export function computeGraphLaplacian(
	W: Float64Array[],
	normalized = false,
): Float64Array[] {
	const n = W.length;
	const D = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) D[i]! += W[i]?.[j] ?? 0;
	}
	const L: Float64Array[] = Array.from({ length: n }, (_, i) => {
		const row = new Float64Array(n);
		for (let j = 0; j < n; j++) {
			row[j] = i === j ? (D[i] ?? 0) : -(W[i]?.[j] ?? 0);
		}
		return row;
	});
	if (!normalized) return L;
	// Normalized: D^{-1/2} L D^{-1/2}
	const dSqrtInv = new Float64Array(n).map((_, i) => {
		const d = D[i] ?? 0;
		return d > 0 ? 1 / Math.sqrt(d) : 0;
	});
	return L.map((row, i) => {
		const normRow = new Float64Array(n);
		for (let j = 0; j < n; j++) {
			normRow[j] = (row[j] ?? 0) * (dSqrtInv[i] ?? 0) * (dSqrtInv[j] ?? 0);
		}
		return normRow;
	});
}

/** Build k-NN weight graph for semi-supervised learning. */
export function buildKNNGraph(
	X: Float64Array[],
	kNeighbors = 5,
	symmetric = true,
): Float64Array[] {
	const n = X.length;
	const k = Math.min(kNeighbors, n - 1);
	const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
	for (let i = 0; i < n; i++) {
		const dists = Array.from({ length: n }, (_, j) => {
			if (j === i) return { d: Number.POSITIVE_INFINITY, j };
			let d = 0;
			for (let dim = 0; dim < X[0]!.length; dim++) {
				const diff = (X[i]?.[dim] ?? 0) - (X[j]?.[dim] ?? 0);
				d += diff * diff;
			}
			return { d: Math.sqrt(d), j };
		}).sort((a, b) => a.d - b.d);
		for (let ki = 0; ki < k; ki++) {
			const { j, d } = dists[ki]!;
			W[i]![j] = Math.exp(-(d * d));
			if (symmetric) W[j]![i] = Math.exp(-(d * d));
		}
	}
	return W;
}
