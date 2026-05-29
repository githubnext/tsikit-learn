/**
 * Neural network extensions: Transformer-inspired layers, attention mechanism.
 * Port of sklearn.neural_network extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Scaled dot-product attention. */
export function scaledDotProductAttention(
	Q: Float64Array[],
	K: Float64Array[],
	V: Float64Array[],
): Float64Array[] {
	const dK = Q[0]?.length ?? 1;
	const scale = 1 / Math.sqrt(dK);
	const n = Q.length;
	const m = K.length;
	// Compute attention weights
	const scores: Float64Array[] = Q.map((q) => {
		const row = new Float64Array(m);
		for (let j = 0; j < m; j++) {
			let dot = 0;
			for (let d = 0; d < dK; d++) dot += (q[d] ?? 0) * (K[j]?.[d] ?? 0);
			row[j] = dot * scale;
		}
		return row;
	});
	// Softmax
	const attnWeights = scores.map((row) => {
		const maxVal = row.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY);
		const exp = new Float64Array(m).map((_, j) => Math.exp((row[j] ?? 0) - maxVal));
		const sumExp = exp.reduce((s, v) => s + v, 0);
		return new Float64Array(m).map((_, j) => (exp[j] ?? 0) / (sumExp || 1));
	});
	// Apply attention to V
	const dV = V[0]?.length ?? 1;
	return attnWeights.map((weights) => {
		const out = new Float64Array(dV);
		for (let j = 0; j < m; j++) {
			for (let d = 0; d < dV; d++) {
				out[d]! += (weights[j] ?? 0) * (V[j]?.[d] ?? 0);
			}
		}
		return out;
	});
}

/** Multi-head attention layer. */
export class MultiHeadAttention {
	private Wq_: Float64Array[][] | null = null;
	private Wk_: Float64Array[][] | null = null;
	private Wv_: Float64Array[][] | null = null;
	private Wo_: Float64Array[] | null = null;
	readonly nHeads: number;
	readonly dModel: number;
	readonly dKey: number;

	constructor(options: { nHeads?: number; dModel?: number } = {}) {
		this.nHeads = options.nHeads ?? 4;
		this.dModel = options.dModel ?? 64;
		this.dKey = Math.floor(this.dModel / this.nHeads);
	}

	initialize(randomState = 0): this {
		let rng = randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		const scale = Math.sqrt(2 / this.dModel);
		const initMatrix = (rows: number, cols: number): Float64Array[] =>
			Array.from({ length: rows }, () =>
				new Float64Array(cols).map(() => (rand() * 2 - 1) * scale),
			);
		this.Wq_ = Array.from({ length: this.nHeads }, () => initMatrix(this.dModel, this.dKey));
		this.Wk_ = Array.from({ length: this.nHeads }, () => initMatrix(this.dModel, this.dKey));
		this.Wv_ = Array.from({ length: this.nHeads }, () => initMatrix(this.dModel, this.dKey));
		this.Wo_ = initMatrix(this.nHeads * this.dKey, this.dModel);
		return this;
	}

	forward(X: Float64Array[]): Float64Array[] {
		if (this.Wq_ === null) throw new NotFittedError("MultiHeadAttention not initialized.");
		const n = X.length;
		const headOutputs: Float64Array[][] = [];
		for (let h = 0; h < this.nHeads; h++) {
			const Wq = this.Wq_[h]!;
			const Wk = this.Wk_![h]!;
			const Wv = this.Wv_![h]!;
			const Q = X.map((x) => {
				const q = new Float64Array(this.dKey);
				for (let k = 0; k < this.dKey; k++) {
					for (let d = 0; d < x.length; d++) q[k]! += (x[d] ?? 0) * (Wq[d]?.[k] ?? 0);
				}
				return q;
			});
			const K = X.map((x) => {
				const k = new Float64Array(this.dKey);
				for (let ki = 0; ki < this.dKey; ki++) {
					for (let d = 0; d < x.length; d++) k[ki]! += (x[d] ?? 0) * (Wk[d]?.[ki] ?? 0);
				}
				return k;
			});
			const V = X.map((x) => {
				const v = new Float64Array(this.dKey);
				for (let k = 0; k < this.dKey; k++) {
					for (let d = 0; d < x.length; d++) v[k]! += (x[d] ?? 0) * (Wv[d]?.[k] ?? 0);
				}
				return v;
			});
			headOutputs.push(scaledDotProductAttention(Q, K, V));
		}
		// Concatenate heads and project
		return X.map((_, i) => {
			const concat = new Float64Array(this.nHeads * this.dKey);
			for (let h = 0; h < this.nHeads; h++) {
				for (let k = 0; k < this.dKey; k++) {
					concat[h * this.dKey + k] = headOutputs[h]?.[i]?.[k] ?? 0;
				}
			}
			const out = new Float64Array(this.dModel);
			for (let d = 0; d < this.dModel; d++) {
				for (let c = 0; c < concat.length; c++) {
					out[d]! += (concat[c] ?? 0) * (this.Wo_?.[c]?.[d] ?? 0);
				}
			}
			return out;
		});
	}
}

/** Layer normalization. */
export function layerNorm(
	X: Float64Array[],
	gamma?: Float64Array,
	beta?: Float64Array,
	eps = 1e-5,
): Float64Array[] {
	return X.map((row) => {
		const n = row.length;
		let mean = 0;
		for (let j = 0; j < n; j++) mean += row[j] ?? 0;
		mean /= n;
		let variance = 0;
		for (let j = 0; j < n; j++) {
			const d = (row[j] ?? 0) - mean;
			variance += d * d;
		}
		variance /= n;
		const std = Math.sqrt(variance + eps);
		const out = new Float64Array(n);
		for (let j = 0; j < n; j++) {
			const normalized = ((row[j] ?? 0) - mean) / std;
			out[j] = normalized * (gamma?.[j] ?? 1) + (beta?.[j] ?? 0);
		}
		return out;
	});
}
