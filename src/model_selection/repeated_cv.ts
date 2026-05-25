/**
 * Repeated cross-validation iterators.
 * Port of sklearn.model_selection._repeated
 */

import type { GroupSplitResult } from "./group_cv.js";

export interface SplitResult {
	train: Int32Array;
	test: Int32Array;
}

function kfoldSplit(n: number, nSplits: number, seed: number): SplitResult[] {
	// Simple reproducible shuffle
	const indices = Array.from({ length: n }, (_, i) => i);
	let rng = seed;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0x100000000;
	};
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		const tmp = indices[i]!;
		indices[i] = indices[j]!;
		indices[j] = tmp;
	}

	const foldSize = Math.floor(n / nSplits);
	const results: SplitResult[] = [];
	for (let f = 0; f < nSplits; f++) {
		const start = f * foldSize;
		const end = f === nSplits - 1 ? n : start + foldSize;
		const test = new Int32Array(indices.slice(start, end));
		const train = new Int32Array([...indices.slice(0, start), ...indices.slice(end)]);
		results.push({ train, test });
	}
	return results;
}

/**
 * Repeated K-fold cross validation.
 * Port of sklearn.model_selection.RepeatedKFold
 */
export class RepeatedKFold {
	nSplits: number;
	nRepeats: number;
	randomState: number | null;

	constructor(nSplits = 5, nRepeats = 10, randomState: number | null = null) {
		this.nSplits = nSplits;
		this.nRepeats = nRepeats;
		this.randomState = randomState;
	}

	split(X: Float64Array[], _y?: Int32Array, _groups?: Int32Array): SplitResult[] {
		const n = X.length;
		const results: SplitResult[] = [];
		let seed = this.randomState ?? 42;
		for (let r = 0; r < this.nRepeats; r++) {
			const splits = kfoldSplit(n, this.nSplits, seed);
			results.push(...splits);
			seed = (seed * 22695477 + 1) & 0x7fffffff;
		}
		return results;
	}

	getNumSplits(): number {
		return this.nSplits * this.nRepeats;
	}
}

/**
 * Repeated Stratified K-Fold cross validator.
 * Port of sklearn.model_selection.RepeatedStratifiedKFold
 */
export class RepeatedStratifiedKFold {
	nSplits: number;
	nRepeats: number;
	randomState: number | null;

	constructor(nSplits = 5, nRepeats = 10, randomState: number | null = null) {
		this.nSplits = nSplits;
		this.nRepeats = nRepeats;
		this.randomState = randomState;
	}

	split(X: Float64Array[], y: Int32Array, _groups?: Int32Array): SplitResult[] {
		const n = X.length;
		const results: SplitResult[] = [];
		let seed = this.randomState ?? 42;

		// Get unique classes
		const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);

		for (let r = 0; r < this.nRepeats; r++) {
			// Stratified shuffle
			let rng = seed;
			const nextRng = (): number => {
				rng = (rng * 1664525 + 1013904223) & 0xffffffff;
				return (rng >>> 0) / 0x100000000;
			};

			// Build per-class index lists
			const classIndices = new Map<number, number[]>();
			for (const c of classes) classIndices.set(c, []);
			for (let i = 0; i < n; i++) classIndices.get(y[i]!)!.push(i);

			// Shuffle each class
			for (const indices of classIndices.values()) {
				for (let i = indices.length - 1; i > 0; i--) {
					const j = Math.floor(nextRng() * (i + 1));
					const tmp = indices[i]!;
					indices[i] = indices[j]!;
					indices[j] = tmp;
				}
			}

			// Assign to folds interleaved
			const folds: number[][] = Array.from({ length: this.nSplits }, () => []);
			for (const indices of classIndices.values()) {
				for (let i = 0; i < indices.length; i++) {
					folds[i % this.nSplits]!.push(indices[i]!);
				}
			}

			for (const testFold of folds) {
				const testSet = new Set(testFold);
				const train: number[] = [];
				const test: number[] = [];
				for (let i = 0; i < n; i++) {
					if (testSet.has(i)) test.push(i);
					else train.push(i);
				}
				results.push({ train: new Int32Array(train), test: new Int32Array(test) });
			}

			seed = (seed * 22695477 + 1) & 0x7fffffff;
		}
		return results;
	}

	getNumSplits(): number {
		return this.nSplits * this.nRepeats;
	}
}
