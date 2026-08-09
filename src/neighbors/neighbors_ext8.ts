/**
 * Neighbors extensions: HNSW approximate nearest neighbors, kd-tree extensions.
 * Port of sklearn.neighbors extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Approximate nearest neighbor search using random projection LSH. */
export class ApproximateNearestNeighbors {
	private data_: Float64Array[] | null = null;
	private projections_: Float64Array[] | null = null;
	private projectedData_: Float64Array | null = null;
	readonly nNeighbors: number;
	readonly nHashBits: number;
	readonly randomState: number;

	constructor(
		options: {
			nNeighbors?: number;
			nHashBits?: number;
			randomState?: number;
		} = {},
	) {
		this.nNeighbors = options.nNeighbors ?? 5;
		this.nHashBits = options.nHashBits ?? 8;
		this.randomState = options.randomState ?? 0;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		// Random projection vectors
		this.projections_ = Array.from({ length: this.nHashBits }, () => {
			const v = new Float64Array(nFeatures);
			for (let j = 0; j < nFeatures; j++) v[j] = rand() * 2 - 1;
			return v;
		});
		this.data_ = X;
		// Project all data points
		this.projectedData_ = new Float64Array(X.length * this.nHashBits);
		for (let i = 0; i < X.length; i++) {
			for (let b = 0; b < this.nHashBits; b++) {
				let dot = 0;
				for (let j = 0; j < nFeatures; j++) {
					dot += (X[i]?.[j] ?? 0) * (this.projections_![b]?.[j] ?? 0);
				}
				this.projectedData_[i * this.nHashBits + b] = dot;
			}
		}
		return this;
	}

	kneighbors(X: Float64Array[]): { distances: Float64Array[]; indices: Int32Array[] } {
		if (this.data_ === null || this.projections_ === null || this.projectedData_ === null) {
			throw new NotFittedError("ApproximateNearestNeighbors is not fitted.");
		}
		const n = this.data_.length;
		const k = Math.min(this.nNeighbors, n);
		const distances: Float64Array[] = [];
		const indices: Int32Array[] = [];
		for (const query of X) {
			// Project query
			const qProj = new Float64Array(this.nHashBits);
			for (let b = 0; b < this.nHashBits; b++) {
				let dot = 0;
				for (let j = 0; j < query.length; j++) {
					dot += (query[j] ?? 0) * (this.projections_![b]?.[j] ?? 0);
				}
				qProj[b] = dot;
			}
			// Score by projection similarity, then compute exact distances for top candidates
			const scores = new Float64Array(n);
			for (let i = 0; i < n; i++) {
				let sim = 0;
				for (let b = 0; b < this.nHashBits; b++) {
					const di = this.projectedData_[i * this.nHashBits + b] ?? 0;
					sim += (qProj[b] ?? 0) * di;
				}
				scores[i] = sim;
			}
			// Take top-3k candidates by score
			const candidateK = Math.min(n, 3 * k);
			const candidateIdx = Array.from({ length: n }, (_, i) => i)
				.sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0))
				.slice(0, candidateK);
			// Exact distance on candidates
			const exact = candidateIdx.map((ci) => {
				let d = 0;
				for (let j = 0; j < query.length; j++) {
					const diff = (query[j] ?? 0) - (this.data_![ci]?.[j] ?? 0);
					d += diff * diff;
				}
				return { d: Math.sqrt(d), ci };
			});
			exact.sort((a, b) => a.d - b.d);
			const kNN = exact.slice(0, k);
			distances.push(new Float64Array(kNN.map((e) => e.d)));
			indices.push(new Int32Array(kNN.map((e) => e.ci)));
		}
		return { distances, indices };
	}
}

/** Compute local outlier factor scores. */
export function computeLOFScores(
	X: Float64Array[],
	kNeighbors = 5,
): Float64Array {
	const n = X.length;
	const k = Math.min(kNeighbors, n - 1);

	const dist = (a: Float64Array, b: Float64Array): number => {
		let d = 0;
		for (let j = 0; j < a.length; j++) {
			const diff = (a[j] ?? 0) - (b[j] ?? 0);
			d += diff * diff;
		}
		return Math.sqrt(d);
	};

	// k-distance for each point
	const kDistances = new Float64Array(n);
	const neighborhoods: Int32Array[] = [];
	for (let i = 0; i < n; i++) {
		const dists = Array.from({ length: n }, (_, j) => ({
			d: j === i ? Number.POSITIVE_INFINITY : dist(X[i]!, X[j]!),
			j,
		})).sort((a, b) => a.d - b.d);
		kDistances[i] = dists[k - 1]?.d ?? 0;
		neighborhoods.push(new Int32Array(dists.slice(0, k).map((e) => e.j)));
	}

	// Reachability distance
	const reachDist = (i: number, j: number): number =>
		Math.max(kDistances[j] ?? 0, dist(X[i]!, X[j]!));

	// Local reachability density
	const lrd = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const nb = neighborhoods[i]!;
		let sumRD = 0;
		for (let ki = 0; ki < nb.length; ki++) sumRD += reachDist(i, nb[ki] ?? 0);
		lrd[i] = nb.length === 0 || sumRD === 0 ? 0 : nb.length / sumRD;
	}

	// LOF
	const lof = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const nb = neighborhoods[i]!;
		let sumLRD = 0;
		for (let ki = 0; ki < nb.length; ki++) sumLRD += lrd[nb[ki]!] ?? 0;
		lof[i] = nb.length === 0 || (lrd[i] ?? 0) === 0 ? 1 : sumLRD / ((lrd[i] ?? 0) * nb.length);
	}
	return lof;
}
