/**
 * Manifold extensions: UMAP utilities, parametric t-SNE.
 * Port of sklearn.manifold extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Trustworthiness score for manifold embedding quality. */
export function trustworthiness(
	X: Float64Array[],
	XEmbedded: Float64Array[],
	nNeighbors = 5,
): number {
	const n = X.length;
	const k = Math.min(nNeighbors, n - 1);

	const dist = (a: Float64Array, b: Float64Array): number => {
		let d = 0;
		for (let j = 0; j < a.length; j++) {
			const diff = (a[j] ?? 0) - (b[j] ?? 0);
			d += diff * diff;
		}
		return d;
	};

	// Rank each point's neighbors in the original space
	const origRanks: Int32Array[] = X.map((xi, i) => {
		const dists = Array.from({ length: n }, (_, j) => ({
			d: j === i ? Number.POSITIVE_INFINITY : dist(xi, X[j]!),
			j,
		})).sort((a, b) => a.d - b.d);
		const ranks = new Int32Array(n);
		for (let r = 0; r < n; r++) ranks[dists[r]!.j] = r + 1;
		return ranks;
	});

	// k-NN in embedded space
	const embNN: Int32Array[] = XEmbedded.map((xi, i) => {
		const dists = Array.from({ length: n }, (_, j) => ({
			d: j === i ? Number.POSITIVE_INFINITY : dist(xi, XEmbedded[j]!),
			j,
		})).sort((a, b) => a.d - b.d);
		return new Int32Array(dists.slice(0, k).map((e) => e.j));
	});

	let sum = 0;
	for (let i = 0; i < n; i++) {
		for (let ki = 0; ki < k; ki++) {
			const j = embNN[i]![ki]!;
			const r = origRanks[i]![j] ?? 0;
			if (r > k) sum += r - k;
		}
	}

	return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * sum;
}

/** Sammon mapping for dimensionality reduction (non-linear). */
export class SammonMapping {
	private embedding_: Float64Array[] | null = null;
	readonly nComponents: number;
	readonly maxIter: number;
	readonly learningRate: number;
	readonly randomState: number;

	constructor(
		options: {
			nComponents?: number;
			maxIter?: number;
			learningRate?: number;
			randomState?: number;
		} = {},
	) {
		this.nComponents = options.nComponents ?? 2;
		this.maxIter = options.maxIter ?? 200;
		this.learningRate = options.learningRate ?? 0.3;
		this.randomState = options.randomState ?? 0;
	}

	fit(X: Float64Array[]): this {
		const n = X.length;
		const k = this.nComponents;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};

		// Init embedding randomly
		const Y: Float64Array[] = Array.from({ length: n }, () => {
			const row = new Float64Array(k);
			for (let j = 0; j < k; j++) row[j] = rand() * 2 - 1;
			return row;
		});

		// Compute input pairwise distances
		const D = Array.from({ length: n }, (_, i) =>
			new Float64Array(n).map((_, j) => {
				let d = 0;
				for (let dim = 0; dim < X[0]!.length; dim++) {
					const diff = (X[i]?.[dim] ?? 0) - (X[j]?.[dim] ?? 0);
					d += diff * diff;
				}
				return Math.sqrt(d);
			}),
		);
		const dSum = D.reduce((s, row) => s + row.reduce((rs, v) => rs + v, 0), 0);

		for (let iter = 0; iter < this.maxIter; iter++) {
			for (let i = 0; i < n; i++) {
				const grad = new Float64Array(k);
				for (let j = 0; j < n; j++) {
					if (i === j) continue;
					const dij = D[i]![j] ?? 0;
					if (dij === 0) continue;
					let dij_y = 0;
					for (let dim = 0; dim < k; dim++) {
						const diff = (Y[i]?.[dim] ?? 0) - (Y[j]?.[dim] ?? 0);
						dij_y += diff * diff;
					}
					dij_y = Math.sqrt(dij_y) || 1e-10;
					const factor = (dij_y - dij) / (dij * dij_y);
					for (let dim = 0; dim < k; dim++) {
					grad[dim]! += factor * ((Y[i]?.[dim] ?? 0) - (Y[j]?.[dim] ?? 0));
					}
				}
				for (let dim = 0; dim < k; dim++) {
				Y[i]![dim]! -= (this.learningRate / dSum) * (grad[dim] ?? 0);
				}
			}
		}
		this.embedding_ = Y;
		return this;
	}

	get embedding(): Float64Array[] {
		if (this.embedding_ === null) throw new NotFittedError("SammonMapping is not fitted.");
		return this.embedding_;
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).embedding_ ?? [];
	}
}

/** Compute neighborhood hit metric for embedding quality. */
export function neighborhoodHit(
	X: Float64Array[],
	XEmbedded: Float64Array[],
	labels: Int32Array,
	kNeighbors = 5,
): number {
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

	let hits = 0;
	for (let i = 0; i < n; i++) {
		const dists = Array.from({ length: n }, (_, j) => ({
			d: j === i ? Number.POSITIVE_INFINITY : dist(XEmbedded[i]!, XEmbedded[j]!),
			j,
		})).sort((a, b) => a.d - b.d);
		const nn = dists.slice(0, k);
		let same = 0;
		for (const { j } of nn) {
			if ((labels[j] ?? 0) === (labels[i] ?? 0)) same++;
		}
		hits += same / k;
	}
	return hits / n;
}
