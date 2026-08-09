/**
 * Cluster extensions: BIRCH algorithm utilities.
 * Port of sklearn.cluster.birch extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Clustering Feature (CF) node for BIRCH. */
interface CFEntry {
	n: number;
	ls: Float64Array; // linear sum
	ss: number; // squared sum
}

function newCFEntry(dim: number): CFEntry {
	return { n: 0, ls: new Float64Array(dim), ss: 0 };
}

function addToCF(cf: CFEntry, x: Float64Array): void {
	cf.n++;
	for (let j = 0; j < cf.ls.length; j++) cf.ls[j]! += x[j] ?? 0;
	for (let j = 0; j < x.length; j++) cf.ss += (x[j] ?? 0) * (x[j] ?? 0);
}

function cfCentroid(cf: CFEntry): Float64Array {
	const c = new Float64Array(cf.ls.length);
	for (let j = 0; j < cf.ls.length; j++) c[j] = cf.n === 0 ? 0 : (cf.ls[j] ?? 0) / cf.n;
	return c;
}

function cfRadius(cf: CFEntry): number {
	if (cf.n === 0) return 0;
	const centroid = cfCentroid(cf);
	let r = 0;
	const avgSS = cf.ss / cf.n;
	for (let j = 0; j < centroid.length; j++) r += (centroid[j] ?? 0) * (centroid[j] ?? 0);
	return Math.sqrt(Math.max(0, avgSS - r));
}

function euclidean(a: Float64Array, b: Float64Array): number {
	let d = 0;
	for (let j = 0; j < a.length; j++) {
		const diff = (a[j] ?? 0) - (b[j] ?? 0);
		d += diff * diff;
	}
	return Math.sqrt(d);
}

/** Simplified BIRCH clustering implementation. */
export class BirchSimple {
	private subclusterCentroids_: Float64Array[] | null = null;
	private labels_: Int32Array | null = null;
	readonly threshold: number;
	readonly branchingFactor: number;
	readonly nClusters: number | null;

	constructor(
		options: {
			threshold?: number;
			branchingFactor?: number;
			nClusters?: number | null;
		} = {},
	) {
		this.threshold = options.threshold ?? 0.5;
		this.branchingFactor = options.branchingFactor ?? 50;
		this.nClusters = options.nClusters ?? 3;
	}

	fit(X: Float64Array[]): this {
		const nFeatures = X[0]?.length ?? 0;
		const subclusters: CFEntry[] = [];

		for (const x of X) {
			if (subclusters.length === 0) {
				const cf = newCFEntry(nFeatures);
				addToCF(cf, x);
				subclusters.push(cf);
				continue;
			}
			// Find closest subcluster
			let bestIdx = 0;
			let bestDist = Number.POSITIVE_INFINITY;
			for (let k = 0; k < subclusters.length; k++) {
				const d = euclidean(cfCentroid(subclusters[k]!), x);
				if (d < bestDist) {
					bestDist = d;
					bestIdx = k;
				}
			}
			// Check if we can add to this subcluster
			const cf = subclusters[bestIdx]!;
			const testCF = newCFEntry(nFeatures);
			Object.assign(testCF, { n: cf.n, ls: new Float64Array(cf.ls), ss: cf.ss });
			addToCF(testCF, x);
			if (cfRadius(testCF) <= this.threshold) {
				addToCF(cf, x);
			} else {
				const newCF = newCFEntry(nFeatures);
				addToCF(newCF, x);
				subclusters.push(newCF);
			}
		}

		this.subclusterCentroids_ = subclusters.map((cf) => cfCentroid(cf));

		// Assign labels via final clustering of subclusters
		const nTarget = Math.min(this.nClusters ?? subclusters.length, subclusters.length);
		const clusterLabels = kMeansLabels(this.subclusterCentroids_, nTarget);

		this.labels_ = new Int32Array(X.length);
		for (let i = 0; i < X.length; i++) {
			let bestK = 0;
			let bestD = Number.POSITIVE_INFINITY;
			for (let k = 0; k < (this.subclusterCentroids_?.length ?? 0); k++) {
				const d = euclidean(X[i]!, this.subclusterCentroids_![k]!);
				if (d < bestD) {
					bestD = d;
					bestK = k;
				}
			}
			this.labels_[i] = clusterLabels[bestK] ?? 0;
		}
		return this;
	}

	predict(X: Float64Array[]): Int32Array {
		if (this.subclusterCentroids_ === null) throw new NotFittedError("BirchSimple is not fitted.");
		const nTarget = Math.min(this.nClusters ?? this.subclusterCentroids_.length, this.subclusterCentroids_.length);
		const clusterLabels = kMeansLabels(this.subclusterCentroids_, nTarget);
		return new Int32Array(
			X.map((x) => {
				let bestK = 0;
				let bestD = Number.POSITIVE_INFINITY;
				for (let k = 0; k < (this.subclusterCentroids_?.length ?? 0); k++) {
					const d = euclidean(x, this.subclusterCentroids_![k]!);
					if (d < bestD) {
						bestD = d;
						bestK = k;
					}
				}
				return clusterLabels[bestK] ?? 0;
			}),
		);
	}

	get labels(): Int32Array {
		if (this.labels_ === null) throw new NotFittedError("BirchSimple is not fitted.");
		return this.labels_;
	}
}

function kMeansLabels(X: Float64Array[], k: number): Int32Array {
	if (k >= X.length) return new Int32Array(X.length).map((_, i) => i);
	const centroids = X.slice(0, k).map((x) => new Float64Array(x));
	const labels = new Int32Array(X.length);
	for (let iter = 0; iter < 10; iter++) {
		for (let i = 0; i < X.length; i++) {
			let best = 0;
			let bestD = Number.POSITIVE_INFINITY;
			for (let c = 0; c < k; c++) {
				const d = euclidean(X[i]!, centroids[c]!);
				if (d < bestD) {
					bestD = d;
					best = c;
				}
			}
			labels[i] = best;
		}
		const dim = X[0]?.length ?? 0;
		const newCentroids = Array.from({ length: k }, () => new Float64Array(dim));
		const counts = new Int32Array(k);
		for (let i = 0; i < X.length; i++) {
			const c = labels[i] ?? 0;
			counts[c]!++;
			for (let j = 0; j < dim; j++) newCentroids[c]![j]! += X[i]?.[j] ?? 0;
		}
		for (let c = 0; c < k; c++) {
			if ((counts[c] ?? 0) > 0) {
				for (let j = 0; j < dim; j++) newCentroids[c]![j]! /= counts[c]!;
				centroids[c] = newCentroids[c]!;
			}
		}
	}
	return labels;
}
