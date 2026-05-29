/**
 * Extended biclustering utilities: consensus biclustering, evaluation metrics.
 * Port of sklearn.cluster.bicluster extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute the consensus score between two sets of biclusters. */
export function consensusScore(
	a: { rowLabels: Int32Array; colLabels: Int32Array },
	b: { rowLabels: Int32Array; colLabels: Int32Array },
): number {
	const nRows = a.rowLabels.length;
	const nCols = a.colLabels.length;
	const aRows = new Set<number>();
	const bRows = new Set<number>();
	for (let i = 0; i < nRows; i++) {
		if ((a.rowLabels[i] ?? 0) === 1) aRows.add(i);
		if ((b.rowLabels[i] ?? 0) === 1) bRows.add(i);
	}
	const aCols = new Set<number>();
	const bCols = new Set<number>();
	for (let j = 0; j < nCols; j++) {
		if ((a.colLabels[j] ?? 0) === 1) aCols.add(j);
		if ((b.colLabels[j] ?? 0) === 1) bCols.add(j);
	}
	const rowInter = [...aRows].filter((r) => bRows.has(r)).length;
	const colInter = [...aCols].filter((c) => bCols.has(c)).length;
	const aSize = aRows.size * aCols.size;
	const bSize = bRows.size * bCols.size;
	if (aSize === 0 || bSize === 0) return 0;
	return (rowInter * colInter) / Math.sqrt(aSize * bSize);
}

/** Check if a biclustering result is non-degenerate (has at least one row and column in each bicluster). */
export function checkBiclustersNonDegenerate(
	rowLabels: Int32Array,
	colLabels: Int32Array,
	nClusters: number,
): boolean {
	for (let k = 0; k < nClusters; k++) {
		let rowCount = 0;
		let colCount = 0;
		for (let i = 0; i < rowLabels.length; i++) {
			if ((rowLabels[i] ?? 0) === k) rowCount++;
		}
		for (let j = 0; j < colLabels.length; j++) {
			if ((colLabels[j] ?? 0) === k) colCount++;
		}
		if (rowCount === 0 || colCount === 0) return false;
	}
	return true;
}

/** Bicluster evaluator for measuring residue and volume. */
export class BiclusterEvaluator {
	private rowLabels_: Int32Array | null = null;
	private colLabels_: Int32Array | null = null;
	private data_: Float64Array[] | null = null;

	fit(
		data: Float64Array[],
		rowLabels: Int32Array,
		colLabels: Int32Array,
	): this {
		this.data_ = data;
		this.rowLabels_ = rowLabels;
		this.colLabels_ = colLabels;
		return this;
	}

	/** Compute the average residue of a bicluster (lower is better). */
	averageResidue(clusterId: number): number {
		if (this.data_ === null || this.rowLabels_ === null || this.colLabels_ === null) {
			throw new NotFittedError("BiclusterEvaluator is not fitted.");
		}
		const rows: number[] = [];
		const cols: number[] = [];
		for (let i = 0; i < this.rowLabels_.length; i++) {
			if ((this.rowLabels_[i] ?? 0) === clusterId) rows.push(i);
		}
		for (let j = 0; j < this.colLabels_.length; j++) {
			if ((this.colLabels_[j] ?? 0) === clusterId) cols.push(j);
		}
		if (rows.length === 0 || cols.length === 0) return 0;
		let grandMean = 0;
		for (const i of rows) {
			for (const j of cols) {
				grandMean += this.data_[i]?.[j] ?? 0;
			}
		}
		grandMean /= rows.length * cols.length;
		const rowMeans = rows.map((i) => {
			let s = 0;
			for (const j of cols) s += this.data_![i]?.[j] ?? 0;
			return s / cols.length;
		});
		const colMeans = cols.map((j) => {
			let s = 0;
			for (const i of rows) s += this.data_![i]?.[j] ?? 0;
			return s / rows.length;
		});
		let residue = 0;
		for (let ri = 0; ri < rows.length; ri++) {
			for (let ci = 0; ci < cols.length; ci++) {
				const val = this.data_[rows[ri]!]?.[cols[ci]!] ?? 0;
				const r =
					val -
					(rowMeans[ri] ?? 0) -
					(colMeans[ci] ?? 0) +
					grandMean;
				residue += r * r;
			}
		}
		return residue / (rows.length * cols.length);
	}
}

/** Generate a checkerboard matrix for testing biclustering algorithms. */
export function makeCheckerboard(
	shape: [number, number],
	nClusters: [number, number],
	noise = 0.0,
	seed = 0,
): { data: Float64Array[]; rowLabels: Int32Array; colLabels: Int32Array } {
	const [nRows, nCols] = shape;
	const [nRowClusters, nColClusters] = nClusters;
	const rowLabels = new Int32Array(nRows);
	const colLabels = new Int32Array(nCols);
	for (let i = 0; i < nRows; i++) {
		rowLabels[i] = i % nRowClusters;
	}
	for (let j = 0; j < nCols; j++) {
		colLabels[j] = j % nColClusters;
	}
	let rng = seed;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const data: Float64Array[] = Array.from({ length: nRows }, (_, i) => {
		const row = new Float64Array(nCols);
		for (let j = 0; j < nCols; j++) {
			const same = (rowLabels[i] ?? 0) === (colLabels[j] ?? 0) % nRowClusters ? 1 : 0;
			row[j] = same + noise * (rand() - 0.5);
		}
		return row;
	});
	return { data, rowLabels, colLabels };
}
