/**
 * Discriminant analysis extensions: regularized LDA, QDA extensions.
 * Port of sklearn.discriminant_analysis extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Regularized Linear Discriminant Analysis (RLDA). */
export class RegularizedLDA {
	private means_: Map<number, Float64Array> | null = null;
	private globalMean_: Float64Array | null = null;
	private projMatrix_: Float64Array[] | null = null;
	private classes_: Int32Array | null = null;
	readonly nComponents: number;
	readonly regParam: number;

	constructor(options: { nComponents?: number; regParam?: number } = {}) {
		this.nComponents = options.nComponents ?? 2;
		this.regParam = options.regParam ?? 0.0;
	}

	fit(X: Float64Array[], y: Int32Array): this {
		const n = X.length;
		const p = X[0]?.length ?? 0;
		const classes = [...new Set([...y])].sort((a, b) => a - b);
		this.classes_ = new Int32Array(classes);
		const nClasses = classes.length;
		const k = Math.min(this.nComponents, nClasses - 1, p);

		// Compute class means
		const means = new Map<number, Float64Array>();
		const classCounts = new Map<number, number>();
		for (const cls of classes) {
			means.set(cls, new Float64Array(p));
			classCounts.set(cls, 0);
		}
		for (let i = 0; i < n; i++) {
			const cls = y[i] ?? 0;
			const mean = means.get(cls)!;
			const count = classCounts.get(cls) ?? 0;
			for (let j = 0; j < p; j++) mean[j]! += X[i]?.[j] ?? 0;
			classCounts.set(cls, count + 1);
		}
		for (const cls of classes) {
			const cnt = classCounts.get(cls) ?? 1;
			const m = means.get(cls)!;
			for (let j = 0; j < p; j++) m[j]! /= cnt;
		}
		this.means_ = means;

		// Global mean
		const globalMean = new Float64Array(p);
		for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) globalMean[j]! += X[i]?.[j] ?? 0;
		for (let j = 0; j < p; j++) globalMean[j]! /= n;
		this.globalMean_ = globalMean;

		// Between-class scatter Sb
		const Sb = Array.from({ length: p }, () => new Float64Array(p));
		for (const cls of classes) {
			const m = means.get(cls)!;
			const cnt = classCounts.get(cls) ?? 0;
			const d = new Float64Array(p).map((_, j) => (m[j] ?? 0) - (globalMean[j] ?? 0));
			for (let a = 0; a < p; a++) {
				for (let b = 0; b < p; b++) Sb[a]![b]! += cnt * (d[a] ?? 0) * (d[b] ?? 0);
			}
		}

		// Within-class scatter Sw (with regularization)
		const Sw = Array.from({ length: p }, () => new Float64Array(p));
		for (let i = 0; i < n; i++) {
			const cls = y[i] ?? 0;
			const m = means.get(cls)!;
			const d = new Float64Array(p).map((_, j) => (X[i]?.[j] ?? 0) - (m[j] ?? 0));
			for (let a = 0; a < p; a++) {
				for (let b = 0; b < p; b++) Sw[a]![b]! += (d[a] ?? 0) * (d[b] ?? 0);
			}
		}
		// Add regularization
		for (let j = 0; j < p; j++) Sw[j]![j]! += this.regParam * n;

		// Simplified: compute first k directions using power iteration on Sw^{-1} * Sb
		const SwInv = invertSmall(Sw);
		const M = Array.from({ length: p }, (_, a) =>
			new Float64Array(p).map((_, b) => {
				let s = 0;
				for (let c = 0; c < p; c++) s += (SwInv[a]?.[c] ?? 0) * (Sb[c]?.[b] ?? 0);
				return s;
			}),
		);
		// Power iteration for top-k eigenvectors
		const dirs: Float64Array[] = [];
		for (let comp = 0; comp < k; comp++) {
			let v = new Float64Array(p);
			v[comp % p] = 1;
			for (let iter = 0; iter < 50; iter++) {
				const newV = new Float64Array(p);
				for (let a = 0; a < p; a++) {
					for (let b = 0; b < p; b++) newV[a]! += (M[a]?.[b] ?? 0) * (v[b] ?? 0);
				}
				// Deflate previous dirs
				for (const d of dirs) {
					let dot = 0;
					for (let j = 0; j < p; j++) dot += (newV[j] ?? 0) * (d[j] ?? 0);
					for (let j = 0; j < p; j++) newV[j]! -= dot * (d[j] ?? 0);
				}
				let norm = 0;
				for (let j = 0; j < p; j++) norm += (newV[j] ?? 0) ** 2;
				norm = Math.sqrt(norm) || 1;
				for (let j = 0; j < p; j++) newV[j]! /= norm;
				v = newV;
			}
			dirs.push(v);
		}
		this.projMatrix_ = dirs;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (this.projMatrix_ === null || this.globalMean_ === null) throw new NotFittedError("RegularizedLDA is not fitted.");
		return X.map((row) => {
			const out = new Float64Array(this.projMatrix_!.length);
			for (let c = 0; c < this.projMatrix_!.length; c++) {
				for (let j = 0; j < row.length; j++) {
					out[c]! += ((row[j] ?? 0) - (this.globalMean_![j] ?? 0)) * (this.projMatrix_![c]?.[j] ?? 0);
				}
			}
			return out;
		});
	}

	fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}

	predict(X: Float64Array[]): Int32Array {
		if (this.means_ === null || this.classes_ === null) throw new NotFittedError("RegularizedLDA is not fitted.");
		const Xproj = this.transform(X);
		const meanProj = new Map<number, Float64Array>();
		for (const cls of this.classes_) {
			const m = this.means_.get(cls)!;
			const row = new Float64Array(m.length);
			for (let j = 0; j < m.length; j++) row[j] = (m[j] ?? 0) - (this.globalMean_![j] ?? 0);
			// Project mean
			const pm = new Float64Array(this.projMatrix_!.length);
			for (let c = 0; c < this.projMatrix_!.length; c++) {
				for (let j = 0; j < row.length; j++) pm[c]! += (row[j] ?? 0) * (this.projMatrix_![c]?.[j] ?? 0);
			}
			meanProj.set(cls, pm);
		}
		return new Int32Array(
			Xproj.map((xi) => {
				let bestCls = this.classes_![0] ?? 0;
				let bestDist = Number.POSITIVE_INFINITY;
				for (const cls of this.classes_!) {
					const pm = meanProj.get(cls)!;
					let d = 0;
					for (let c = 0; c < xi.length; c++) {
						const diff = (xi[c] ?? 0) - (pm[c] ?? 0);
						d += diff * diff;
					}
					if (d < bestDist) { bestDist = d; bestCls = cls; }
				}
				return bestCls;
			}),
		);
	}
}

function invertSmall(A: Float64Array[]): Float64Array[] {
	const n = A.length;
	const aug = A.map((row, i) => {
		const r = new Float64Array(2 * n);
		for (let j = 0; j < n; j++) r[j] = row[j] ?? 0;
		r[n + i] = 1;
		return r;
	});
	for (let col = 0; col < n; col++) {
		let maxRow = col;
		for (let row = col + 1; row < n; row++) {
			if (Math.abs(aug[row]?.[col] ?? 0) > Math.abs(aug[maxRow]?.[col] ?? 0)) maxRow = row;
		}
		const tmp = aug[col]!; aug[col] = aug[maxRow]!; aug[maxRow] = tmp;
		const pivot = aug[col]?.[col] ?? 1;
		if (Math.abs(pivot) < 1e-12) continue;
		for (let j = 0; j < 2 * n; j++) aug[col]![j]! /= pivot;
		for (let row = 0; row < n; row++) {
			if (row === col) continue;
			const f = aug[row]?.[col] ?? 0;
			for (let j = 0; j < 2 * n; j++) aug[row]![j]! -= f * (aug[col]?.[j] ?? 0);
		}
	}
	return aug.map((row) => new Float64Array(row.slice(n)));
}
