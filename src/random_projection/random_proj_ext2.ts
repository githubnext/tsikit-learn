/**
 * Random projection extensions: Johnson-Lindenstrauss lemma utilities.
 * Port of sklearn.random_projection extensions.
 */

/** Compute Johnson-Lindenstrauss bound: min dimensions for eps-JL embedding. */
export function johnsonLindenstraussBound(
	nSamples: number,
	eps: number,
): number {
	if (eps <= 0 || eps >= 1) throw new Error("eps must be in (0, 1)");
	const denominator =
		eps * eps / 2 - eps * eps * eps / 3;
	return Math.ceil(4 * Math.log(nSamples) / denominator);
}

/** Gaussian random projection matrix (dense). */
export class GaussianRandomProjectionMatrix {
	readonly nComponents: number;
	readonly nFeatures: number;
	readonly randomState: number;
	private matrix_: Float64Array[] | null = null;

	constructor(options: {
		nComponents: number;
		nFeatures: number;
		randomState?: number;
	}) {
		this.nComponents = options.nComponents;
		this.nFeatures = options.nFeatures;
		this.randomState = options.randomState ?? 0;
	}

	generate(): Float64Array[] {
		if (this.matrix_ !== null) return this.matrix_;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		const scale = Math.sqrt(this.nFeatures);
		// Box-Muller for Gaussian
		const gauss = (): number => {
			const u1 = Math.max(rand(), 1e-10);
			const u2 = rand();
			return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) / scale;
		};
		this.matrix_ = Array.from({ length: this.nFeatures }, () =>
			new Float64Array(this.nComponents).map(() => gauss()),
		);
		return this.matrix_;
	}

	project(X: Float64Array[]): Float64Array[] {
		const mat = this.generate();
		return X.map((row) => {
			const out = new Float64Array(this.nComponents);
			for (let j = 0; j < row.length; j++) {
				for (let c = 0; c < this.nComponents; c++) {
					out[c]! += (row[j] ?? 0) * (mat[j]?.[c] ?? 0);
				}
			}
			return out;
		});
	}
}

/** Sparse random projection with density control. */
export class SparseRandomProjectionMatrix {
	readonly nComponents: number;
	readonly nFeatures: number;
	readonly density: number;
	readonly randomState: number;
	private matrix_: Float64Array[] | null = null;

	constructor(options: {
		nComponents: number;
		nFeatures: number;
		density?: number;
		randomState?: number;
	}) {
		this.nComponents = options.nComponents;
		this.nFeatures = options.nFeatures;
		this.density = options.density ?? 1 / Math.sqrt(options.nFeatures);
		this.randomState = options.randomState ?? 0;
	}

	generate(): Float64Array[] {
		if (this.matrix_ !== null) return this.matrix_;
		let rng = this.randomState;
		const rand = (): number => {
			rng = (rng * 1664525 + 1013904223) & 0xffffffff;
			return (rng >>> 0) / 0xffffffff;
		};
		const scale = Math.sqrt(1 / (this.density * this.nComponents));
		this.matrix_ = Array.from({ length: this.nFeatures }, () => {
			const row = new Float64Array(this.nComponents);
			for (let c = 0; c < this.nComponents; c++) {
				const u = rand();
				if (u < this.density / 2) row[c] = -scale;
				else if (u < this.density) row[c] = scale;
			}
			return row;
		});
		return this.matrix_;
	}

	project(X: Float64Array[]): Float64Array[] {
		const mat = this.generate();
		return X.map((row) => {
			const out = new Float64Array(this.nComponents);
			for (let j = 0; j < row.length; j++) {
				const mj = mat[j];
				if (mj === undefined) continue;
				for (let c = 0; c < this.nComponents; c++) {
					const mc = mj[c];
					if (mc !== 0) out[c]! += (row[j] ?? 0) * (mc ?? 0);
				}
			}
			return out;
		});
	}
}

/** Compute embedding distortion: max ratio of pairwise distances. */
export function embeddingDistortion(
	X: Float64Array[],
	XProj: Float64Array[],
	nPairs = 100,
	randomState = 0,
): number {
	const n = X.length;
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const pairs: Array<[number, number]> = Array.from({ length: nPairs }, () => {
		const i = Math.floor(rand() * n);
		let j = Math.floor(rand() * n);
		if (j === i) j = (j + 1) % n;
		return [i, j];
	});

	const dist = (a: Float64Array, b: Float64Array): number => {
		let d = 0;
		for (let k = 0; k < a.length; k++) {
			const diff = (a[k] ?? 0) - (b[k] ?? 0);
			d += diff * diff;
		}
		return Math.sqrt(d);
	};

	let maxDistortion = 0;
	for (const [i, j] of pairs) {
		const dOrig = dist(X[i]!, X[j]!);
		const dProj = dist(XProj[i]!, XProj[j]!);
		if (dOrig > 0) {
			const ratio = dProj / dOrig;
			maxDistortion = Math.max(maxDistortion, Math.abs(ratio - 1));
		}
	}
	return maxDistortion;
}
