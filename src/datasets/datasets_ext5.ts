/**
 * Datasets extensions: synthetic datasets for benchmarking.
 * Port of sklearn.datasets extensions.
 */

/** Generate a dataset for benchmarking classifiers (Swiss roll with labels). */
export function makeSwissRoll(
	nSamples = 100,
	noise = 0.0,
	randomState = 0,
): { X: Float64Array[]; t: Float64Array } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const t = new Float64Array(nSamples).map(() => 1.5 * Math.PI * (1 + 2 * rand()));
	const X: Float64Array[] = Array.from({ length: nSamples }, (_, i) => {
		const ti = t[i] ?? 0;
		return new Float64Array([
			ti * Math.cos(ti) + noise * (rand() - 0.5),
			21 * rand() + noise * (rand() - 0.5),
			ti * Math.sin(ti) + noise * (rand() - 0.5),
		]);
	});
	return { X, t };
}

/** Generate a dataset of S-curve manifold. */
export function makeSCurve(
	nSamples = 100,
	noise = 0.0,
	randomState = 0,
): { X: Float64Array[]; t: Float64Array } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const t = new Float64Array(nSamples).map(() => 3 * Math.PI * (rand() - 0.5));
	const X: Float64Array[] = Array.from({ length: nSamples }, (_, i) => {
		const ti = t[i] ?? 0;
		return new Float64Array([
			Math.sin(ti) + noise * (rand() - 0.5),
			2 * rand() + noise * (rand() - 0.5),
			Math.sign(ti) * (Math.cos(ti) - 1) + noise * (rand() - 0.5),
		]);
	});
	return { X, t };
}

/** Generate a checkerboard dataset. */
export function makeCheckerboardData(
	nSamples = 200,
	nSquares = 4,
	randomState = 0,
): { X: Float64Array[]; y: Int32Array } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const X: Float64Array[] = Array.from({ length: nSamples }, () => new Float64Array([rand(), rand()]));
	const y = new Int32Array(nSamples).map((_, i) => {
		const x1 = X[i]?.[0] ?? 0;
		const x2 = X[i]?.[1] ?? 0;
		const sq1 = Math.floor(x1 * nSquares);
		const sq2 = Math.floor(x2 * nSquares);
		return (sq1 + sq2) % 2;
	});
	return { X, y };
}

/** Generate a dataset of XOR pattern. */
export function makeXOR(
	nSamples = 200,
	noise = 0.1,
	randomState = 0,
): { X: Float64Array[]; y: Int32Array } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const X: Float64Array[] = Array.from({ length: nSamples }, () =>
		new Float64Array([rand() * 2 - 1, rand() * 2 - 1]),
	);
	const y = new Int32Array(nSamples).map((_, i) => {
		const x1 = (X[i]?.[0] ?? 0) + noise * (rand() - 0.5);
		const x2 = (X[i]?.[1] ?? 0) + noise * (rand() - 0.5);
		return x1 * x2 > 0 ? 1 : 0;
	});
	return { X, y };
}

/** Generate low-rank data with noise. */
export function makeLowRankMatrix(
	nSamples = 100,
	nFeatures = 50,
	effectiveRank = 10,
	tailStrength = 0.5,
	randomState = 0,
): Float64Array[] {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const n = nSamples;
	const p = nFeatures;
	const k = Math.min(n, p, effectiveRank * 2);
	// Generate random orthogonal-ish basis
	const U: Float64Array[] = Array.from({ length: n }, () =>
		new Float64Array(k).map(() => rand() * 2 - 1),
	);
	const V: Float64Array[] = Array.from({ length: k }, () =>
		new Float64Array(p).map(() => rand() * 2 - 1),
	);
	// Singular values decay
	const S = new Float64Array(k).map((_, i) => {
		const hi = Math.exp(-i / effectiveRank);
		const lo = tailStrength / k;
		return hi * (1 - tailStrength) + lo;
	});
	const X: Float64Array[] = Array.from({ length: n }, (_, i) => {
		const row = new Float64Array(p);
		for (let c = 0; c < k; c++) {
			for (let j = 0; j < p; j++) {
				row[j]! += (U[i]?.[c] ?? 0) * (S[c] ?? 0) * (V[c]?.[j] ?? 0);
			}
		}
		return row;
	});
	return X;
}

/** Generate a multilabel classification dataset. */
export function makeMultilabelClassification(
	nSamples = 100,
	nFeatures = 20,
	nClasses = 5,
	nLabels = 2,
	randomState = 0,
): { X: Float64Array[]; Y: Int32Array[] } {
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const X: Float64Array[] = Array.from({ length: nSamples }, () =>
		new Float64Array(nFeatures).map(() => rand()),
	);
	const Y: Int32Array[] = Array.from({ length: nSamples }, () => {
		const labels = new Int32Array(nClasses);
		// Select nLabels distinct labels
		const chosen = new Set<number>();
		while (chosen.size < Math.min(nLabels, nClasses)) {
			chosen.add(Math.floor(rand() * nClasses));
		}
		for (const c of chosen) labels[c] = 1;
		return labels;
	});
	return { X, Y };
}
