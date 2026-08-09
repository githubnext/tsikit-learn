/**
 * Inspection extensions: SHAP integration utilities, feature importance aggregation.
 * Port of sklearn.inspection extensions.
 */

/** Compute permutation feature importance for a regressor/classifier. */
export function permutationImportance(
	X: Float64Array[],
	y: Float64Array | Int32Array,
	scorer: (X: Float64Array[], y: Float64Array | Int32Array) => number,
	nRepeats = 5,
	randomState = 0,
): { importanceMean: Float64Array; importanceStd: Float64Array } {
	const nFeatures = X[0]?.length ?? 0;
	const baseScore = scorer(X, y);
	const importanceMean = new Float64Array(nFeatures);
	const importanceStd = new Float64Array(nFeatures);
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	for (let j = 0; j < nFeatures; j++) {
		const scores = new Float64Array(nRepeats);
		for (let rep = 0; rep < nRepeats; rep++) {
			// Permute feature j
			const permutedX = X.map((row) => new Float64Array(row));
			const perm = Array.from({ length: X.length }, (_, i) => i);
			for (let k = perm.length - 1; k > 0; k--) {
				const idx = Math.floor(rand() * (k + 1));
				const tmp = perm[k]!;
				perm[k] = perm[idx]!;
				perm[idx] = tmp;
			}
			for (let i = 0; i < X.length; i++) {
				permutedX[i]![j] = X[perm[i]!]?.[j] ?? 0;
			}
			scores[rep] = baseScore - scorer(permutedX, y);
		}
		let mean = 0;
		for (let rep = 0; rep < nRepeats; rep++) mean += scores[rep] ?? 0;
		mean /= nRepeats;
		let variance = 0;
		for (let rep = 0; rep < nRepeats; rep++) {
			const d = (scores[rep] ?? 0) - mean;
			variance += d * d;
		}
		importanceMean[j] = mean;
		importanceStd[j] = Math.sqrt(variance / nRepeats);
	}
	return { importanceMean, importanceStd };
}

/** Aggregate feature importances from multiple estimators (ensemble). */
export function aggregateFeatureImportances(
	importances: Float64Array[],
	normalize = true,
): Float64Array {
	const nFeatures = importances[0]?.length ?? 0;
	const agg = new Float64Array(nFeatures);
	for (const imp of importances) {
		for (let j = 0; j < nFeatures; j++) agg[j]! += imp[j] ?? 0;
	}
	if (importances.length > 0) {
		for (let j = 0; j < nFeatures; j++) agg[j]! /= importances.length;
	}
	if (normalize) {
		let total = 0;
		for (let j = 0; j < nFeatures; j++) total += agg[j] ?? 0;
		if (total > 0) for (let j = 0; j < nFeatures; j++) agg[j]! /= total;
	}
	return agg;
}

/** Compute Shapley values approximation using SHAP Kernel method (simplified). */
export function kernelSHAP(
	x: Float64Array,
	model: (X: Float64Array[]) => Float64Array,
	background: Float64Array[],
	nSamples = 50,
	randomState = 0,
): Float64Array {
	const p = x.length;
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	// Background average prediction
	const bgPreds = model(background);
	let bgMean = 0;
	for (let i = 0; i < bgPreds.length; i++) bgMean += bgPreds[i] ?? 0;
	bgMean /= bgPreds.length;
	// Perturb and compute marginal contributions
	const shapValues = new Float64Array(p);
	for (let s = 0; s < nSamples; s++) {
		// Random subset
		const mask = new Uint8Array(p);
		for (let j = 0; j < p; j++) mask[j] = rand() > 0.5 ? 1 : 0;
		const maskSize = mask.reduce((acc, v) => acc + v, 0);
		if (maskSize === 0 || maskSize === p) continue;
		// Build perturbed samples
		const perturbed = background.map((bg) => {
			const row = new Float64Array(p);
			for (let j = 0; j < p; j++) row[j] = (mask[j] ?? 0) === 1 ? (x[j] ?? 0) : (bg[j] ?? 0);
			return row;
		});
		const pertPreds = model(perturbed);
		let pertMean = 0;
		for (const v of pertPreds) pertMean += v;
		pertMean /= pertPreds.length;
		// SHAP kernel weight
		const kernelWeight = (p - 1) / (combinations(p, maskSize) * maskSize * (p - maskSize));
		for (let j = 0; j < p; j++) {
			if ((mask[j] ?? 0) === 1) shapValues[j]! += kernelWeight * (pertMean - bgMean);
		}
	}
	// Normalize
	const shapSum = shapValues.reduce((s, v) => s + v, 0);
	const fullPred = model([x])[0] ?? 0;
	const scale = shapSum === 0 ? 1 : (fullPred - bgMean) / shapSum;
	for (let j = 0; j < p; j++) shapValues[j]! *= scale;
	return shapValues;
}

function combinations(n: number, k: number): number {
	if (k > n) return 0;
	if (k === 0 || k === n) return 1;
	let result = 1;
	for (let i = 0; i < Math.min(k, n - k); i++) {
		result = (result * (n - i)) / (i + 1);
	}
	return result;
}

/** Compute H-statistic for 2-way feature interaction. */
export function hStatistic(
	X: Float64Array[],
	model: (X: Float64Array[]) => Float64Array,
	featureI: number,
	featureJ: number,
): number {
	const n = X.length;
	// Partial dependence approximation
	const pdij = model(X);
	const pdi = X.map((row) => {
		const masked = new Float64Array(row);
		masked[featureJ] = 0; // zero out j
		return model([masked])[0] ?? 0;
	});
	const pdj = X.map((row) => {
		const masked = new Float64Array(row);
		masked[featureI] = 0;
		return model([masked])[0] ?? 0;
	});
	let numerator = 0;
	let denominator = 0;
	for (let i = 0; i < n; i++) {
		const diff = (pdij[i] ?? 0) - (pdi[i] ?? 0) - (pdj[i] ?? 0);
		numerator += diff * diff;
		denominator += (pdij[i] ?? 0) * (pdij[i] ?? 0);
	}
	return denominator === 0 ? 0 : Math.sqrt(numerator / denominator);
}
