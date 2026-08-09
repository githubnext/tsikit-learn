/**
 * Additional metrics: Brier score loss, hinge loss, log loss extensions.
 * Port of sklearn.metrics extensions.
 */

/** Compute hinge loss (used by SVMs). */
export function hingeLoss(
	yTrue: Int32Array,
	decisionFunction: Float64Array,
	labels?: Int32Array,
): number {
	const n = yTrue.length;
	let loss = 0;
	for (let i = 0; i < n; i++) {
		const y = (yTrue[i] ?? 0) === 1 ? 1 : -1;
		const margin = 1 - y * (decisionFunction[i] ?? 0);
		loss += Math.max(0, margin);
	}
	return loss / n;
}

/** Compute Brier score loss for probability predictions. */
export function brierScoreLoss(
	yTrue: Int32Array,
	yProb: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let totalWeight = 0;
	let weightedLoss = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const diff = (yTrue[i] ?? 0) - (yProb[i] ?? 0);
		weightedLoss += w * diff * diff;
		totalWeight += w;
	}
	return totalWeight === 0 ? 0 : weightedLoss / totalWeight;
}

/** Compute zero-one loss (fraction of misclassifications). */
export function zeroOneLoss(
	yTrue: Int32Array,
	yPred: Int32Array,
	normalize = true,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let wrong = 0;
	let total = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) wrong += w;
		total += w;
	}
	return normalize ? wrong / total : wrong;
}

/** Compute balanced accuracy score (macro-averaged recall). */
export function balancedAccuracyScore(
	yTrue: Int32Array,
	yPred: Int32Array,
	sampleWeight?: Float64Array,
	adjusted = false,
): number {
	const classes = [...new Set([...yTrue])].sort((a, b) => a - b);
	const nClasses = classes.length;
	let totalRecall = 0;
	for (const cls of classes) {
		let tp = 0;
		let total = 0;
		for (let i = 0; i < yTrue.length; i++) {
			if ((yTrue[i] ?? 0) === cls) {
				total += sampleWeight?.[i] ?? 1;
				if ((yPred[i] ?? 0) === cls) tp += sampleWeight?.[i] ?? 1;
			}
		}
		totalRecall += total === 0 ? 0 : tp / total;
	}
	const balanced = totalRecall / nClasses;
	if (adjusted) return (balanced - 1 / nClasses) / (1 - 1 / nClasses);
	return balanced;
}

/** Compute Jaccard similarity score. */
export function jaccardScore(
	yTrue: Int32Array,
	yPred: Int32Array,
	average: "micro" | "macro" | "weighted" | "binary" | "samples" = "binary",
	sampleWeight?: Float64Array,
): number {
	const classes = [...new Set([...yTrue, ...yPred])].sort((a, b) => a - b);
	if (average === "binary") {
		let tp = 0;
		let fpFn = 0;
		for (let i = 0; i < yTrue.length; i++) {
			const w = sampleWeight?.[i] ?? 1;
			const t = (yTrue[i] ?? 0) === 1;
			const p = (yPred[i] ?? 0) === 1;
			if (t && p) tp += w;
			else if (t || p) fpFn += w;
		}
		return tp + fpFn === 0 ? 0 : tp / (tp + fpFn);
	}
	const perClass = classes.map((cls) => {
		let tp = 0;
		let fpFn = 0;
		let support = 0;
		for (let i = 0; i < yTrue.length; i++) {
			const w = sampleWeight?.[i] ?? 1;
			const t = (yTrue[i] ?? 0) === cls;
			const p = (yPred[i] ?? 0) === cls;
			if (t && p) tp += w;
			else if (t || p) fpFn += w;
			if (t) support += w;
		}
		return { jaccard: tp + fpFn === 0 ? 0 : tp / (tp + fpFn), support };
	});
	if (average === "macro") {
		return perClass.reduce((s, c) => s + c.jaccard, 0) / classes.length;
	}
	// weighted
	const totalSupport = perClass.reduce((s, c) => s + c.support, 0);
	return totalSupport === 0
		? 0
		: perClass.reduce((s, c) => s + c.jaccard * c.support, 0) / totalSupport;
}

/** Compute Cohen's kappa statistic. */
export function cohenKappaScore(
	y1: Int32Array,
	y2: Int32Array,
	labels?: Int32Array,
	weights?: "linear" | "quadratic" | null,
	sampleWeight?: Float64Array,
): number {
	const classes = labels ?? Int32Array.from([...new Set([...y1, ...y2])].sort((a, b) => a - b));
	const n = classes.length;
	const classIdx = new Map<number, number>();
	for (let k = 0; k < n; k++) classIdx.set(classes[k]!, k);

	const confMat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0) as number[]);
	for (let i = 0; i < y1.length; i++) {
		const r = classIdx.get(y1[i] ?? 0);
		const c = classIdx.get(y2[i] ?? 0);
		if (r !== undefined && c !== undefined) {
			confMat[r]![c]! += sampleWeight?.[i] ?? 1;
		}
	}

	let total = 0;
	for (const row of confMat) for (const v of row) total += v;
	if (total === 0) return 0;

	const rowSums = confMat.map((row) => row.reduce((s, v) => s + v, 0));
	const colSums = Array.from({ length: n }, (_, c) =>
		confMat.reduce((s, row) => s + (row[c] ?? 0), 0),
	);

	let pObs = 0;
	let pExp = 0;

	if (!weights) {
		for (let k = 0; k < n; k++) pObs += confMat[k]![k] ?? 0;
		pObs /= total;
		for (let k = 0; k < n; k++) pExp += rowSums[k]! * colSums[k]!;
		pExp /= total * total;
	} else {
		const w: number[][] = Array.from({ length: n }, (_, r) =>
			Array.from({ length: n }, (__, c) => {
				const diff = Math.abs(r - c);
				return weights === "linear" ? diff : diff * diff;
			}),
		);
		const maxW = Math.max(...w.flatMap((row) => row));
		for (let r = 0; r < n; r++)
			for (let c = 0; c < n; c++) w[r]![c] = 1 - (w[r]![c] ?? 0) / (maxW || 1);
		for (let r = 0; r < n; r++)
			for (let c = 0; c < n; c++) {
				pObs += (w[r]![c] ?? 0) * (confMat[r]![c] ?? 0);
				pExp += (w[r]![c] ?? 0) * rowSums[r]! * colSums[c]!;
			}
		pObs /= total;
		pExp /= total * total;
	}

	return pExp === 1 ? 1 : (pObs - pExp) / (1 - pExp);
}

/** Compute Hamming loss (fraction of labels that are incorrectly predicted). */
export function hammingLoss(
	yTrue: Int32Array,
	yPred: Int32Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let wrong = 0;
	let total = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) wrong += w;
		total += w;
	}
	return total === 0 ? 0 : wrong / total;
}
