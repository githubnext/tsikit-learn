/**
 * Extended model selection: PermutationTestScore, learning curve extensions.
 * Port of sklearn.model_selection extensions.
 */

/** Result of a permutation test. */
export interface PermutationTestResult {
	score: number;
	permutationScores: Float64Array;
	pValue: number;
}

/** Compute a permutation test score. */
export function permutationTestScore(
	X: Float64Array[],
	y: Int32Array,
	scorer: (X: Float64Array[], y: Int32Array) => number,
	nPermutations = 100,
	randomState = 0,
): PermutationTestResult {
	const score = scorer(X, y);
	const permutationScores = new Float64Array(nPermutations);
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	const yPerm = new Int32Array(y);
	for (let p = 0; p < nPermutations; p++) {
		// Fisher-Yates shuffle
		for (let i = yPerm.length - 1; i > 0; i--) {
			const j = Math.floor(rand() * (i + 1));
			const tmp = yPerm[i]!;
			yPerm[i] = yPerm[j]!;
			yPerm[j] = tmp;
		}
		permutationScores[p] = scorer(X, yPerm);
	}
	let count = 0;
	for (let p = 0; p < nPermutations; p++) {
		if ((permutationScores[p] ?? 0) >= score) count++;
	}
	return { score, permutationScores, pValue: (count + 1) / (nPermutations + 1) };
}

/** Compute learning curve data: train sizes, train scores, test scores. */
export interface LearningCurveResult {
	trainSizes: Int32Array;
	trainScores: Float64Array[];
	testScores: Float64Array[];
}

export function computeLearningCurveData(
	nSamples: number,
	trainSizeFractions: number[],
	nCv: number,
	scorer: (trainIdx: Int32Array, testIdx: Int32Array) => { train: number; test: number },
	randomState = 0,
): LearningCurveResult {
	const trainSizes = new Int32Array(
		trainSizeFractions.map((f) => Math.max(1, Math.round(f * nSamples))),
	);
	const trainScores: Float64Array[] = Array.from(trainSizes, () => new Float64Array(nCv));
	const testScores: Float64Array[] = Array.from(trainSizes, () => new Float64Array(nCv));
	let rng = randomState;
	const rand = (): number => {
		rng = (rng * 1664525 + 1013904223) & 0xffffffff;
		return (rng >>> 0) / 0xffffffff;
	};
	for (let cvIdx = 0; cvIdx < nCv; cvIdx++) {
		// Simple random split
		const indices = Array.from({ length: nSamples }, (_, i) => i);
		for (let i = indices.length - 1; i > 0; i--) {
			const j = Math.floor(rand() * (i + 1));
			const tmp = indices[i]!;
			indices[i] = indices[j]!;
			indices[j] = tmp;
		}
		const testSize = Math.max(1, Math.round(nSamples * 0.2));
		const testIdx = new Int32Array(indices.slice(0, testSize));
		for (let si = 0; si < trainSizes.length; si++) {
			const ts = trainSizes[si] ?? 1;
			const trainIdx = new Int32Array(indices.slice(testSize, testSize + ts));
			const { train, test } = scorer(trainIdx, testIdx);
			trainScores[si]![cvIdx] = train;
			testScores[si]![cvIdx] = test;
		}
	}
	return { trainSizes, trainScores, testScores };
}

/** Compute cross-validation predictions (for stacking, etc.). */
export function crossValPredict(
	X: Float64Array[],
	y: Int32Array,
	nFolds: number,
	predictor: (trainX: Float64Array[], trainY: Int32Array, testX: Float64Array[]) => Float64Array,
): Float64Array {
	const n = X.length;
	const predictions = new Float64Array(n);
	const foldSize = Math.floor(n / nFolds);
	for (let fold = 0; fold < nFolds; fold++) {
		const testStart = fold * foldSize;
		const testEnd = fold === nFolds - 1 ? n : testStart + foldSize;
		const trainIdx: number[] = [];
		const testIdx: number[] = [];
		for (let i = 0; i < n; i++) {
			if (i >= testStart && i < testEnd) testIdx.push(i);
			else trainIdx.push(i);
		}
		const trainX = trainIdx.map((i) => X[i]!);
		const trainY = new Int32Array(trainIdx.map((i) => y[i] ?? 0));
		const testX = testIdx.map((i) => X[i]!);
		const preds = predictor(trainX, trainY, testX);
		for (let i = 0; i < testIdx.length; i++) {
			predictions[testIdx[i]!] = preds[i] ?? 0;
		}
	}
	return predictions;
}

/** HalvingRandomSearchCV-style iteration budget computation. */
export function computeSuccessiveHalvingBudget(
	nCandidates: number,
	minResources: number,
	maxResources: number,
	factor = 3,
): Array<{ nCandidates: number; resources: number }> {
	const schedule: Array<{ nCandidates: number; resources: number }> = [];
	let candidates = nCandidates;
	let resources = minResources;
	while (candidates > 0 && resources <= maxResources) {
		schedule.push({ nCandidates: candidates, resources });
		candidates = Math.floor(candidates / factor);
		resources = Math.min(resources * factor, maxResources);
	}
	return schedule;
}
