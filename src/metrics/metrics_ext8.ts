/**
 * Additional metrics: regression metrics extensions.
 * Port of sklearn.metrics.regression extensions.
 */

/** Compute mean pinball loss for quantile regression. */
export function meanPinballLoss(
	yTrue: Float64Array,
	yPred: Float64Array,
	alpha = 0.5,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let loss = 0;
	let totalWeight = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const diff = (yTrue[i] ?? 0) - (yPred[i] ?? 0);
		loss += w * (diff >= 0 ? alpha * diff : (alpha - 1) * diff);
		totalWeight += w;
	}
	return totalWeight === 0 ? 0 : loss / totalWeight;
}

/** Compute mean absolute percentage error (MAPE). */
export function meanAbsolutePercentageError(
	yTrue: Float64Array,
	yPred: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let loss = 0;
	let totalWeight = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const trueVal = yTrue[i] ?? 0;
		if (trueVal === 0) continue;
		loss += w * Math.abs(((yTrue[i] ?? 0) - (yPred[i] ?? 0)) / trueVal);
		totalWeight += w;
	}
	return totalWeight === 0 ? 0 : loss / totalWeight;
}

/** Compute mean squared log error. */
export function meanSquaredLogError(
	yTrue: Float64Array,
	yPred: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let loss = 0;
	let totalWeight = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const diff =
			Math.log1p(Math.max(0, yTrue[i] ?? 0)) -
			Math.log1p(Math.max(0, yPred[i] ?? 0));
		loss += w * diff * diff;
		totalWeight += w;
	}
	return totalWeight === 0 ? 0 : loss / totalWeight;
}

/** Compute D² tweedie score (generalization of R²). */
export function d2TweedieScore(
	yTrue: Float64Array,
	yPred: Float64Array,
	power = 0,
): number {
	const n = yTrue.length;
	let yMean = 0;
	for (let i = 0; i < n; i++) yMean += yTrue[i] ?? 0;
	yMean /= n;

	const tweedieDeviance = (y: number, mu: number): number => {
		if (power === 0) return (y - mu) ** 2;
		if (power === 1) return 2 * (y * Math.log(y / mu) - (y - mu));
		if (power === 2) return 2 * (Math.log(mu / y) + (y - mu) / mu);
		return (
			2 *
			((Math.pow(y, 2 - power) / ((1 - power) * (2 - power))) -
				(y * Math.pow(mu, 1 - power)) / (1 - power) +
				Math.pow(mu, 2 - power) / (2 - power))
		);
	};

	let devRes = 0;
	let devNull = 0;
	for (let i = 0; i < n; i++) {
		devRes += tweedieDeviance(yTrue[i] ?? 0, yPred[i] ?? 0);
		devNull += tweedieDeviance(yTrue[i] ?? 0, yMean);
	}
	return devNull === 0 ? 1 : 1 - devRes / devNull;
}

/** Compute max error (maximum residual error). */
export function maxError(yTrue: Float64Array, yPred: Float64Array): number {
	let maxErr = 0;
	for (let i = 0; i < yTrue.length; i++) {
		const err = Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
		if (err > maxErr) maxErr = err;
	}
	return maxErr;
}

/** Compute median absolute error. */
export function medianAbsoluteError(
	yTrue: Float64Array,
	yPred: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const errors: [number, number][] = [];
	for (let i = 0; i < yTrue.length; i++) {
		errors.push([Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0)), sampleWeight?.[i] ?? 1]);
	}
	errors.sort((a, b) => a[0] - b[0]);
	if (!sampleWeight) {
		const mid = Math.floor(errors.length / 2);
		if (errors.length % 2 === 0) {
			return ((errors[mid - 1]?.[0] ?? 0) + (errors[mid]?.[0] ?? 0)) / 2;
		}
		return errors[mid]?.[0] ?? 0;
	}
	// Weighted median
	let totalW = 0;
	for (const [, w] of errors) totalW += w;
	let cumW = 0;
	for (const [err, w] of errors) {
		cumW += w;
		if (cumW >= totalW / 2) return err;
	}
	return errors[errors.length - 1]?.[0] ?? 0;
}

/** Compute mean Poisson deviance. */
export function meanPoissonDeviance(
	yTrue: Float64Array,
	yPred: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let dev = 0;
	let totalW = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const y = yTrue[i] ?? 0;
		const mu = yPred[i] ?? 0;
		if (mu <= 0) throw new Error("yPred must be positive for Poisson deviance");
		dev += w * 2 * (y * Math.log(y > 0 ? y / mu : 1) - (y - mu));
		totalW += w;
	}
	return totalW === 0 ? 0 : dev / totalW;
}

/** Compute mean gamma deviance. */
export function meanGammaDeviance(
	yTrue: Float64Array,
	yPred: Float64Array,
	sampleWeight?: Float64Array,
): number {
	const n = yTrue.length;
	let dev = 0;
	let totalW = 0;
	for (let i = 0; i < n; i++) {
		const w = sampleWeight?.[i] ?? 1;
		const y = yTrue[i] ?? 0;
		const mu = yPred[i] ?? 0;
		dev += w * 2 * (Math.log(mu / y) + y / mu - 1);
		totalW += w;
	}
	return totalW === 0 ? 0 : dev / totalW;
}
