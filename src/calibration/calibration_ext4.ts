/**
 * Calibration extensions: histogram binning, isotonic calibration.
 * Port of sklearn.calibration extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** Calibration curve (reliability diagram) computation. */
export function calibrationCurveExt(
	yTrue: Int32Array,
	yProb: Float64Array,
	nBins = 5,
	strategy: "uniform" | "quantile" = "uniform",
): { fractionPositive: Float64Array; meanPredictedValue: Float64Array; binCounts: Int32Array } {
	const n = yTrue.length;
	let binEdges: number[];
	if (strategy === "uniform") {
		binEdges = Array.from({ length: nBins + 1 }, (_, k) => k / nBins);
	} else {
		const sorted = Float64Array.from(yProb).sort();
		binEdges = [0];
		for (let k = 1; k < nBins; k++) {
			binEdges.push(sorted[Math.floor((k * n) / nBins)] ?? 0);
		}
		binEdges.push(1);
	}

	const fractionPositive = new Float64Array(nBins);
	const meanPredictedValue = new Float64Array(nBins);
	const binCounts = new Int32Array(nBins);

	for (let i = 0; i < n; i++) {
		const p = yProb[i] ?? 0;
		let bin = nBins - 1;
		for (let k = 0; k < nBins; k++) {
			if (p < (binEdges[k + 1] ?? 1)) {
				bin = k;
				break;
			}
		}
		binCounts[bin]!++;
		fractionPositive[bin]! += yTrue[i] ?? 0;
		meanPredictedValue[bin]! += p;
	}
	for (let k = 0; k < nBins; k++) {
		if ((binCounts[k] ?? 0) > 0) {
			fractionPositive[k]! /= binCounts[k]!;
			meanPredictedValue[k]! /= binCounts[k]!;
		}
	}
	return { fractionPositive, meanPredictedValue, binCounts };
}

/** Temperature scaling calibration. */
export class TemperatureScaling {
	private temperature_ = 1.0;
	private fitted_ = false;

	fit(logits: Float64Array, y: Int32Array): this {
		// Find temperature that minimizes NLL on validation data
		let bestNll = Number.POSITIVE_INFINITY;
		let bestTemp = 1.0;
		for (let t = 0.1; t <= 10.0; t += 0.1) {
			let nll = 0;
			for (let i = 0; i < logits.length; i++) {
				const scaled = (logits[i] ?? 0) / t;
				const p = 1 / (1 + Math.exp(-scaled));
				const label = (y[i] ?? 0) === 1 ? 1 : 0;
				nll -= label * Math.log(Math.max(p, 1e-15)) + (1 - label) * Math.log(Math.max(1 - p, 1e-15));
			}
			nll /= logits.length;
			if (nll < bestNll) {
				bestNll = nll;
				bestTemp = t;
			}
		}
		this.temperature_ = bestTemp;
		this.fitted_ = true;
		return this;
	}

	transform(logits: Float64Array): Float64Array {
		if (!this.fitted_) throw new NotFittedError("TemperatureScaling is not fitted.");
		return new Float64Array(logits.map((l) => 1 / (1 + Math.exp(-(l / this.temperature_)))));
	}

	get temperature(): number {
		return this.temperature_;
	}
}

/** Platt scaling (logistic calibration of SVM scores). */
export class PlattScaling {
	private A_ = 0;
	private B_ = 0;
	private fitted_ = false;

	fit(decisionScores: Float64Array, y: Int32Array): this {
		// Fit logistic regression: P(y=1|score) = sigmoid(A*score + B)
		const n = decisionScores.length;
		// Add Platt's prior correction
		const nPos = y.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
		const nNeg = n - nPos;
		const tPos = (nPos + 1) / (nPos + 2);
		const tNeg = 1 / (nNeg + 2);

		let A = 0;
		let B = Math.log((nNeg + 1) / (nPos + 1));
		const lr = 0.001;
		for (let iter = 0; iter < 100; iter++) {
			let dA = 0;
			let dB = 0;
			for (let i = 0; i < n; i++) {
				const t = (y[i] ?? 0) === 1 ? tPos : tNeg;
				const logit = A * (decisionScores[i] ?? 0) + B;
				const p = 1 / (1 + Math.exp(-logit));
				const err = p - t;
				dA += err * (decisionScores[i] ?? 0);
				dB += err;
			}
			A -= lr * dA / n;
			B -= lr * dB / n;
		}
		this.A_ = A;
		this.B_ = B;
		this.fitted_ = true;
		return this;
	}

	transform(decisionScores: Float64Array): Float64Array {
		if (!this.fitted_) throw new NotFittedError("PlattScaling is not fitted.");
		return new Float64Array(
			decisionScores.map((s) => 1 / (1 + Math.exp(-(this.A_ * s + this.B_)))),
		);
	}
}

/** Compute expected calibration error (ECE). */
export function expectedCalibrationError(
	yTrue: Int32Array,
	yProb: Float64Array,
	nBins = 10,
): number {
	const { fractionPositive, meanPredictedValue, binCounts } = calibrationCurveExt(
		yTrue,
		yProb,
		nBins,
	);
	const n = yTrue.length;
	let ece = 0;
	for (let k = 0; k < nBins; k++) {
		const cnt = binCounts[k] ?? 0;
		if (cnt === 0) continue;
		ece += (cnt / n) * Math.abs((fractionPositive[k] ?? 0) - (meanPredictedValue[k] ?? 0));
	}
	return ece;
}
