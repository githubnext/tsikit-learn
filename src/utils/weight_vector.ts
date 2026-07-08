/**
 * Weight vector for online learning (SGD).
 * Port of sklearn.utils.weight_vector
 */

/**
 * A weight vector for online learning algorithms with L2 regularization.
 * Uses lazy updates (accumulation) to avoid O(d) per update.
 * Port of sklearn.utils._weight_vector.WeightVector
 */
export class WeightVector {
	private w: Float64Array;
	private wDenseSq: number;
	private wScale: number;
	private sqNormScaled: number;
	readonly nFeatures: number;

	constructor(nFeatures: number) {
		this.nFeatures = nFeatures;
		this.w = new Float64Array(nFeatures);
		this.wDenseSq = 0;
		this.wScale = 1.0;
		this.sqNormScaled = 0;
	}

	/** Dot product with a feature vector */
	dot(x: Float64Array): number {
		let result = 0;
		for (let j = 0; j < x.length; j++) result += this.w[j]! * (x[j] ?? 0);
		return result * this.wScale;
	}

	/** Scale-aware update: w += step * x */
	addScaled(x: Float64Array, c: number): void {
		const cScaled = c / this.wScale;
		for (let j = 0; j < x.length; j++) {
			this.w[j]! += cScaled * (x[j] ?? 0);
		}
		// Update squared norm lazily
		let xSq = 0;
		let wdotx = 0;
		for (let j = 0; j < x.length; j++) {
			xSq += (x[j] ?? 0) ** 2;
			wdotx += this.w[j]! * (x[j] ?? 0);
		}
		// sqNorm(w + c/wScale * x) = sqNorm(w) + 2*c/wScale * w.x + (c/wScale)^2 * ||x||^2
		this.sqNormScaled += 2 * cScaled * (wdotx - cScaled * xSq) + cScaled ** 2 * xSq;
	}

	/** Scale all weights: w *= c */
	scale(c: number): void {
		this.wScale *= c;
		this.sqNormScaled *= c * c;
		// Avoid underflow
		if (Math.abs(this.wScale) < 1e-9) this._resetScale();
	}

	private _resetScale(): void {
		for (let j = 0; j < this.nFeatures; j++) this.w[j]! *= this.wScale;
		this.wScale = 1.0;
	}

	/** Get the dense weight vector */
	toArray(): Float64Array {
		const result = new Float64Array(this.nFeatures);
		for (let j = 0; j < this.nFeatures; j++) result[j] = this.w[j]! * this.wScale;
		return result;
	}

	/** Set from array */
	fromArray(weights: Float64Array): void {
		for (let j = 0; j < this.nFeatures; j++) this.w[j] = weights[j]!;
		this.wScale = 1.0;
		this.sqNormScaled = 0;
		for (const w of weights) this.sqNormScaled += w * w;
	}

	/** L2 squared norm */
	get sqNorm(): number {
		return this.sqNormScaled;
	}

	/** Reset to zeros */
	reset(): void {
		this.w = new Float64Array(this.nFeatures);
		this.wScale = 1.0;
		this.sqNormScaled = 0;
	}
}

/**
 * Averaged weight vector for ASGD (Averaged SGD).
 */
export class AveragedWeightVector {
	private current: WeightVector;
	private average: Float64Array;
	private nUpdates: number;
	readonly nFeatures: number;

	constructor(nFeatures: number) {
		this.nFeatures = nFeatures;
		this.current = new WeightVector(nFeatures);
		this.average = new Float64Array(nFeatures);
		this.nUpdates = 0;
	}

	dot(x: Float64Array): number {
		return this.current.dot(x);
	}

	addScaled(x: Float64Array, c: number): void {
		this.current.addScaled(x, c);
		const curr = this.current.toArray();
		this.nUpdates++;
		const t = this.nUpdates;
		// Polyak-Ruppert averaging: a_t = a_{t-1} + (w_t - a_{t-1}) / t
		for (let j = 0; j < this.nFeatures; j++) {
			this.average[j] = (this.average[j] ?? 0) + ((curr[j] ?? 0) - (this.average[j] ?? 0)) / t;
		}
	}

	scale(c: number): void {
		this.current.scale(c);
	}

	toArray(): Float64Array {
		return this.current.toArray();
	}

	get averageWeights(): Float64Array {
		return this.average;
	}
}
