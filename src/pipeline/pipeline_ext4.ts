/**
 * Pipeline extensions: TransformerMixin with fit_transform chaining,
 * make_pipeline helper, FunctionTransformer extensions.
 * Port of sklearn.pipeline extensions.
 */

import { NotFittedError } from "../exceptions.js";

/** A step in a pipeline. */
export interface PipelineStepExt {
	name: string;
	transform: (X: Float64Array[]) => Float64Array[];
	fit?: (X: Float64Array[], y?: Int32Array) => void;
	fitTransform?: (X: Float64Array[], y?: Int32Array) => Float64Array[];
}

/** Feature union with weights for combining transformers. */
export class FeatureUnionWeighted {
	private fitted_ = false;
	readonly transformers: Array<{ name: string; transformer: PipelineStepExt; weight: number }>;

	constructor(
		transformers: Array<{
			name: string;
			transformer: PipelineStepExt;
			weight?: number;
		}>,
	) {
		this.transformers = transformers.map((t) => ({
			name: t.name,
			transformer: t.transformer,
			weight: t.weight ?? 1.0,
		}));
	}

	fit(X: Float64Array[], y?: Int32Array): this {
		for (const { transformer } of this.transformers) {
			if (transformer.fit) transformer.fit(X, y);
		}
		this.fitted_ = true;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.fitted_) throw new NotFittedError("FeatureUnionWeighted is not fitted.");
		const outputs = this.transformers.map(({ transformer, weight }) => ({
			features: transformer.transform(X),
			weight,
		}));
		// Concatenate along feature axis
		return X.map((_, i) => {
			const parts: number[] = [];
			for (const { features, weight } of outputs) {
				const row = features[i];
				if (row) {
					for (let j = 0; j < row.length; j++) parts.push((row[j] ?? 0) * weight);
				}
			}
			return new Float64Array(parts);
		});
	}

	fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
		return this.fit(X, y).transform(X);
	}
}

/** Apply a function transformer to data. */
export class FunctionTransformerExt4 {
	private fitted_ = false;
	readonly func: (X: Float64Array[]) => Float64Array[];
	readonly inverseFunc: ((X: Float64Array[]) => Float64Array[]) | undefined;
	readonly validate: boolean;

	constructor(options: {
		func: (X: Float64Array[]) => Float64Array[];
		inverseFunc?: (X: Float64Array[]) => Float64Array[];
		validate?: boolean;
	}) {
		this.func = options.func;
		if (options.inverseFunc !== undefined) this.inverseFunc = options.inverseFunc;
		this.validate = options.validate ?? false;
	}

	fit(X: Float64Array[]): this {
		if (this.validate) {
			for (const row of X) {
				for (let j = 0; j < row.length; j++) {
					if (!Number.isFinite(row[j] ?? 0)) throw new Error("Input contains non-finite values");
				}
			}
		}
		this.fitted_ = true;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.fitted_) throw new NotFittedError("FunctionTransformerExt4 is not fitted.");
		return this.func(X);
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}

	inverseTransform(X: Float64Array[]): Float64Array[] {
		if (!this.inverseFunc) throw new Error("No inverse function provided");
		return this.inverseFunc(X);
	}
}

/** Create a pipeline from a list of transformers and a final estimator. */
export function makePipelineExt(
	steps: PipelineStepExt[],
): {
	fit: (X: Float64Array[], y?: Int32Array) => void;
	transform: (X: Float64Array[]) => Float64Array[];
	fitTransform: (X: Float64Array[], y?: Int32Array) => Float64Array[];
} {
	let fitted = false;
	return {
		fit(X: Float64Array[], y?: Int32Array): void {
			let current = X;
			for (const step of steps) {
				if (step.fit) step.fit(current, y);
				current = step.transform(current);
			}
			fitted = true;
		},
		transform(X: Float64Array[]): Float64Array[] {
			if (!fitted) throw new NotFittedError("Pipeline is not fitted.");
			let current = X;
			for (const step of steps) {
				current = step.transform(current);
			}
			return current;
		},
		fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
			let current = X;
			for (const step of steps) {
				if (step.fitTransform) {
					current = step.fitTransform(current, y);
				} else {
					if (step.fit) step.fit(current, y);
					current = step.transform(current);
				}
			}
			fitted = true;
			return current;
		},
	};
}

/** Column selector for selecting specific columns from a 2D array. */
export class ColumnSelectorExt {
	private fitted_ = false;
	readonly columns: number[];

	constructor(columns: number[]) {
		this.columns = columns;
	}

	fit(_X: Float64Array[]): this {
		this.fitted_ = true;
		return this;
	}

	transform(X: Float64Array[]): Float64Array[] {
		if (!this.fitted_) throw new NotFittedError("ColumnSelectorExt is not fitted.");
		return X.map((row) => new Float64Array(this.columns.map((c) => row[c] ?? 0)));
	}

	fitTransform(X: Float64Array[]): Float64Array[] {
		return this.fit(X).transform(X);
	}
}
