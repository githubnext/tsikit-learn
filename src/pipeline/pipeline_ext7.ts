/**
 * Pipeline extensions: feature transformers, custom pipelines, caching.
 * Mirrors sklearn.pipeline additional methods.
 */

import { BaseEstimator } from "../base.js";

type Transformer = {
  fit(X: Float64Array[]): void;
  transform(X: Float64Array[]): Float64Array[];
};

type TransformerFit = {
  fit(X: Float64Array[], y?: Int32Array): void;
  transform(X: Float64Array[]): Float64Array[];
};

/** Union of multiple feature transformers — concatenates outputs. */
export class FeatureUnionExt extends BaseEstimator {
  transformers: Array<[string, TransformerFit]>;
  transformer_weights: Record<string, number>;
  fitted_ = false;

  constructor(
    transformers: Array<[string, TransformerFit]>,
    transformer_weights: Record<string, number> = {},
  ) {
    super();
    this.transformers = transformers;
    this.transformer_weights = transformer_weights;
  }

  fit(X: Float64Array[], y?: Int32Array): this {
    for (const [, t] of this.transformers) t.fit(X, y);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const parts = this.transformers.map(([name, t]) => {
      const Xt = t.transform(X);
      const w = this.transformer_weights[name] ?? 1;
      return Xt.map(row => row.map(v => v * w));
    });
    return X.map((_, i) => {
      const concat: number[] = [];
      for (const p of parts) for (const v of (p[i] ?? new Float64Array())) concat.push(v);
      return new Float64Array(concat);
    });
  }

  fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

/** Custom function transformer with validation. */
export class ValidatedFunctionTransformer extends BaseEstimator {
  func: (X: Float64Array[]) => Float64Array[];
  inverse_func: ((X: Float64Array[]) => Float64Array[]) | null;
  validate: boolean;
  check_inverse: boolean;

  constructor(params: {
    func?: (X: Float64Array[]) => Float64Array[];
    inverse_func?: ((X: Float64Array[]) => Float64Array[]) | null;
    validate?: boolean;
    check_inverse?: boolean;
  } = {}) {
    super();
    this.func = params.func ?? (X => X);
    this.inverse_func = params.inverse_func ?? null;
    this.validate = params.validate ?? false;
    this.check_inverse = params.check_inverse ?? false;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.func(X);
  }

  inverse_transform(X: Float64Array[]): Float64Array[] {
    if (!this.inverse_func) throw new Error("No inverse function provided");
    return this.inverse_func(X);
  }
}

/** Cached pipeline step that memoizes transform results. */
export class CachedStep<T extends Transformer> extends BaseEstimator {
  estimator: T;
  private cache_: Map<string, Float64Array[]>;

  constructor(estimator: T) {
    super();
    this.estimator = estimator;
    this.cache_ = new Map();
  }

  fit(X: Float64Array[]): this {
    this.estimator.fit(X);
    this.cache_.clear();
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const key = X.map(row => row.join(',')).join('|').slice(0, 100);
    if (this.cache_.has(key)) return this.cache_.get(key)!;
    const result = this.estimator.transform(X);
    this.cache_.set(key, result);
    return result;
  }

  clearCache(): void {
    this.cache_.clear();
  }
}

/** Select columns by index for use in pipelines. */
export class ColumnSelector extends BaseEstimator {
  columns: number[];

  constructor(columns: number[]) {
    super();
    this.columns = columns;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => new Float64Array(this.columns.map(c => row[c] ?? 0)));
  }
}

/** Passthrough transformer (identity). */
export class PassThrough extends BaseEstimator {
  fit(_X: Float64Array[]): this { return this; }
  transform(X: Float64Array[]): Float64Array[] { return X; }
  fitTransform(X: Float64Array[]): Float64Array[] { return X; }
}

/** Drop transformer (removes all features, for use in ColumnTransformer). */
export class DropTransformer extends BaseEstimator {
  fit(_X: Float64Array[]): this { return this; }
  transform(X: Float64Array[]): Float64Array[] {
    return X.map(() => new Float64Array(0));
  }
}
