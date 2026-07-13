/**
 * Pipeline utilities — FunctionTransformer, ColumnTransformer, FeatureUnion.
 */

export interface TransformerLike {
  fit(X: Float64Array[]): this;
  transform(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[]): Float64Array[];
}

interface RegressorLike {
  fit(X: Float64Array[], y: Float64Array): RegressorLike;
  predict(X: Float64Array[]): Float64Array;
}

interface TransformerLikeForTarget {
  fit(y: Float64Array[]): TransformerLikeForTarget;
  transform(y: Float64Array[]): Float64Array[];
  inverseTransform(y: Float64Array[]): Float64Array[];
}

export class FunctionTransformer implements TransformerLike {
  func: (X: Float64Array[]) => Float64Array[];
  inverseFunc: ((X: Float64Array[]) => Float64Array[]) | null;
  validate: boolean;

  constructor(
    func: (X: Float64Array[]) => Float64Array[],
    inverseFunc: ((X: Float64Array[]) => Float64Array[]) | null = null,
    validate = false,
  ) {
    this.func = func;
    this.inverseFunc = inverseFunc;
    this.validate = validate;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.func(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.inverseFunc) throw new Error("No inverse function provided");
    return this.inverseFunc(X);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class ColumnTransformer implements TransformerLike {
  transformers: Array<[string, TransformerLike | "passthrough" | "drop", number[]]>;
  remainder: "drop" | "passthrough";
  private _nFeaturesIn: number = 0;
  private _fittedTransformers: Array<{ name: string; transformer: TransformerLike | "passthrough" | "drop"; cols: number[] }> = [];

  constructor(
    transformers: Array<[string, TransformerLike | "passthrough" | "drop", number[]]>,
    remainder: "drop" | "passthrough" = "drop",
  ) {
    this.transformers = transformers;
    this.remainder = remainder;
  }

  fit(X: Float64Array[]): this {
    this._nFeaturesIn = X[0]?.length ?? 0;
    this._fittedTransformers = this.transformers.map(([name, t, cols]) => {
      if (t !== "drop" && t !== "passthrough") {
        const Xcols = X.map((row) => Float64Array.from(cols, (c) => row[c] ?? 0));
        t.fit(Xcols);
      }
      return { name, transformer: t, cols };
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const parts: Float64Array[][] = [];
    const usedCols = new Set<number>();

    for (const { transformer: t, cols } of this._fittedTransformers) {
      for (const c of cols) usedCols.add(c);
      if (t === "drop") continue;
      const Xcols = X.map((row) => Float64Array.from(cols, (c) => row[c] ?? 0));
      parts.push(t === "passthrough" ? Xcols : t.transform(Xcols));
    }

    if (this.remainder === "passthrough") {
      const remainCols = Array.from({ length: this._nFeaturesIn }, (_, i) => i).filter((i) => !usedCols.has(i));
      if (remainCols.length > 0) parts.push(X.map((row) => Float64Array.from(remainCols, (c) => row[c] ?? 0)));
    }

    if (parts.length === 0) return X.map(() => new Float64Array(0));
    return X.map((_, i) => {
      const combined: number[] = [];
      for (const part of parts) combined.push(...Array.from(part[i] ?? []));
      return Float64Array.from(combined);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class FeatureUnion implements TransformerLike {
  transformerList: Array<[string, TransformerLike]>;
  nJobsHint: number;
  private _fittedTransformers: TransformerLike[] = [];

  constructor(transformerList: Array<[string, TransformerLike]>, nJobsHint = 1) {
    this.transformerList = transformerList;
    this.nJobsHint = nJobsHint;
  }

  fit(X: Float64Array[]): this {
    this._fittedTransformers = this.transformerList.map(([, t]) => t.fit(X));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const parts = this._fittedTransformers.map((t) => t.transform(X));
    if (parts.length === 0) return X.map(() => new Float64Array(0));
    return X.map((_, i) => {
      const combined: number[] = [];
      for (const part of parts) combined.push(...Array.from(part[i] ?? []));
      return Float64Array.from(combined);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this._fittedTransformers = this.transformerList.map(([, t]) => {
      if (t.fitTransform) {
        t.fitTransform(X);
        return t;
      }
      return t.fit(X);
    });
    return this.transform(X);
  }

  getParamsNames(): string[] {
    return this.transformerList.map(([name]) => name);
  }
}

export class TransformedTargetRegressor {
  regressor: RegressorLike;
  transformer: TransformerLikeForTarget;
  private _regressor: RegressorLike | null = null;

  constructor(
    regressor: RegressorLike,
    transformer: TransformerLikeForTarget,
  ) {
    this.regressor = regressor;
    this.transformer = transformer;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const yWrapped = Array.from(y, (v) => Float64Array.from([v]));
    this.transformer.fit(yWrapped);
    const yTransformed = this.transformer.transform(yWrapped);
    const yFlat = Float64Array.from(yTransformed, (row) => row[0] ?? 0);
    this._regressor = this.regressor.fit(X, yFlat);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this._regressor) throw new Error("Not fitted");
    const rawPreds = this._regressor.predict(X);
    const wrapped = Array.from(rawPreds, (v) => Float64Array.from([v]));
    const invTransformed = this.transformer.inverseTransform(wrapped);
    return Float64Array.from(invTransformed, (row) => row[0] ?? 0);
  }
}
