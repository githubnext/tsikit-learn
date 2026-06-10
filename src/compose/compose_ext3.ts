/**
 * FeatureUnion extension, ColumnTransformer, and Pipeline utilities.
 */

export interface Transformer {
  fit(X: Float64Array[], y?: Int32Array | Float64Array): this;
  transform(X: Float64Array[]): Float64Array[];
}

export class FeatureUnionExt {
  private fitted_ = false;

  constructor(private transformers: Array<[string, Transformer]>) {}

  fit(X: Float64Array[], y?: Int32Array | Float64Array): this {
    for (const [, t] of this.transformers) t.fit(X, y);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const outputs = this.transformers.map(([, t]) => t.transform(X));
    return X.map((_, i) => {
      const parts = outputs.map(o => Array.from(o[i] ?? new Float64Array()));
      return new Float64Array(parts.flat());
    });
  }

  fitTransform(X: Float64Array[], y?: Int32Array | Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  get transformerNames(): string[] { return this.transformers.map(([name]) => name); }
}

export interface ColumnSpec {
  name: string;
  transformer: Transformer;
  columns: number[];
}

export class ColumnTransformerExt {
  private fitted_ = false;

  constructor(private specs: ColumnSpec[], private remainderPassthrough = false) {}

  fit(X: Float64Array[], y?: Int32Array | Float64Array): this {
    for (const spec of this.specs) {
      const subset = X.map(row => new Float64Array(spec.columns.map(c => row[c] ?? 0)));
      spec.transformer.fit(subset, y);
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const nCols = X[0]?.length ?? 0;
    const usedCols = new Set(this.specs.flatMap(s => s.columns));
    return X.map((row, i) => {
      const parts: number[] = [];
      for (const spec of this.specs) {
        const subset = new Float64Array(spec.columns.map(c => row[c] ?? 0));
        const transformed = spec.transformer.transform([subset])[0]!;
        parts.push(...Array.from(transformed));
      }
      if (this.remainderPassthrough) {
        for (let c = 0; c < nCols; c++) {
          if (!usedCols.has(c)) parts.push(row[c] ?? 0);
        }
      }
      void i;
      return new Float64Array(parts);
    });
  }
}

export class Pipeline {
  private fitted_ = false;

  constructor(
    private steps: Array<[string, Transformer]>,
    private finalEstimator?: {
      fit(X: Float64Array[], y: Int32Array | Float64Array): void;
      predict(X: Float64Array[]): Float64Array | Int32Array;
    }
  ) {}

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    let current = X;
    for (const [, t] of this.steps) {
      t.fit(current, y);
      current = t.transform(current);
    }
    if (this.finalEstimator) this.finalEstimator.fit(current, y);
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    if (!this.fitted_ || !this.finalEstimator) throw new Error('Not fitted');
    let current = X;
    for (const [, t] of this.steps) current = t.transform(current);
    return this.finalEstimator.predict(current);
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    let current = X;
    for (const [, t] of this.steps) current = t.transform(current);
    return current;
  }
}

export class FunctionTransformer implements Transformer {
  constructor(
    private fn: (X: Float64Array[]) => Float64Array[],
    private inverseFn?: (X: Float64Array[]) => Float64Array[]
  ) {}

  fit(_X: Float64Array[], _y?: Int32Array | Float64Array): this { return this; }
  transform(X: Float64Array[]): Float64Array[] { return this.fn(X); }
  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.inverseFn) throw new Error('No inverse function provided');
    return this.inverseFn(X);
  }
}
