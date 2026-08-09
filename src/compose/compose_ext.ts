/**
 * Additional compose utilities: ColumnTransformer extensions, pipeline utilities.
 * Port of sklearn.compose extensions.
 */

import { NotFittedError } from "../exceptions.js";

type Transformer = {
  fit(X: Float64Array[]): Transformer;
  transform(X: Float64Array[]): Float64Array[];
};

/**
 * Passthrough transformer — returns input unchanged.
 */
export class PassthroughTransformer {
  fit(_X: Float64Array[]): this { return this; }
  transform(X: Float64Array[]): Float64Array[] { return X; }
  fitTransform(X: Float64Array[]): Float64Array[] { return X; }
}

/**
 * Column dropper — removes specified columns from the feature matrix.
 */
export class ColumnDropper {
  private colsToDrop: number[];

  constructor(colsToDrop: number[]) {
    this.colsToDrop = colsToDrop.slice().sort((a, b) => a - b);
  }

  fit(_X: Float64Array[]): this { return this; }

  transform(X: Float64Array[]): Float64Array[] {
    const n = X[0]?.length ?? 0;
    const keepCols = Array.from({ length: n }, (_, i) => i).filter(i => !this.colsToDrop.includes(i));
    return X.map(row => Float64Array.from(keepCols, col => row[col] ?? 0));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }
}

/**
 * Feature renamer (for metadata purposes) — wraps a transformer and adds column names.
 */
export class NamedTransformer {
  private transformer: Transformer;
  private featureNames: string[];

  constructor(transformer: Transformer, featureNames: string[]) {
    this.transformer = transformer;
    this.featureNames = featureNames;
  }

  fit(X: Float64Array[]): this {
    this.transformer.fit(X);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.transformer.transform(X);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }

  getFeatureNamesOut(): string[] {
    return this.featureNames;
  }
}

/**
 * Horizontal feature concatenator — joins feature matrices column-wise.
 */
export class HorizontalConcatenator {
  private transformers: Transformer[];
  private fitted = false;

  constructor(transformers: Transformer[]) {
    this.transformers = transformers;
  }

  fit(X: Float64Array[]): this {
    for (const t of this.transformers) t.fit(X);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("HorizontalConcatenator not fitted");
    const parts = this.transformers.map(t => t.transform(X));
    return X.map((_, i) => {
      const cols: number[] = [];
      for (const part of parts) {
        const row = part[i] ?? new Float64Array(0);
        for (let j = 0; j < row.length; j++) cols.push(row[j] ?? 0);
      }
      return Float64Array.from(cols);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Conditional column transformer — applies transformer only if condition is met.
 */
export class ConditionalTransformer {
  private transformer: Transformer;
  private condition: (X: Float64Array[]) => boolean;
  private fitted = false;
  private conditionMet = false;

  constructor(transformer: Transformer, condition: (X: Float64Array[]) => boolean) {
    this.transformer = transformer;
    this.condition = condition;
  }

  fit(X: Float64Array[]): this {
    this.conditionMet = this.condition(X);
    if (this.conditionMet) this.transformer.fit(X);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("ConditionalTransformer not fitted");
    return this.conditionMet ? this.transformer.transform(X) : X;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }
}

/**
 * Sample weight propagator — ensures sample weights flow through a pipeline step.
 */
export class WeightedTransformerWrapper {
  private transformer: Transformer;

  constructor(transformer: Transformer) {
    this.transformer = transformer;
  }

  fit(X: Float64Array[], _sampleWeight?: Float64Array): this {
    this.transformer.fit(X);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.transformer.transform(X);
  }

  fitTransform(X: Float64Array[], sampleWeight?: Float64Array): Float64Array[] {
    this.fit(X, sampleWeight);
    return this.transform(X);
  }
}

/**
 * Feature selector by column index range.
 */
export class ColumnSliceSelector {
  private start: number;
  private end: number | undefined;

  constructor(start: number, end?: number) {
    this.start = start;
    this.end = end;
  }

  fit(_X: Float64Array[]): this { return this; }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => row.slice(this.start, this.end));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }
}
