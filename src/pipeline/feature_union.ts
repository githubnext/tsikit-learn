/**
 * FeatureUnion — concatenate results of multiple transformer objects.
 * Mirrors sklearn.pipeline.FeatureUnion.
 */

import { checkIsFitted } from "../utils/estimator_checks.js";

export interface TransformerStep {
  name: string;
  transformer: {
    fit(X: Float64Array[], y?: Float64Array | Int32Array | null): unknown;
    transform(X: Float64Array[]): Float64Array[];
    fitTransform?(
      X: Float64Array[],
      y?: Float64Array | Int32Array | null,
    ): Float64Array[];
  };
  weight?: number;
}

export interface FeatureUnionOptions {
  transformerList: TransformerStep[];
  nJobs?: number | null;
  transformerWeights?: Record<string, number> | null;
  verbose?: boolean;
}

/**
 * Concatenates results from a list of transformers side by side.
 */
export class FeatureUnion {
  transformerList: TransformerStep[];
  transformerWeights: Record<string, number> | null;
  verbose: boolean;
  private fitted_: boolean = false;

  constructor(options: FeatureUnionOptions) {
    this.transformerList = options.transformerList;
    this.transformerWeights = options.transformerWeights ?? null;
    this.verbose = options.verbose ?? false;
  }

  fit(X: Float64Array[], y?: Float64Array | Int32Array | null): this {
    for (const step of this.transformerList) {
      step.transformer.fit(X, y ?? null);
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error("FeatureUnion not fitted");
    const parts = this.transformerList.map((step) => {
      const transformed = step.transformer.fitTransform
        ? step.transformer.fitTransform(X)
        : step.transformer.transform(X);
      const weight = this.transformerWeights?.[step.name] ?? step.weight ?? 1.0;
      if (weight !== 1.0) {
        return transformed.map((row) => {
          const out = new Float64Array(row.length);
          for (let j = 0; j < row.length; j++) out[j] = (row[j] ?? 0) * weight;
          return out;
        });
      }
      return transformed;
    });

    return X.map((_, i) => {
      const rows = parts.map((p) => p[i]!);
      const totalLen = rows.reduce((s, r) => s + r.length, 0);
      const out = new Float64Array(totalLen);
      let offset = 0;
      for (const row of rows) {
        out.set(row, offset);
        offset += row.length;
      }
      return out;
    });
  }

  fitTransform(
    X: Float64Array[],
    y?: Float64Array | Int32Array | null,
  ): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getFeatureNamesOut(inputFeatures?: string[]): string[] {
    const names: string[] = [];
    for (const step of this.transformerList) {
      const t = step.transformer as {
        getFeatureNamesOut?: (f?: string[]) => string[];
      };
      if (typeof t.getFeatureNamesOut === "function") {
        const stepNames = t.getFeatureNamesOut(inputFeatures);
        names.push(...stepNames.map((n) => `${step.name}__${n}`));
      }
    }
    return names;
  }
}

/**
 * Shorthand constructor for FeatureUnion.
 */
export function makeUnion(
  ...transformers: Array<{
    name: string;
    transformer: TransformerStep["transformer"];
  }>
): FeatureUnion {
  return new FeatureUnion({
    transformerList: transformers.map((t) => ({
      name: t.name,
      transformer: t.transformer,
    })),
  });
}
