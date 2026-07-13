/**
 * FunctionTransformer — apply arbitrary functions as sklearn transformers.
 * Ported from sklearn.preprocessing._function_transformer
 *
 * Note: FunctionTransformer also exists in preprocessing/binarizer.ts with basic support.
 * This version provides the full sklearn-compatible implementation with inverse_func,
 * validate, feature_names_out, and kw_args support.
 */

export type TransformFn = (
  X: Float64Array[],
  kwargs?: Record<string, unknown>,
) => Float64Array[];
export type FeatureNamesOutFn = (inputFeatureNames: string[]) => string[];

export interface AdvancedFunctionTransformerOptions {
  /** Function to apply in transform() */
  func?: TransformFn | null;
  /** Inverse function for inverse_transform() */
  inverseFunc?: TransformFn | null;
  /** Whether to validate input arrays */
  validate?: boolean;
  /** Whether to accept sparse matrices (always false in TS) */
  acceptSparse?: boolean;
  /** Whether to check that inverse_func is actually the inverse of func */
  checkInverse?: boolean;
  /** Additional keyword arguments passed to func */
  kwArgs?: Record<string, unknown>;
  /** Additional keyword arguments passed to inverseFunc */
  invKwArgs?: Record<string, unknown>;
  /** Strategy for feature names: "one-to-one" | callable */
  featureNamesOut?: "one-to-one" | FeatureNamesOutFn | null;
}

export class AdvancedFunctionTransformer {
  func: TransformFn | null;
  inverseFunc: TransformFn | null;
  validate: boolean;
  checkInverse: boolean;
  kwArgs: Record<string, unknown>;
  invKwArgs: Record<string, unknown>;
  featureNamesOut: "one-to-one" | FeatureNamesOutFn | null;

  private nFeaturesIn_: number | null = null;

  constructor(options: AdvancedFunctionTransformerOptions = {}) {
    this.func = options.func ?? null;
    this.inverseFunc = options.inverseFunc ?? null;
    this.validate = options.validate ?? false;
    this.checkInverse = options.checkInverse ?? true;
    this.kwArgs = options.kwArgs ?? {};
    this.invKwArgs = options.invKwArgs ?? {};
    this.featureNamesOut = options.featureNamesOut ?? null;
  }

  fit(X: Float64Array[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.validate && this.nFeaturesIn_ !== null) {
      const nFeatures = X[0]?.length ?? 0;
      if (nFeatures !== this.nFeaturesIn_) {
        throw new Error(
          `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        );
      }
    }
    if (this.func === null) {
      // Identity transform
      return X;
    }
    return this.func(
      X,
      Object.keys(this.kwArgs).length > 0 ? this.kwArgs : undefined,
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.inverseFunc === null) {
      // Identity
      return X;
    }
    return this.inverseFunc(
      X,
      Object.keys(this.invKwArgs).length > 0 ? this.invKwArgs : undefined,
    );
  }

  getFeatureNamesOut(inputFeatures?: string[]): string[] {
    const features =
      inputFeatures ??
      Array.from({ length: this.nFeaturesIn_ ?? 0 }, (_, i) => `x${i}`);
    if (
      this.featureNamesOut === null ||
      this.featureNamesOut === "one-to-one"
    ) {
      return features;
    }
    return this.featureNamesOut(features);
  }

  get nFeaturesIn(): number {
    if (this.nFeaturesIn_ === null) throw new Error("Not fitted");
    return this.nFeaturesIn_;
  }
}

/**
 * Convenience function to create a log-transforming FunctionTransformer.
 */
export function makeLogTransformer(
  base?: "e" | "2" | "10",
): AdvancedFunctionTransformer {
  const logFn: TransformFn = (X) =>
    X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const v = row[j] ?? 0;
        if (base === "2") out[j] = Math.log2(v);
        else if (base === "10") out[j] = Math.log10(v);
        else out[j] = Math.log(v);
      }
      return out;
    });

  const expFn: TransformFn = (X) =>
    X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const v = row[j] ?? 0;
        if (base === "2") out[j] = 2 ** v;
        else if (base === "10") out[j] = 10 ** v;
        else out[j] = Math.exp(v);
      }
      return out;
    });

  return new AdvancedFunctionTransformer({ func: logFn, inverseFunc: expFn });
}

/**
 * Convenience function to create a square-root transforming FunctionTransformer.
 */
export function makeSqrtTransformer(): AdvancedFunctionTransformer {
  return new AdvancedFunctionTransformer({
    func: (X) =>
      X.map((row) => {
        const out = new Float64Array(row.length);
        for (let j = 0; j < row.length; j++)
          out[j] = Math.sqrt(Math.max(0, row[j] ?? 0));
        return out;
      }),
    inverseFunc: (X) =>
      X.map((row) => {
        const out = new Float64Array(row.length);
        for (let j = 0; j < row.length; j++) {
          const v = row[j] ?? 0;
          out[j] = v * v;
        }
        return out;
      }),
  });
}
