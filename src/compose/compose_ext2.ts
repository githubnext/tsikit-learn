/**
 * Additional compose utilities: pipeline transformers, set_output interface.
 * Port of sklearn.compose extensions.
 */

/**
 * Function transformer — wraps a pure function as a transformer.
 */
export class LambdaTransformer {
  private fn: (X: Float64Array[]) => Float64Array[];
  private inverseFn?: (X: Float64Array[]) => Float64Array[];

  constructor(
    fn: (X: Float64Array[]) => Float64Array[],
    inverseFn?: (X: Float64Array[]) => Float64Array[]
  ) {
    this.fn = fn;
    if (inverseFn !== undefined) this.inverseFn = inverseFn;
  }

  fit(_X: Float64Array[]): this { return this; }
  transform(X: Float64Array[]): Float64Array[] { return this.fn(X); }
  fitTransform(X: Float64Array[]): Float64Array[] { return this.transform(X); }
  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.inverseFn) throw new Error("No inverse function provided");
    return this.inverseFn(X);
  }
}

/**
 * Set-output mixin — adds getFeatureNamesOut support.
 */
export class SetOutputMixin {
  protected outputNames_: string[] | null = null;

  setOutput(transform: "default" | "pandas"): this {
    // Note: in TypeScript we track the output format preference
    this.outputNames_ = null; // reset
    return this;
  }

  getFeatureNamesOut(inputFeatures?: string[]): string[] {
    if (this.outputNames_) return this.outputNames_;
    if (inputFeatures) return inputFeatures;
    throw new Error("Feature names not available — call fit first or provide inputFeatures");
  }
}

/**
 * Metadata router — routes sample weights and other metadata through pipelines.
 */
export class MetadataRouter {
  private routes: Map<string, Set<string>> = new Map();

  addStep(stepName: string, params: string[]): this {
    this.routes.set(stepName, new Set(params));
    return this;
  }

  route(stepName: string, kwargs: Record<string, unknown>): Record<string, unknown> {
    const allowed = this.routes.get(stepName);
    if (!allowed) return {};
    const result: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(kwargs)) {
      if (allowed.has(key)) result[key] = val;
    }
    return result;
  }
}

/**
 * Clone utility — deep-copies an estimator's parameters.
 */
export function cloneEstimator<T extends { getParams?(): Record<string, unknown> }>(
  estimator: (new (params: Record<string, unknown>) => T) & T,
  params?: Record<string, unknown>,
): T {
  const baseParams = estimator.getParams ? estimator.getParams() : {};
  return new estimator({ ...baseParams, ...params });
}

/**
 * Parallel pipeline — runs multiple transformers on the same data in "parallel" (in sequence, TypeScript is single-threaded) and stacks outputs.
 */
export class ParallelTransformerPipeline {
  private transformers: Array<{ name: string; transformer: { fit(X: Float64Array[]): unknown; transform(X: Float64Array[]): Float64Array[] } }>;
  private fitted = false;

  constructor(transformers: Array<{ name: string; transformer: { fit(X: Float64Array[]): unknown; transform(X: Float64Array[]): Float64Array[] } }>) {
    this.transformers = transformers;
  }

  fit(X: Float64Array[]): this {
    for (const { transformer } of this.transformers) transformer.fit(X);
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new Error("ParallelTransformerPipeline not fitted");
    const outputs = this.transformers.map(({ transformer }) => transformer.transform(X));
    // Horizontally stack
    return X.map((_, i) => {
      const cols: number[] = [];
      for (const out of outputs) {
        const row = out[i] ?? new Float64Array(0);
        for (let j = 0; j < row.length; j++) cols.push(row[j] ?? 0);
      }
      return Float64Array.from(cols);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this.transform(X);
  }

  getFeatureNamesOut(): string[] {
    const names: string[] = [];
    for (const { name, transformer } of this.transformers) {
      const t = transformer as { getFeatureNamesOut?: () => string[] };
      if (t.getFeatureNamesOut) {
        names.push(...t.getFeatureNamesOut().map(n => `${name}__${n}`));
      }
    }
    return names;
  }
}
