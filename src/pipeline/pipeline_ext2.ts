/**
 * Additional pipeline utilities: TransformerMixin extensions, Pipeline utilities.
 * Mirrors sklearn.pipeline extras.
 */

export type Transformer = {
  fit(X: Float64Array[], y?: Int32Array | Float64Array): Transformer;
  transform(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[], y?: Int32Array | Float64Array): Float64Array[];
};

export type Estimator = {
  fit(X: Float64Array[], y: Int32Array | Float64Array): Estimator;
  predict(X: Float64Array[]): Int32Array | Float64Array;
  score?(X: Float64Array[], y: Int32Array | Float64Array): number;
};

export class PipelineExt {
  steps: Array<{ name: string; transformer: Transformer }>;
  finalEstimator: Estimator;

  constructor(
    steps: Array<{ name: string; transformer: Transformer }>,
    finalEstimator: Estimator,
  ) {
    this.steps = steps;
    this.finalEstimator = finalEstimator;
  }

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    let Xt = X;
    for (const step of this.steps) {
      if (step.transformer.fitTransform) {
        Xt = step.transformer.fitTransform(Xt, y);
      } else {
        step.transformer.fit(Xt, y);
        Xt = step.transformer.transform(Xt);
      }
    }
    this.finalEstimator.fit(Xt, y);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    let Xt = X;
    for (const step of this.steps) {
      Xt = step.transformer.transform(Xt);
    }
    return Xt;
  }

  predict(X: Float64Array[]): Int32Array | Float64Array {
    const Xt = this.transform(X);
    return this.finalEstimator.predict(Xt);
  }

  score(X: Float64Array[], y: Int32Array | Float64Array): number {
    const Xt = this.transform(X);
    return this.finalEstimator.score?.(Xt, y) ?? 0;
  }

  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const step of this.steps) {
      params[step.name] = step.transformer;
    }
    params["final_estimator"] = this.finalEstimator;
    return params;
  }
}

export function makePipeline(
  ...stepsAndEstimator: Array<Transformer | Estimator>
): PipelineExt {
  const transformers = stepsAndEstimator.slice(0, -1) as Transformer[];
  const estimator = stepsAndEstimator[stepsAndEstimator.length - 1] as Estimator;
  const steps = transformers.map((t, i) => ({ name: `step_${i}`, transformer: t }));
  return new PipelineExt(steps, estimator);
}

export class FeatureUnionExt {
  transformerList: Array<{ name: string; transformer: Transformer }>;
  nJobsHint: number;

  constructor(
    transformerList: Array<{ name: string; transformer: Transformer }>,
    options: { nJobsHint?: number } = {},
  ) {
    this.transformerList = transformerList;
    this.nJobsHint = options.nJobsHint ?? 1;
  }

  fit(X: Float64Array[], y?: Int32Array | Float64Array): this {
    for (const t of this.transformerList) {
      if (t.transformer.fitTransform) {
        t.transformer.fitTransform(X, y);
      } else {
        t.transformer.fit(X, y);
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const parts = this.transformerList.map((t) => t.transformer.transform(X));
    if (parts.length === 0) return X;
    const n = X.length;
    return Array.from({ length: n }, (_, i) => {
      const rows = parts.map((p) => p[i] ?? new Float64Array(0));
      const totalLen = rows.reduce((a, r) => a + r.length, 0);
      const out = new Float64Array(totalLen);
      let offset = 0;
      for (const row of rows) {
        out.set(row, offset);
        offset += row.length;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[], y?: Int32Array | Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
