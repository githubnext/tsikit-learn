/**
 * Pipeline extensions: TransformingPipeline, FeatureUnionExt.
 * Mirrors sklearn.pipeline advanced utilities.
 */

import { BaseEstimator } from "../base.js";

type Transformer = {
  fit(X: Float64Array[], y?: Float64Array | Int32Array): unknown;
  transform(X: Float64Array[]): Float64Array[];
  fit_transform?(X: Float64Array[], y?: Float64Array | Int32Array): Float64Array[];
};

type Estimator = {
  fit(X: Float64Array[], y: Float64Array | Int32Array): unknown;
  predict(X: Float64Array[]): Float64Array | Int32Array;
  score(X: Float64Array[], y: Float64Array | Int32Array): number;
};

export interface SelectivePipelineParams {
  memory?: boolean;
  verbose?: boolean;
}

/** SelectivePipeline: pipeline that allows disabling specific steps. */
export class SelectivePipeline extends BaseEstimator {
  steps: Array<[string, Transformer | Estimator]>;
  memory: boolean;
  verbose: boolean;
  named_steps: Record<string, Transformer | Estimator>;
  disabled_steps: Set<string> = new Set();

  constructor(
    steps: Array<[string, Transformer | Estimator]>,
    params: SelectivePipelineParams = {},
  ) {
    super();
    this.steps = steps;
    this.memory = params.memory ?? false;
    this.verbose = params.verbose ?? false;
    this.named_steps = Object.fromEntries(steps);
  }

  disable_step(name: string): this {
    this.disabled_steps.add(name);
    return this;
  }

  enable_step(name: string): this {
    this.disabled_steps.delete(name);
    return this;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [name, step] = this.steps[i]!;
      if (this.disabled_steps.has(name)) continue;
      const t = step as Transformer;
      if (t.fit_transform) Xt = t.fit_transform(Xt, y);
      else { t.fit(Xt, y); Xt = t.transform(Xt); }
    }
    const [lastName, lastStep] = this.steps[this.steps.length - 1]!;
    if (!this.disabled_steps.has(lastName)) (lastStep as Estimator).fit(Xt, y);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    let Xt = X;
    for (const [name, step] of this.steps) {
      if (this.disabled_steps.has(name)) continue;
      const t = step as Transformer;
      if (t.transform) Xt = t.transform(Xt);
    }
    return Xt;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [name, step] = this.steps[i]!;
      if (this.disabled_steps.has(name)) continue;
      Xt = (step as Transformer).transform(Xt);
    }
    const [lastName, lastStep] = this.steps[this.steps.length - 1]!;
    if (this.disabled_steps.has(lastName)) return new Float64Array(Xt.map(() => 0));
    return (lastStep as Estimator).predict(Xt);
  }

  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [name, step] = this.steps[i]!;
      if (this.disabled_steps.has(name)) continue;
      Xt = (step as Transformer).transform(Xt);
    }
    const [lastName, lastStep] = this.steps[this.steps.length - 1]!;
    if (this.disabled_steps.has(lastName)) return 0;
    return (lastStep as Estimator).score(Xt, y);
  }
}

export interface HeterogeneousEnsemblePipelineParams {
  voting?: "hard" | "soft";
  weights?: number[] | null;
}

/** HeterogeneousEnsemblePipeline: ensemble of diverse pipelines. */
export class HeterogeneousEnsemblePipeline extends BaseEstimator {
  pipelines: Array<[string, { fit(X: Float64Array[], y: Int32Array): unknown; predict(X: Float64Array[]): Int32Array; score(X: Float64Array[], y: Int32Array): number }]>;
  voting: "hard" | "soft";
  weights: number[] | null;

  constructor(
    pipelines: Array<[string, { fit(X: Float64Array[], y: Int32Array): unknown; predict(X: Float64Array[]): Int32Array; score(X: Float64Array[], y: Int32Array): number }]>,
    params: HeterogeneousEnsemblePipelineParams = {},
  ) {
    super();
    this.pipelines = pipelines;
    this.voting = params.voting ?? "hard";
    this.weights = params.weights ?? null;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    for (const [, pipeline] of this.pipelines) pipeline.fit(X, y);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const preds = this.pipelines.map(([, p]) => p.predict(X));
    const n = X.length;
    const classes = new Set<number>();
    for (const pred of preds) for (const c of pred) classes.add(c);
    const classArr = [...classes].sort((a, b) => a - b);
    return new Int32Array(n).map((_, i) => {
      const votes = new Map<number, number>();
      for (let e = 0; e < preds.length; e++) {
        const c = preds[e]?.[i] ?? 0;
        const w = this.weights?.[e] ?? 1;
        votes.set(c, (votes.get(c) ?? 0) + w);
      }
      let best = classArr[0] ?? 0, bestVotes = 0;
      for (const [c, v] of votes) if (v > bestVotes) { best = c; bestVotes = v; }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}
