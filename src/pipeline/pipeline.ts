/**
 * Pipeline: chained estimators.
 * Mirrors sklearn.pipeline.Pipeline and make_pipeline.
 */

import { NotFittedError } from "../exceptions.js";

export interface PipelineStep {
  fit?(X: Float64Array[], y?: Float64Array): this;
  transform?(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[], y?: Float64Array): Float64Array[];
  predict?(X: Float64Array[]): Float64Array;
  score?(X: Float64Array[], y: Float64Array): number;
}

export class Pipeline {
  steps: [string, PipelineStep][];

  constructor(steps: [string, PipelineStep][]) {
    this.steps = steps;
  }

  fit(X: Float64Array[], y?: Float64Array): this {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, step] = this.steps[i] as [string, PipelineStep];
      if (step.fitTransform) {
        Xt = step.fitTransform(Xt, y);
      } else {
        step.fit?.(Xt, y);
        Xt = step.transform?.(Xt) ?? Xt;
      }
    }
    const [, lastStep] = this.steps[this.steps.length - 1] as [
      string,
      PipelineStep,
    ];
    if (y !== undefined) {
      lastStep.fit?.(Xt, y);
    } else {
      if (lastStep.fitTransform) {
        lastStep.fitTransform(Xt);
      } else {
        lastStep.fit?.(Xt);
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    let Xt = X;
    for (const [, step] of this.steps) {
      if (!step.transform)
        throw new Error("Step does not have transform method");
      Xt = step.transform(Xt);
    }
    return Xt;
  }

  fitTransform(X: Float64Array[], y?: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  predict(X: Float64Array[]): Float64Array {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, step] = this.steps[i] as [string, PipelineStep];
      if (!step.transform) throw new NotFittedError("Pipeline");
      Xt = step.transform(Xt);
    }
    const [, lastStep] = this.steps[this.steps.length - 1] as [
      string,
      PipelineStep,
    ];
    if (!lastStep.predict) throw new Error("Last step has no predict method");
    return lastStep.predict(Xt);
  }

  score(X: Float64Array[], y: Float64Array): number {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, step] = this.steps[i] as [string, PipelineStep];
      if (!step.transform) throw new NotFittedError("Pipeline");
      Xt = step.transform(Xt);
    }
    const [, lastStep] = this.steps[this.steps.length - 1] as [
      string,
      PipelineStep,
    ];
    if (!lastStep.score) throw new Error("Last step has no score method");
    return lastStep.score(Xt, y);
  }

  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const [name, step] of this.steps) {
      params[name] = step;
    }
    return params;
  }
}

export function makePipeline(...steps: PipelineStep[]): Pipeline {
  return new Pipeline(steps.map((step, i) => [`step_${i}`, step]));
}
