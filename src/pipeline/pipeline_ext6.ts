/**
 * Additional pipeline transformers and utilities.
 * Port of sklearn.pipeline extensions.
 */

import { NotFittedError } from "../exceptions.js";

type Transformer = {
  fit(X: Float64Array[], y?: Int32Array): unknown;
  transform(X: Float64Array[]): Float64Array[];
};

/**
 * Caching pipeline that avoids re-fitting expensive transformers.
 */
export class MemorizedPipeline {
  private steps_: Array<{ name: string; transformer: Transformer }>;
  private fitted = false;
  private fitCache_: Map<string, Float64Array[]> = new Map();

  constructor(steps: Array<{ name: string; transformer: Transformer }>) {
    this.steps_ = steps;
  }

  fit(X: Float64Array[], y?: Int32Array): this {
    let currentX = X;
    for (const { name, transformer } of this.steps_) {
      transformer.fit(currentX, y);
      currentX = transformer.transform(currentX);
      this.fitCache_.set(name, currentX);
    }
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("MemorizedPipeline not fitted");
    let currentX = X;
    for (const { transformer } of this.steps_) {
      currentX = transformer.transform(currentX);
    }
    return currentX;
  }

  fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
    this.fit(X, y);
    return this.fitCache_.get(this.steps_[this.steps_.length - 1]?.name ?? "") ?? X;
  }

  getStepOutput(stepName: string): Float64Array[] | undefined {
    return this.fitCache_.get(stepName);
  }
}

/**
 * Sequential feature selector pipeline step.
 */
export class SequentialFeatureSelectorExt6 {
  private nFeatures: number;
  private direction: "forward" | "backward";
  private selectedFeatures_: number[] = [];
  private fitted = false;

  constructor(options: { nFeatures?: number; direction?: "forward" | "backward" } = {}) {
    this.nFeatures = options.nFeatures ?? 5;
    this.direction = options.direction ?? "forward";
  }

  fit(
    X: Float64Array[],
    y: Int32Array,
    scoreFn: (X: Float64Array[], y: Int32Array) => number,
  ): this {
    const nTotal = X[0]?.length ?? 0;
    let selected: number[] = this.direction === "forward" ? [] : Array.from({ length: nTotal }, (_, i) => i);

    while (
      (this.direction === "forward" && selected.length < this.nFeatures) ||
      (this.direction === "backward" && selected.length > this.nFeatures)
    ) {
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestFeature = -1;

      const candidates = this.direction === "forward"
        ? Array.from({ length: nTotal }, (_, i) => i).filter(i => !selected.includes(i))
        : selected;

      for (const feat of candidates) {
        const trial = this.direction === "forward" ? [...selected, feat] : selected.filter(i => i !== feat);
        const subX = X.map(row => Float64Array.from(trial, col => row[col] ?? 0));
        const score = scoreFn(subX, y);
        if (score > bestScore) { bestScore = score; bestFeature = feat; }
      }

      if (bestFeature === -1) break;
      if (this.direction === "forward") selected.push(bestFeature);
      else selected = selected.filter(i => i !== bestFeature);
    }

    this.selectedFeatures_ = selected;
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("SequentialFeatureSelector not fitted");
    return X.map(row => Float64Array.from(this.selectedFeatures_, col => row[col] ?? 0));
  }

  get supportMask(): boolean[] {
    return Array.from({ length: this.selectedFeatures_.length > 0 ? Math.max(...this.selectedFeatures_) + 1 : 0 }, (_, i) => this.selectedFeatures_.includes(i));
  }
}

/**
 * Grid search over pipeline hyperparameters.
 */
export class PipelineGridSearch {
  private bestParams_: Record<string, unknown> = {};
  private bestScore_ = Number.NEGATIVE_INFINITY;
  private fitted = false;

  fitSearch(
    pipelineFactory: (params: Record<string, unknown>) => {
      fit(X: Float64Array[], y: Int32Array): unknown;
      predict(X: Float64Array[]): Int32Array;
    },
    paramGrid: Record<string, unknown[]>,
    X: Float64Array[],
    y: Int32Array,
    scoreFn: (yTrue: Int32Array, yPred: Int32Array) => number,
    nFolds = 3,
  ): this {
    // Generate all parameter combinations
    const keys = Object.keys(paramGrid);
    const combinations: Record<string, unknown>[] = [{}];
    for (const key of keys) {
      const values = paramGrid[key]!;
      const newCombinations: Record<string, unknown>[] = [];
      for (const combo of combinations) {
        for (const val of values) {
          newCombinations.push({ ...combo, [key]: val });
        }
      }
      combinations.splice(0, combinations.length, ...newCombinations);
    }

    const n = X.length;
    const foldSize = Math.floor(n / nFolds);

    for (const params of combinations) {
      let totalScore = 0;
      for (let fold = 0; fold < nFolds; fold++) {
        const testStart = fold * foldSize;
        const testEnd = Math.min(testStart + foldSize, n);
        const trainX = X.filter((_, i) => i < testStart || i >= testEnd);
        const trainY = Int32Array.from(y.filter((_, i) => i < testStart || i >= testEnd));
        const testX = X.slice(testStart, testEnd);
        const testY = Int32Array.from(Array.from(y).slice(testStart, testEnd));

        const pipeline = pipelineFactory(params);
        pipeline.fit(trainX, trainY);
        const preds = pipeline.predict(testX);
        totalScore += scoreFn(testY, preds);
      }
      const avgScore = totalScore / nFolds;
      if (avgScore > this.bestScore_) {
        this.bestScore_ = avgScore;
        this.bestParams_ = params;
      }
    }

    this.fitted = true;
    return this;
  }

  get bestParams(): Record<string, unknown> {
    if (!this.fitted) throw new NotFittedError("PipelineGridSearch not fitted");
    return this.bestParams_;
  }

  get bestScore(): number {
    if (!this.fitted) throw new NotFittedError("PipelineGridSearch not fitted");
    return this.bestScore_;
  }
}
