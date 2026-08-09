/**
 * Pipeline extensions: TransformedTargetRegressor, make_pipeline, FeatureUnion extensions.
 */

export interface Transformer {
  fit(X: Float64Array[]): this;
  transform(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[]): Float64Array[];
}

export interface Regressor {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
}

export class TransformedTargetRegressor {
  private regressor_: Regressor | null = null;
  private yMean = 0;
  private yStd = 1;

  constructor(
    private readonly regressorFactory: () => Regressor,
    private readonly funcTransform: (y: Float64Array) => Float64Array = (y) => y,
    private readonly funcInverseTransform: (y: Float64Array) => Float64Array = (y) => y,
    private readonly standardizeTarget = false
  ) {}

  fit(X: Float64Array[], y: Float64Array): this {
    let yT = this.funcTransform(y);
    if (this.standardizeTarget) {
      this.yMean = yT.reduce((a, b) => a + b, 0) / yT.length;
      const variance = yT.reduce((a, b) => a + (b - this.yMean) ** 2, 0) / yT.length;
      this.yStd = Math.sqrt(Math.max(variance, 1e-10));
      yT = new Float64Array(yT.map((v) => (v - this.yMean) / this.yStd));
    }
    this.regressor_ = this.regressorFactory();
    this.regressor_.fit(X, yT);
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.regressor_) throw new Error("Not fitted");
    let pred = this.regressor_.predict(X);
    if (this.standardizeTarget) pred = new Float64Array(pred.map((v) => v * this.yStd + this.yMean));
    return this.funcInverseTransform(pred);
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    const ssTot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    const ssRes = pred.reduce((s, v, i) => s + (v - (y[i] ?? 0)) ** 2, 0);
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}

export class SequentialFeatureSelector {
  private selectedIndices_: number[] = [];
  private nFeaturesIn_ = 0;

  constructor(
    private readonly estimatorFactory: () => { fit: (X: Float64Array[], y: Float64Array | Int32Array) => void; score?: (X: Float64Array[], y: Float64Array | Int32Array) => number },
    private readonly nFeaturesToSelect: number | "auto" = "auto",
    private readonly direction: "forward" | "backward" = "forward",
    private readonly scoringFn?: (y: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number
  ) {
    void this.scoringFn;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    const n = this.nFeaturesIn_;
    const k = this.nFeaturesToSelect === "auto" ? Math.ceil(n / 2) : this.nFeaturesToSelect;
    let remaining = Array.from({ length: n }, (_, i) => i);
    let selected: number[] = [];

    for (let step = 0; step < k; step++) {
      let bestScore = -Number.POSITIVE_INFINITY;
      let bestFeature = -1;
      for (const fi of remaining) {
        const features = [...selected, fi].sort((a, b) => a - b);
        const Xi = X.map((x) => new Float64Array(features.map((f) => x[f] ?? 0)));
        const est = this.estimatorFactory();
        est.fit(Xi, y);
        const score = est.score ? est.score(Xi, y) : 0.5;
        if (score > bestScore) { bestScore = score; bestFeature = fi; }
      }
      if (bestFeature < 0) break;
      selected.push(bestFeature);
      remaining = remaining.filter((f) => f !== bestFeature);
    }
    this.selectedIndices_ = selected.sort((a, b) => a - b);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(this.selectedIndices_.map((f) => x[f] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupport(): boolean[] {
    const result = new Array(this.nFeaturesIn_).fill(false) as boolean[];
    for (const i of this.selectedIndices_) result[i] = true;
    return result;
  }
}

export class ColumnSelector {
  constructor(private readonly columns: number[]) {}

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(this.columns.map((c) => x[c] ?? 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.transform(X); }
}

export class FunctionTransformerExt {
  constructor(
    private readonly fn: (X: Float64Array[]) => Float64Array[],
    private readonly inverseFn?: (X: Float64Array[]) => Float64Array[]
  ) {}

  fit(_X: Float64Array[]): this { return this; }
  transform(X: Float64Array[]): Float64Array[] { return this.fn(X); }
  fitTransform(X: Float64Array[]): Float64Array[] { return this.fn(X); }
  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.inverseFn) throw new Error("No inverse transform defined");
    return this.inverseFn(X);
  }
}
