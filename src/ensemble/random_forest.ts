/**
 * Random Forest Classifier and Regressor.
 * Mirrors sklearn.ensemble.RandomForestClassifier / RandomForestRegressor.
 */

import { NotFittedError } from "../exceptions.js";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../tree/decision_tree.js";

function bootstrapSample(n: number): number[] {
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    indices.push(Math.floor(Math.random() * n));
  }
  return indices;
}

export class RandomForestClassifier {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  maxFeatures: number | "sqrt" | "log2";

  estimators_: DecisionTreeClassifier[] | null = null;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      nEstimators?: number;
      maxDepth?: number;
      minSamplesSplit?: number;
      maxFeatures?: number | "sqrt" | "log2";
    } = {},
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
  }

  private _getFeatureSubset(nFeatures: number): number[] {
    let k: number;
    if (this.maxFeatures === "sqrt") k = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    else if (this.maxFeatures === "log2") k = Math.max(1, Math.round(Math.log2(nFeatures)));
    else k = Math.min(nFeatures, this.maxFeatures as number);

    const indices = Array.from({ length: nFeatures }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i] as number;
      indices[i] = indices[j] as number;
      indices[j] = tmp;
    }
    return indices.slice(0, k);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    this.classes_ = new Float64Array(
      Array.from(new Set(Array.from(y))).sort((a, b) => a - b),
    );

    this.estimators_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const sampleIdx = bootstrapSample(n);
      const featIdx = this._getFeatureSubset(nFeatures);

      const XSub = sampleIdx.map((i) => {
        const xi = X[i] ?? new Float64Array(nFeatures);
        return new Float64Array(featIdx.map((f) => xi[f] ?? 0));
      });
      const ySub = new Float64Array(sampleIdx.map((i) => y[i] ?? 0));

      const tree = new DecisionTreeClassifier({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
      });
      tree.fit(XSub, ySub);
      // Store feature indices with tree
      (tree as DecisionTreeClassifier & { featIdx_: number[] }).featIdx_ = featIdx;
      this.estimators_.push(tree);
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_ === null || this.classes_ === null)
      throw new NotFittedError("RandomForestClassifier");

    const classes = this.classes_;
    return new Float64Array(
      X.map((xi) => {
        const votes = new Map<number, number>();
        for (const tree of this.estimators_ as (DecisionTreeClassifier & { featIdx_: number[] })[]) {
          const featIdx = tree.featIdx_;
          const xSub = new Float64Array(featIdx.map((f) => xi[f] ?? 0));
          const pred = (tree.predict([xSub]))[0] ?? 0;
          votes.set(pred, (votes.get(pred) ?? 0) + 1);
        }
        let bestClass = classes[0] ?? 0;
        let bestCount = 0;
        for (const [cls, cnt] of votes) {
          if (cnt > bestCount) {
            bestCount = cnt;
            bestClass = cls;
          }
        }
        return bestClass;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class RandomForestRegressor {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  maxFeatures: number | "sqrt" | "log2";

  estimators_: DecisionTreeRegressor[] | null = null;

  constructor(
    options: {
      nEstimators?: number;
      maxDepth?: number;
      minSamplesSplit?: number;
      maxFeatures?: number | "sqrt" | "log2";
    } = {},
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
  }

  private _getFeatureSubset(nFeatures: number): number[] {
    let k: number;
    if (this.maxFeatures === "sqrt") k = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    else if (this.maxFeatures === "log2") k = Math.max(1, Math.round(Math.log2(nFeatures)));
    else k = Math.min(nFeatures, this.maxFeatures as number);

    const indices = Array.from({ length: nFeatures }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i] as number;
      indices[i] = indices[j] as number;
      indices[j] = tmp;
    }
    return indices.slice(0, k);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;

    this.estimators_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const sampleIdx = bootstrapSample(n);
      const featIdx = this._getFeatureSubset(nFeatures);

      const XSub = sampleIdx.map((i) => {
        const xi = X[i] ?? new Float64Array(nFeatures);
        return new Float64Array(featIdx.map((f) => xi[f] ?? 0));
      });
      const ySub = new Float64Array(sampleIdx.map((i) => y[i] ?? 0));

      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
      });
      tree.fit(XSub, ySub);
      (tree as DecisionTreeRegressor & { featIdx_: number[] }).featIdx_ = featIdx;
      this.estimators_.push(tree);
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_ === null) throw new NotFittedError("RandomForestRegressor");
    return new Float64Array(
      X.map((xi) => {
        let sum = 0;
        for (const tree of this.estimators_ as (DecisionTreeRegressor & { featIdx_: number[] })[]) {
          const featIdx = tree.featIdx_;
          const xSub = new Float64Array(featIdx.map((f) => xi[f] ?? 0));
          sum += (tree.predict([xSub]))[0] ?? 0;
        }
        return sum / (this.estimators_?.length ?? 1);
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}
