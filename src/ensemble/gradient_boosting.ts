/**
 * Gradient Boosting Classifier and Regressor.
 * Mirrors sklearn.ensemble.GradientBoostingClassifier / GradientBoostingRegressor.
 */

import { NotFittedError } from "../exceptions.js";
import { DecisionTreeRegressor } from "../tree/decision_tree.js";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
}

export class GradientBoostingRegressor {
  nEstimators: number;
  learningRate: number;
  maxDepth: number;
  subsample: number;

  estimators_: DecisionTreeRegressor[] | null = null;
  initialPred_: number = 0;

  constructor(
    options: {
      nEstimators?: number;
      learningRate?: number;
      maxDepth?: number;
      subsample?: number;
    } = {},
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.subsample = options.subsample ?? 1.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this.initialPred_ = Array.from(y).reduce((a, b) => a + b, 0) / n;
    const pred = new Float64Array(n).fill(this.initialPred_);

    this.estimators_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const residuals = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        residuals[i] = (y[i] ?? 0) - (pred[i] ?? 0);
      }

      // Subsample
      let sampleIdx: number[];
      if (this.subsample < 1.0) {
        const k = Math.max(1, Math.round(n * this.subsample));
        sampleIdx = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const tmp = sampleIdx[i] as number;
          sampleIdx[i] = sampleIdx[j] as number;
          sampleIdx[j] = tmp;
        }
        sampleIdx = sampleIdx.slice(0, k);
      } else {
        sampleIdx = Array.from({ length: n }, (_, i) => i);
      }

      const XSub = sampleIdx.map((i) => X[i] ?? new Float64Array(0));
      const rSub = new Float64Array(sampleIdx.map((i) => residuals[i] ?? 0));

      const tree = new DecisionTreeRegressor({ maxDepth: this.maxDepth });
      tree.fit(XSub, rSub);
      this.estimators_.push(tree);

      const treePred = tree.predict(X);
      for (let i = 0; i < n; i++) {
        pred[i] = (pred[i] ?? 0) + this.learningRate * (treePred[i] ?? 0);
      }
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.estimators_ === null)
      throw new NotFittedError("GradientBoostingRegressor");
    const pred = new Float64Array(X.length).fill(this.initialPred_);
    for (const tree of this.estimators_) {
      const tp = tree.predict(X);
      for (let i = 0; i < pred.length; i++) {
        pred[i] = (pred[i] ?? 0) + this.learningRate * (tp[i] ?? 0);
      }
    }
    return pred;
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

export class GradientBoostingClassifier {
  nEstimators: number;
  learningRate: number;
  maxDepth: number;

  estimators_: DecisionTreeRegressor[] | null = null;
  initialPred_: number = 0;
  classes_: Float64Array | null = null;

  constructor(
    options: {
      nEstimators?: number;
      learningRate?: number;
      maxDepth?: number;
    } = {},
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort(
      (a, b) => a - b,
    );
    this.classes_ = new Float64Array(uniqueClasses);
    const posClass = uniqueClasses[uniqueClasses.length - 1] ?? 1;

    // Binary cross-entropy
    const yBin = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      yBin[i] = (y[i] ?? 0) === posClass ? 1 : 0;
    }

    const posRate = Array.from(yBin).reduce((a, b) => a + b, 0) / n;
    this.initialPred_ = Math.log((posRate + 1e-10) / (1 - posRate + 1e-10));
    const F = new Float64Array(n).fill(this.initialPred_);

    this.estimators_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const residuals = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = sigmoid(F[i] ?? 0);
        residuals[i] = (yBin[i] ?? 0) - p;
      }

      const tree = new DecisionTreeRegressor({ maxDepth: this.maxDepth });
      tree.fit(X, residuals);
      this.estimators_.push(tree);

      const tp = tree.predict(X);
      for (let i = 0; i < n; i++) {
        F[i] = (F[i] ?? 0) + this.learningRate * (tp[i] ?? 0);
      }
    }

    return this;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (this.estimators_ === null)
      throw new NotFittedError("GradientBoostingClassifier");
    const F = new Float64Array(X.length).fill(this.initialPred_);
    for (const tree of this.estimators_) {
      const tp = tree.predict(X);
      for (let i = 0; i < F.length; i++) {
        F[i] = (F[i] ?? 0) + this.learningRate * (tp[i] ?? 0);
      }
    }
    return Array.from(F).map((f) => {
      const p = sigmoid(f);
      return new Float64Array([1 - p, p]);
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null)
      throw new NotFittedError("GradientBoostingClassifier");
    const classes = this.classes_;
    const proba = this.predictProba(X);
    const posClass = classes[classes.length - 1] ?? 1;
    const negClass = classes[0] ?? 0;
    return new Float64Array(
      proba.map((p) => ((p[1] ?? 0) >= 0.5 ? posClass : negClass)),
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
