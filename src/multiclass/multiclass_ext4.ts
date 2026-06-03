/**
 * Multiclass extensions: Directed Acyclic Graph (DAG) classifier, nested dichotomies.
 * Mirrors sklearn.multiclass additional methods.
 */

import { BaseEstimator } from "../base.js";

type BinaryClassifier = {
  fit(X: Float64Array[], y: Int32Array): void;
  predict(X: Float64Array[]): Int32Array;
  predict_proba?(X: Float64Array[]): Float64Array[];
};

/** Directed Acyclic Graph SVM for multiclass classification. */
export class DAGClassifier extends BaseEstimator {
  estimatorFactory: () => BinaryClassifier;
  estimators_: Map<string, BinaryClassifier> = new Map();
  classes_: Int32Array = new Int32Array(0);

  constructor(estimatorFactory: () => BinaryClassifier) {
    super();
    this.estimatorFactory = estimatorFactory;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    for (let i = 0; i < classes.length; i++) {
      for (let j = i + 1; j < classes.length; j++) {
        const ci = classes[i]!;
        const cj = classes[j]!;
        const idx = Array.from({ length: X.length }, (_, k) => k).filter(k => y[k] === ci || y[k] === cj);
        const Xij = idx.map(k => X[k]!);
        const yij = new Int32Array(idx.map(k => y[k] === ci ? 0 : 1));
        const est = this.estimatorFactory();
        est.fit(Xij, yij);
        this.estimators_.set(`${ci}_${cj}`, est);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => {
      const remaining = [...Array.from(this.classes_)];
      while (remaining.length > 1) {
        const ci = remaining[0]!;
        const cj = remaining[remaining.length - 1]!;
        const key = `${Math.min(ci, cj)}_${Math.max(ci, cj)}`;
        const est = this.estimators_.get(key);
        if (!est) break;
        const pred = est.predict([row])[0] ?? 0;
        const loser = pred === 0 ? cj : ci;
        const loserIdx = remaining.indexOf(loser);
        if (loserIdx >= 0) remaining.splice(loserIdx, 1);
        else break;
      }
      return remaining[0] ?? 0;
    }));
  }
}

/** Error correcting output codes with random codes. */
export class RandomECOC extends BaseEstimator {
  n_estimators: number;
  estimatorFactory: () => BinaryClassifier;
  code_book_: Float64Array[] = [];
  estimators_: BinaryClassifier[] = [];
  classes_: Int32Array = new Int32Array(0);

  constructor(estimatorFactory: () => BinaryClassifier, params: { n_estimators?: number } = {}) {
    super();
    this.estimatorFactory = estimatorFactory;
    this.n_estimators = params.n_estimators ?? 15;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;
    const m = this.n_estimators;

    // Generate random binary code book
    this.code_book_ = Array.from({ length: k }, () =>
      new Float64Array(m).map(() => Math.random() < 0.5 ? 1 : -1),
    );

    for (let t = 0; t < m; t++) {
      const yBin = new Int32Array(y.length).map((_, i) => {
        const ci = classes.indexOf(y[i] ?? 0);
        return (this.code_book_[ci]?.[t] ?? 0) > 0 ? 1 : 0;
      });
      const est = this.estimatorFactory();
      est.fit(X, yBin);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const m = this.n_estimators;
    const k = this.classes_.length;

    const predictions = this.estimators_.map(est => est.predict(X));
    const codeMatrix: Float64Array[] = Array.from({ length: n }, (_, i) =>
      new Float64Array(m).map((_, t) => (predictions[t]?.[i] ?? 0) === 1 ? 1 : -1),
    );

    return new Int32Array(n).map((_, i) => {
      let bestClass = 0;
      let bestDist = Infinity;
      for (let ci = 0; ci < k; ci++) {
        let hamming = 0;
        for (let t = 0; t < m; t++) {
          if ((codeMatrix[i]?.[t] ?? 0) !== (this.code_book_[ci]?.[t] ?? 0)) hamming++;
        }
        if (hamming < bestDist) { bestDist = hamming; bestClass = ci; }
      }
      return this.classes_[bestClass] ?? 0;
    });
  }
}

/** Nested dichotomies for multiclass classification. */
export class NestedDichotomies extends BaseEstimator {
  estimatorFactory: () => BinaryClassifier;
  tree_: NDNode | null = null;
  classes_: Int32Array = new Int32Array(0);

  constructor(estimatorFactory: () => BinaryClassifier) {
    super();
    this.estimatorFactory = estimatorFactory;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    this.tree_ = this._buildTree(X, y, classes);
    return this;
  }

  private _buildTree(X: Float64Array[], y: Int32Array, classes: number[]): NDNode {
    if (classes.length === 1) return { leaf: true, cls: classes[0] ?? 0 };
    const mid = Math.floor(classes.length / 2);
    const left = classes.slice(0, mid);
    const right = classes.slice(mid);
    const leftSet = new Set(left);
    const idx = Array.from({ length: y.length }, (_, i) => i).filter(i => left.includes(y[i] ?? 0) || right.includes(y[i] ?? 0));
    const Xsub = idx.map(i => X[i]!);
    const ysub = new Int32Array(idx.map(i => leftSet.has(y[i] ?? 0) ? 0 : 1));
    const est = this.estimatorFactory();
    est.fit(Xsub, ysub);

    const leftX = X.filter((_, i) => leftSet.has(y[i] ?? 0));
    const leftY = new Int32Array(Array.from(y).filter(v => leftSet.has(v)));
    const rightX = X.filter((_, i) => !leftSet.has(y[i] ?? 0) && classes.includes(y[i] ?? 0));
    const rightY = new Int32Array(Array.from(y).filter(v => right.includes(v)));

    return {
      leaf: false,
      estimator: est,
      left: this._buildTree(leftX, leftY, left),
      right: this._buildTree(rightX, rightY, right),
    };
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => this._predict(row, this.tree_!)));
  }

  private _predict(x: Float64Array, node: NDNode): number {
    if (node.leaf) return node.cls ?? 0;
    const pred = node.estimator!.predict([x])[0] ?? 0;
    return pred === 0 ? this._predict(x, node.left!) : this._predict(x, node.right!);
  }
}

interface NDNode {
  leaf: boolean;
  cls?: number;
  estimator?: BinaryClassifier;
  left?: NDNode;
  right?: NDNode;
}
