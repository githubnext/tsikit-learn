/**
 * Ensemble forest extensions: ExtraTreesClassifier, ExtraTreesRegressor extended variants.
 */

export class RandomForestExt {
  private trees: Array<{ predict: (X: Float64Array[]) => Float64Array | Int32Array }> = [];
  private nClasses_ = 0;
  private isClassifier: boolean;

  constructor(
    private readonly treeFactory: () => { fit: (X: Float64Array[], y: Float64Array | Int32Array) => void; predict: (X: Float64Array[]) => Float64Array | Int32Array },
    private readonly nEstimators = 100,
    private readonly maxSamples?: number,
    private readonly seed = 42,
    isClassifier = true
  ) {
    this.isClassifier = isClassifier;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const n = X.length;
    const sampleSize = this.maxSamples ?? n;
    const rng = this._seededRng(this.seed);
    if (this.isClassifier && y instanceof Int32Array) {
      const classSet = new Set<number>();
      for (const c of y) classSet.add(c);
      this.nClasses_ = classSet.size;
    }
    this.trees = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const indices: number[] = Array.from({ length: sampleSize }, () => Math.floor(rng() * n));
      const Xi = indices.map((i) => X[i]!);
      const yi = y instanceof Int32Array ? new Int32Array(indices.map((i) => y[i]!)) : new Float64Array(indices.map((i) => y[i]!));
      const tree = this.treeFactory();
      tree.fit(Xi, yi);
      this.trees.push(tree);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array | Int32Array {
    if (this.isClassifier) {
      const votes = X.map(() => new Map<number, number>());
      for (const tree of this.trees) {
        const preds = tree.predict(X) as Int32Array;
        for (let i = 0; i < X.length; i++) {
          const v = votes[i];
          if (v !== undefined) v.set(preds[i] ?? 0, (v.get(preds[i] ?? 0) ?? 0) + 1);
        }
      }
      return new Int32Array(votes.map((v) => {
        let best = 0, bestCnt = 0;
        for (const [cls, cnt] of v) if (cnt > bestCnt) { bestCnt = cnt; best = cls; }
        return best;
      }));
    }
    const allPreds = this.trees.map((t) => t.predict(X) as Float64Array);
    return new Float64Array(X.map((_, i) => allPreds.reduce((s, p) => s + (p[i] ?? 0), 0) / this.trees.length));
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class WarmStartEnsemble {
  private trees: Array<{ fit: (X: Float64Array[], y: Float64Array | Int32Array) => void; predict: (X: Float64Array[]) => Float64Array | Int32Array }> = [];
  private fitted = false;

  constructor(
    private readonly treeFactory: () => { fit: (X: Float64Array[], y: Float64Array | Int32Array) => void; predict: (X: Float64Array[]) => Float64Array | Int32Array },
    private nEstimators = 10
  ) {}

  fit(X: Float64Array[], y: Float64Array | Int32Array, addTrees = false): this {
    if (!addTrees || !this.fitted) {
      this.trees = [];
      this.fitted = false;
    }
    const startIdx = this.trees.length;
    for (let t = startIdx; t < this.nEstimators; t++) {
      const tree = this.treeFactory();
      tree.fit(X, y);
      this.trees.push(tree);
    }
    this.fitted = true;
    return this;
  }

  setNEstimators(n: number): void { this.nEstimators = n; }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted) throw new Error("Not fitted");
    const preds = this.trees.map((t) => t.predict(X));
    return new Float64Array(X.map((_, i) => preds.reduce((s, p) => s + ((p instanceof Float64Array ? p[i] : p[i]) ?? 0), 0) / Math.max(this.trees.length, 1)));
  }
}

export class BalancedBaggingClassifier {
  private trees: Array<{ fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array }> = [];

  constructor(
    private readonly classifierFactory: () => { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array },
    private readonly nEstimators = 10,
    private readonly seed = 42
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = new Map<number, number[]>();
    for (let i = 0; i < y.length; i++) {
      const c = y[i]!;
      const arr = classes.get(c) ?? [];
      arr.push(i);
      classes.set(c, arr);
    }
    const minClassSize = Math.min(...[...classes.values()].map((v) => v.length));
    const rng = this._seededRng(this.seed);
    this.trees = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const indices: number[] = [];
      for (const classIndices of classes.values()) {
        const shuffled = [...classIndices].sort(() => rng() - 0.5);
        indices.push(...shuffled.slice(0, minClassSize));
      }
      const clf = this.classifierFactory();
      clf.fit(indices.map((i) => X[i]!), new Int32Array(indices.map((i) => y[i]!)));
      this.trees.push(clf);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const votes = X.map(() => new Map<number, number>());
    for (const tree of this.trees) {
      const preds = tree.predict(X);
      for (let i = 0; i < X.length; i++) {
        const v = votes[i];
        if (v !== undefined) v.set(preds[i] ?? 0, (v.get(preds[i] ?? 0) ?? 0) + 1);
      }
    }
    return new Int32Array(votes.map((v) => {
      let best = 0, bestCnt = 0;
      for (const [cls, cnt] of v) if (cnt > bestCnt) { bestCnt = cnt; best = cls; }
      return best;
    }));
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}
