/**
 * Semi-supervised learning extensions: CoTraining, SelfTrainingClassifier extensions.
 */

export class CoTraining {
  private clf1: { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] } | null = null;
  private clf2: { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] } | null = null;

  constructor(
    private readonly classifierFactory1: () => typeof CoTraining.prototype.clf1 & object,
    private readonly classifierFactory2: () => typeof CoTraining.prototype.clf2 & object,
    private readonly k = 5,
    private readonly maxIter = 30
  ) {}

  fit(
    X1Labeled: Float64Array[], X2Labeled: Float64Array[], y: Int32Array,
    X1Unlabeled: Float64Array[], X2Unlabeled: Float64Array[]
  ): this {
    let labels = new Int32Array(y);
    let X1L = [...X1Labeled];
    let X2L = [...X2Labeled];
    let X1U = [...X1Unlabeled];
    let X2U = [...X2Unlabeled];

    this.clf1 = this.classifierFactory1();
    this.clf2 = this.classifierFactory2();

    for (let iter = 0; iter < this.maxIter; iter++) {
      if (X1U.length === 0) break;
      // Train both classifiers
      this.clf1?.fit(X1L, labels);
      this.clf2?.fit(X2L, labels);
      // Each classifier labels k most confident unlabeled points
      const toAdd1 = this._selectConfident(this.clf1!, X1U, this.k);
      const toAdd2 = this._selectConfident(this.clf2!, X2U, this.k);
      const allToAdd = new Set([...toAdd1.indices, ...toAdd2.indices]);
      if (allToAdd.size === 0) break;
      // Add to labeled set
      const pred1 = this.clf1?.predict(X1U) ?? new Int32Array(X1U.length);
      const pred2 = this.clf2?.predict(X2U) ?? new Int32Array(X2U.length);
      const newX1L: Float64Array[] = [];
      const newX2L: Float64Array[] = [];
      const newY: number[] = [];
      const remaining1: Float64Array[] = [];
      const remaining2: Float64Array[] = [];
      for (let i = 0; i < X1U.length; i++) {
        if (toAdd1.indices.has(i)) {
          newX1L.push(X1U[i]!);
          newX2L.push(X2U[i]!);
          newY.push(pred1[i] ?? 0);
        } else if (toAdd2.indices.has(i)) {
          newX1L.push(X1U[i]!);
          newX2L.push(X2U[i]!);
          newY.push(pred2[i] ?? 0);
        } else {
          remaining1.push(X1U[i]!);
          remaining2.push(X2U[i]!);
        }
      }
      X1L = [...X1L, ...newX1L];
      X2L = [...X2L, ...newX2L];
      labels = new Int32Array([...labels, ...newY]);
      X1U = remaining1;
      X2U = remaining2;
    }
    return this;
  }

  private _selectConfident(
    clf: { predict: (X: Float64Array[]) => Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] },
    X: Float64Array[],
    k: number
  ): { indices: Set<number> } {
    const indices = new Set<number>();
    if (X.length === 0) return { indices };
    if (clf.predictProba) {
      const proba = clf.predictProba(X);
      const scored = proba.map((p, i) => ({
        i,
        conf: Math.max(...p),
      })).sort((a, b) => b.conf - a.conf);
      for (let j = 0; j < Math.min(k, scored.length); j++) indices.add(scored[j]!.i);
    } else {
      for (let j = 0; j < Math.min(k, X.length); j++) indices.add(j);
    }
    return { indices };
  }

  predict(X1: Float64Array[], _X2?: Float64Array[]): Int32Array {
    if (!this.clf1) throw new Error("Not fitted");
    return this.clf1.predict(X1);
  }
}

export class LabelPropagationKernel {
  private labels_: Float64Array[] = [];

  constructor(
    private readonly kernel: "rbf" | "knn" = "rbf",
    private readonly gamma = 20.0,
    private readonly nNeighbors = 7,
    private readonly maxIter = 1000,
    private readonly tol = 1e-3
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classSet = new Set<number>();
    for (const c of y) if (c >= 0) classSet.add(c);
    const classes = [...classSet].sort((a, b) => a - b);
    const nClasses = classes.length;
    // Build affinity matrix
    const W = this._buildAffinity(X);
    // Initialize label matrix
    const F: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
    const isLabeled = Array.from(y, (v) => v >= 0);
    for (let i = 0; i < n; i++) {
      if (isLabeled[i]) {
        const ci = classes.indexOf(y[i]!);
        if (ci >= 0) F[i]![ci] = 1;
      }
    }
    const Y0: Float64Array[] = F.map((row) => new Float64Array(row));
    // Row-normalize W
    const D = W.map((row) => row.reduce((a, b) => a + b, 0));
    const T: Float64Array[] = W.map((row, i) => new Float64Array(row.map((v) => v / Math.max(D[i] ?? 1, 1e-10))));
    for (let iter = 0; iter < this.maxIter; iter++) {
      const Fnew: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          for (let c = 0; c < nClasses; c++) {
            Fnew[i]![c] = (Fnew[i]![c] ?? 0) + (T[i]![j] ?? 0) * (F[j]![c] ?? 0);
          }
        }
      }
      for (let i = 0; i < n; i++) {
        if (isLabeled[i]) { for (let c = 0; c < nClasses; c++) Fnew[i]![c] = Y0[i]![c] ?? 0; }
      }
      let diff = 0;
      for (let i = 0; i < n; i++) for (let c = 0; c < nClasses; c++) diff += Math.abs((Fnew[i]![c] ?? 0) - (F[i]![c] ?? 0));
      for (let i = 0; i < n; i++) F[i] = Fnew[i]!;
      if (diff < this.tol) break;
    }
    this.labels_ = F;
    return this;
  }

  private _buildAffinity(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let d = 0;
        const xi = X[i]!;
        const xj = X[j]!;
        for (let f = 0; f < xi.length; f++) d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        const w = this.kernel === "rbf" ? Math.exp(-this.gamma * d) : (d < this.nNeighbors ? 1 : 0);
        W[i]![j] = w;
        W[j]![i] = w;
      }
    }
    return W;
  }

  predict(indices?: number[]): Int32Array {
    const rows = indices ?? this.labels_.map((_, i) => i);
    return new Int32Array(rows.map((i) => {
      const row = this.labels_[i]!;
      let best = 0, bestV = -1;
      for (let c = 0; c < row.length; c++) if ((row[c] ?? 0) > bestV) { bestV = row[c] ?? 0; best = c; }
      return best;
    }));
  }
}
