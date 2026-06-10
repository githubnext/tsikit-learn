/**
 * Label Propagation extensions and self-training extensions.
 */

export class LabelPropagationExt {
  private labelMatrix_!: Float64Array[];
  private fitted_ = false;
  private nClasses_ = 0;

  constructor(
    private kernel: 'rbf' | 'knn' = 'rbf',
    private gamma = 20.0,
    private nNeighbors = 7,
    private maxIter = 1000,
    private tol = 1e-3,
    private alpha = 0.2
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classSet = Array.from(new Set(Array.from(y).filter(v => v >= 0))).sort((a, b) => a - b);
    this.nClasses_ = classSet.length;
    const classMap = new Map(classSet.map((c, i) => [c, i]));

    // Build affinity matrix
    const W = this._buildAffinity(X);
    // Normalize: D^{-1/2} W D^{-1/2}
    const D = W.map(row => Math.sqrt(row.reduce((s, v) => s + v, 0) + 1e-10));
    const T = W.map((row, i) => new Float64Array(row.map((v, j) => v / ((D[i] ?? 1) * (D[j] ?? 1)))));

    // Initialize label matrix
    const F = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(this.nClasses_);
      const ci = classMap.get(y[i] ?? -1);
      if (ci !== undefined) row[ci] = 1;
      return row;
    });
    const Y0 = F.map(row => new Float64Array(row));
    const labeled = Array.from(y).map(v => v >= 0);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // F = T F (1 - alpha) + alpha Y0
      const newF = Array.from({ length: n }, (_, i) =>
        new Float64Array(this.nClasses_).map((_, c) =>
          (1 - this.alpha) * T[i]!.reduce((s, t, j) => s + t * (F[j]![c] ?? 0), 0)
          + this.alpha * (Y0[i]![c] ?? 0)
        )
      );
      // Clamp labeled
      for (let i = 0; i < n; i++) if (labeled[i]) newF[i] = new Float64Array(Y0[i]!);
      const delta = Math.max(...newF.map((row, i) => row.reduce((s, v, c) => s + (v - (F[i]![c] ?? 0)) ** 2, 0)));
      for (let i = 0; i < n; i++) F[i] = newF[i]!;
      if (Math.sqrt(delta) < this.tol) break;
    }
    this.labelMatrix_ = F;
    this.fitted_ = true;
    return this;
  }

  predict(): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(this.labelMatrix_.map(row => {
      let best = 0;
      for (let c = 1; c < this.nClasses_; c++) if ((row[c] ?? 0) > (row[best] ?? 0)) best = c;
      return best;
    }));
  }

  predictProba(): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.labelMatrix_.map(row => {
      const sum = row.reduce((s, v) => s + v, 0);
      return sum > 0 ? new Float64Array(row.map(v => v / sum)) : new Float64Array(row);
    });
  }

  private _buildAffinity(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    if (this.kernel === 'rbf') {
      return Array.from({ length: n }, (_, i) =>
        new Float64Array(n).map((_, j) => {
          const dist2 = X[i]!.reduce((s, v, k) => s + (v - (X[j]![k] ?? 0)) ** 2, 0);
          return Math.exp(-this.gamma * dist2);
        })
      );
    }
    // KNN
    const W = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const dists = Array.from({ length: n }, (_, j) => ({
        j, d: X[i]!.reduce((s, v, k) => s + (v - (X[j]![k] ?? 0)) ** 2, 0)
      })).sort((a, b) => a.d - b.d);
      for (let k = 1; k <= this.nNeighbors && k < n; k++) {
        W[i]![dists[k]!.j] = 1;
        W[dists[k]!.j]![i] = 1;
      }
    }
    return W;
  }
}

export class SelfTrainingExt {
  private fitted_ = false;
  private allLabels_!: Int32Array;

  constructor(
    private baseClassifier: {
      fit(X: Float64Array[], y: Int32Array): void;
      predict(X: Float64Array[]): Int32Array;
      predictProba?(X: Float64Array[]): Float64Array[];
    },
    private threshold = 0.75,
    private maxIter = 10,
    private k = 10
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    let labels = new Int32Array(y);
    const unlabeled = Array.from(y).map((v, i) => v === -1 ? i : -1).filter(i => i >= 0);

    for (let iter = 0; iter < this.maxIter && unlabeled.length > 0; iter++) {
      const labeledIdx = Array.from(labels).map((v, i) => v >= 0 ? i : -1).filter(i => i >= 0);
      if (labeledIdx.length === 0) break;
      const XLabeled = labeledIdx.map(i => X[i]!);
      const yLabeled = new Int32Array(labeledIdx.map(i => labels[i]!));
      this.baseClassifier.fit(XLabeled, yLabeled);
      if (!this.baseClassifier.predictProba) break;
      const XUnlabeled = unlabeled.map(i => X[i]!);
      const probas = this.baseClassifier.predictProba(XUnlabeled);
      // Select top k most confident predictions
      const confidences = probas.map((p, i) => {
        const maxProb = Math.max(...p);
        return { i, prob: maxProb, label: Array.from(p).indexOf(maxProb) };
      }).filter(c => c.prob >= this.threshold).sort((a, b) => b.prob - a.prob).slice(0, this.k);
      if (confidences.length === 0) break;
      for (const { i, label } of confidences) {
        const origIdx = unlabeled[i]!;
        labels[origIdx] = label;
        unlabeled.splice(i, 1);
      }
    }
    this.allLabels_ = labels;
    const labeledFinal = Array.from(labels).map((v, i) => v >= 0 ? i : -1).filter(i => i >= 0);
    this.baseClassifier.fit(labeledFinal.map(i => X[i]!), new Int32Array(labeledFinal.map(i => labels[i]!)));
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.baseClassifier.predict(X);
  }

  get allLabels(): Int32Array { return this.allLabels_; }
}
