/**
 * LabelPropagation and LabelSpreading extensions for semi-supervised learning.
 */

function euclidean2(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return d;
}

export class LabelPropagationExt {
  kernel: "rbf" | "knn";
  gamma: number;
  nNeighbors: number;
  maxIter: number;
  tol: number;
  nClasses: number;
  labelDistributions_: Float64Array[] | null = null;
  transductionLabels_: Int32Array | null = null;
  private _X: Float64Array[] | null = null;

  constructor(
    kernel: "rbf" | "knn" = "rbf",
    gamma = 20,
    nNeighbors = 7,
    maxIter = 1000,
    tol = 1e-3,
    nClasses = 2,
  ) {
    this.kernel = kernel;
    this.gamma = gamma;
    this.nNeighbors = nNeighbors;
    this.maxIter = maxIter;
    this.tol = tol;
    this.nClasses = nClasses;
  }

  private _buildAffinity(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    if (this.kernel === "rbf") {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const d2 = euclidean2(X[i] as Float64Array, X[j] as Float64Array);
          (W[i] as Float64Array)[j] = Math.exp(-this.gamma * d2);
        }
        (W[i] as Float64Array)[i] = 0;
      }
    } else {
      // KNN kernel
      for (let i = 0; i < n; i++) {
        const dists = Array.from({ length: n }, (_, j) => ({ j, d: euclidean2(X[i] as Float64Array, X[j] as Float64Array) }));
        dists.sort((a, b) => a.d - b.d);
        for (const { j } of dists.slice(1, this.nNeighbors + 1)) {
          (W[i] as Float64Array)[j] = 1;
          (W[j] as Float64Array)[i] = 1;
        }
      }
    }
    // Row-normalize
    return W.map((row) => {
      const s = row.reduce((a, b) => a + b, 0);
      return s > 0 ? row.map((v) => v / s) : row;
    });
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    this._X = X;
    const T = this._buildAffinity(X);
    // Initialize label distributions
    const F: Float64Array[] = Array.from({ length: n }, () => new Float64Array(this.nClasses));
    const F0: Float64Array[] = Array.from({ length: n }, () => new Float64Array(this.nClasses));
    for (let i = 0; i < n; i++) {
      const label = y[i] ?? -1;
      if (label >= 0) {
        (F[i] as Float64Array)[label] = 1;
        (F0[i] as Float64Array)[label] = 1;
      } else {
        // Unlabeled: uniform initialization
        for (let c = 0; c < this.nClasses; c++) (F[i] as Float64Array)[c] = 1 / this.nClasses;
      }
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDiff = 0;
      const Fnew: Float64Array[] = Array.from({ length: n }, () => new Float64Array(this.nClasses));
      // F_new = T * F
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const tij = (T[i] as Float64Array)[j] ?? 0;
          for (let c = 0; c < this.nClasses; c++) {
            (Fnew[i]! as Float64Array)[c]! += tij * ((F[j] as Float64Array)[c] ?? 0);
          }
        }
      }
      // Clamp labeled nodes back to F0
      for (let i = 0; i < n; i++) {
        const label = y[i] ?? -1;
        if (label >= 0) {
          for (let c = 0; c < this.nClasses; c++) (Fnew[i] as Float64Array)[c] = (F0[i] as Float64Array)[c] ?? 0;
        }
      }
      // Check convergence
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < this.nClasses; c++) {
          maxDiff = Math.max(maxDiff, Math.abs(((Fnew[i] as Float64Array)[c] ?? 0) - ((F[i] as Float64Array)[c] ?? 0)));
        }
        F[i] = Fnew[i] as Float64Array;
      }
      if (maxDiff < this.tol) break;
    }
    this.labelDistributions_ = F;
    this.transductionLabels_ = Int32Array.from(F, (row) => {
      let best = 0, bestVal = -1;
      for (let c = 0; c < row.length; c++) if ((row[c] ?? 0) > bestVal) { bestVal = row[c] ?? 0; best = c; }
      return best;
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this._X || !this.labelDistributions_) throw new Error("Not fitted");
    const allX = this._X;
    return Int32Array.from(X, (x) => {
      // Find nearest training point
      let bestIdx = 0, bestDist = Number.POSITIVE_INFINITY;
      for (let i = 0; i < allX.length; i++) {
        const d = euclidean2(x, allX[i] as Float64Array);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      return this.transductionLabels_?.[bestIdx] ?? 0;
    });
  }
}

export class SelfTrainingClassifier {
  baseEstimator: { fit(X: Float64Array[], y: Int32Array): this; predict(X: Float64Array[]): Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] };
  threshold: number;
  maxIter: number;
  nClasses: number;
  baseEstimator_: { fit(X: Float64Array[], y: Int32Array): this; predict(X: Float64Array[]): Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] } | null = null;
  nIter_: number = 0;

  constructor(
    baseEstimator: { fit(X: Float64Array[], y: Int32Array): this; predict(X: Float64Array[]): Int32Array; predictProba?: (X: Float64Array[]) => Float64Array[] },
    threshold = 0.75,
    maxIter = 10,
    nClasses = 2,
  ) {
    this.baseEstimator = baseEstimator;
    this.threshold = threshold;
    this.maxIter = maxIter;
    this.nClasses = nClasses;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const labeled = Array.from(y).map((v, i) => ({ labeled: v >= 0, i }));
    const labeledIdx = labeled.filter((e) => e.labeled).map((e) => e.i);
    const unlabeledIdx = labeled.filter((e) => !e.labeled).map((e) => e.i);

    let Xl = labeledIdx.map((i) => X[i] as Float64Array);
    let yl = Int32Array.from(labeledIdx, (i) => y[i] ?? 0);
    let unlIdx = [...unlabeledIdx];

    this.baseEstimator.fit(Xl, yl);
    this.baseEstimator_ = this.baseEstimator;

    for (let iter = 0; iter < this.maxIter && unlIdx.length > 0; iter++) {
      const Xun = unlIdx.map((i) => X[i] as Float64Array);
      let pseudoLabels: Int32Array;

      if (this.baseEstimator.predictProba) {
        const probas = this.baseEstimator.predictProba(Xun);
        const highConf = probas.map((p, i) => {
          const maxP = Math.max(...Array.from(p));
          return maxP >= this.threshold ? i : -1;
        }).filter((i) => i >= 0);
        if (highConf.length === 0) break;
        const confX = highConf.map((i) => Xun[i] as Float64Array);
        const confY = Int32Array.from(highConf, (i) => {
          const p = probas[i] as Float64Array;
          let best = 0, bestP = -1;
          for (let c = 0; c < p.length; c++) if ((p[c] ?? 0) > bestP) { bestP = p[c] ?? 0; best = c; }
          return best;
        });
        Xl = [...Xl, ...confX];
        yl = Int32Array.from([...Array.from(yl), ...Array.from(confY)]);
        unlIdx = unlIdx.filter((_, i) => !highConf.includes(i));
      } else {
        pseudoLabels = this.baseEstimator.predict(Xun);
        Xl = [...Xl, ...Xun];
        yl = Int32Array.from([...Array.from(yl), ...Array.from(pseudoLabels)]);
        unlIdx = [];
      }

      this.baseEstimator.fit(Xl, yl);
      this.nIter_ = iter + 1;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.baseEstimator_) throw new Error("Not fitted");
    return this.baseEstimator_.predict(X);
  }
}
