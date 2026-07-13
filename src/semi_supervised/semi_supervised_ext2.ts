/**
 * Extended semi-supervised learning: SelfTrainingExt, LabelSpreadingExt
 */

export interface BaseClassifier {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export class SelfTrainingExt {
  private baseClassifier: BaseClassifier;
  private threshold: number;
  private maxIter: number;
  private criterion: "threshold" | "k_best";
  private kBest: number;
  nIter_: number = 0;
  labeledIter_: Int32Array | null = null;

  constructor(
    baseClassifier: BaseClassifier,
    threshold = 0.75,
    maxIter = 10,
    criterion: "threshold" | "k_best" = "threshold",
    kBest = 10
  ) {
    this.baseClassifier = baseClassifier;
    this.threshold = threshold;
    this.maxIter = maxIter;
    this.criterion = criterion;
    this.kBest = kBest;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const labels = Int32Array.from(y);
    const labeled = new Int32Array(n);
    const labeledIter = new Int32Array(n).fill(-1);

    for (let i = 0; i < n; i++) {
      if ((y[i] ?? -1) !== -1) { labeled[i] = 1; labeledIter[i] = 0; }
    }

    for (let iter = 1; iter <= this.maxIter; iter++) {
      const labeledIdx = Array.from({ length: n }, (_, i) => i).filter((i) => labeled[i] === 1);
      if (labeledIdx.length === n) break;

      const Xl = labeledIdx.map((i) => X[i]!);
      const yl = new Int32Array(labeledIdx.map((i) => labels[i] ?? 0));

      this.baseClassifier.fit(Xl, yl);

      const unlabeledIdx = Array.from({ length: n }, (_, i) => i).filter((i) => labeled[i] === 0);
      if (unlabeledIdx.length === 0) break;

      const Xu = unlabeledIdx.map((i) => X[i]!);
      const preds = this.baseClassifier.predict(Xu);

      let newlyLabeled = 0;
      if (this.criterion === "threshold" && this.baseClassifier.predictProba) {
        const probas = this.baseClassifier.predictProba(Xu);
        for (let ui = 0; ui < unlabeledIdx.length; ui++) {
          const proba = probas[ui] ?? new Float64Array(0);
          const maxProba = Math.max(...Array.from(proba));
          if (maxProba >= this.threshold) {
            const idx = unlabeledIdx[ui]!;
            labels[idx] = preds[ui] ?? 0;
            labeled[idx] = 1;
            labeledIter[idx] = iter;
            newlyLabeled++;
          }
        }
      } else {
        // k_best: take the k most confident predictions
        for (let ui = 0; ui < Math.min(this.kBest, unlabeledIdx.length); ui++) {
          const idx = unlabeledIdx[ui]!;
          labels[idx] = preds[ui] ?? 0;
          labeled[idx] = 1;
          labeledIter[idx] = iter;
          newlyLabeled++;
        }
      }

      this.nIter_ = iter;
      if (newlyLabeled === 0) break;
    }

    this.labeledIter_ = labeledIter;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return this.baseClassifier.predict(X);
  }
}

export class LabelSpreadingExt {
  private kernel: "rbf" | "knn";
  private gamma: number;
  private nNeighbors: number;
  private alpha: number;
  private maxIter: number;
  private tol: number;
  labelDistributions_: Float64Array[] | null = null;
  classes_: Int32Array | null = null;

  constructor(
    kernel: "rbf" | "knn" = "rbf",
    gamma = 20,
    nNeighbors = 7,
    alpha = 0.2,
    maxIter = 30,
    tol = 1e-3
  ) {
    this.kernel = kernel;
    this.gamma = gamma;
    this.nNeighbors = nNeighbors;
    this.alpha = alpha;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classSet = [...new Set(Array.from(y).filter((v) => v >= 0))].sort((a, b) => a - b);
    const nClasses = classSet.length;
    this.classes_ = new Int32Array(classSet);

    // Build affinity matrix
    const W = this.buildAffinity(X);

    // Normalize: D^{-1/2} W D^{-1/2}
    const degree = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) degree[i]! += W[i]![j] ?? 0;
    const L: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      const di = Math.sqrt(degree[i] ?? 1) || 1;
      for (let j = 0; j < n; j++) {
        const dj = Math.sqrt(degree[j] ?? 1) || 1;
        row[j] = (W[i]![j] ?? 0) / (di * dj);
      }
      return row;
    });

    // Initial label matrix
    const classIndex = new Map(classSet.map((c, i) => [c, i]));
    const Y0: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
    for (let i = 0; i < n; i++) {
      const c = y[i] ?? -1;
      if (c >= 0) { const ci = classIndex.get(c) ?? 0; Y0[i]![ci] = 1; }
    }

    let F = Y0.map((row) => Float64Array.from(row));
    for (let iter = 0; iter < this.maxIter; iter++) {
      const newF: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const lij = L[i]![j] ?? 0;
          for (let c = 0; c < nClasses; c++) {
            newF[i]![c] = (newF[i]![c] ?? 0) + lij * (F[j]![c] ?? 0);
          }
        }
        for (let c = 0; c < nClasses; c++) {
          newF[i]![c] = this.alpha * (newF[i]![c] ?? 0) + (1 - this.alpha) * (Y0[i]![c] ?? 0);
        }
      }
      let delta = 0;
      for (let i = 0; i < n; i++) for (let c = 0; c < nClasses; c++) delta += ((newF[i]![c] ?? 0) - (F[i]![c] ?? 0)) ** 2;
      F = newF as Float64Array<ArrayBuffer>[];
      if (delta < this.tol) break;
    }

    this.labelDistributions_ = F;
    return this;
  }

  private buildAffinity(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    if (this.kernel === "rbf") {
      return Array.from({ length: n }, (_, i) => {
        const row = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          let dist2 = 0;
          for (let k = 0; k < (X[i]?.length ?? 0); k++) dist2 += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
          row[j] = Math.exp(-this.gamma * dist2);
        }
        return row;
      });
    }
    // kNN kernel
    const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const dists = Array.from({ length: n }, (_, j) => {
        let d = 0;
        for (let k = 0; k < (X[i]?.length ?? 0); k++) d += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
        return { j, d };
      }).sort((a, b) => a.d - b.d);
      for (let k = 1; k <= Math.min(this.nNeighbors, n - 1); k++) {
        const neighbor = dists[k]!;
        W[i]![neighbor.j] = 1;
        W[neighbor.j]![i] = 1;
      }
    }
    return W;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.labelDistributions_ || !this.classes_) throw new Error("Not fitted");
    const n = this.labelDistributions_.length;
    const result = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const dist = this.labelDistributions_[i] ?? new Float64Array(0);
      let maxIdx = 0;
      for (let c = 1; c < dist.length; c++) if ((dist[c] ?? 0) > (dist[maxIdx] ?? 0)) maxIdx = c;
      result[i] = this.classes_[maxIdx] ?? 0;
    }
    return result;
  }
}
