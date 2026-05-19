/**
 * LabelSpreading (Full implementation).
 * Mirrors sklearn.semi_supervised.LabelSpreading.
 */

import { NotFittedError } from "../exceptions.js";

export interface LabelSpreadingOptions {
  kernel?: "knn" | "rbf";
  gamma?: number;
  nNeighbors?: number;
  alpha?: number;
  maxIter?: number;
  tol?: number;
  nJobs?: number | null;
}

/**
 * LabelSpreading model using label propagation algorithm.
 */
export class LabelSpreadingFull {
  kernel: "knn" | "rbf";
  gamma: number;
  nNeighbors: number;
  alpha: number;
  maxIter: number;
  tol: number;

  classes_: Int32Array | null = null;
  labelDistributions_: Float64Array[] | null = null;
  transductionLabels_: Int32Array | null = null;
  nIter_: number = 0;
  private X_: Float64Array[] | null = null;

  constructor(options: LabelSpreadingOptions = {}) {
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 20;
    this.nNeighbors = options.nNeighbors ?? 7;
    this.alpha = options.alpha ?? 0.2;
    this.maxIter = options.maxIter ?? 30;
    this.tol = options.tol ?? 1e-3;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.X_ = X;

    // Find classes (excluding -1 which means unlabeled)
    const labeledIdx: number[] = [];
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      if ((y[i] ?? -1) !== -1) {
        classSet.add(y[i]!);
        labeledIdx.push(i);
      }
    }

    this.classes_ = new Int32Array(Array.from(classSet).sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    const classIndex = new Map<number, number>();
    this.classes_.forEach((c, i) => classIndex.set(c, i));

    // Build affinity matrix
    const W = this._buildAffinityMatrix(X, nFeatures);

    // Normalize W to get transition matrix T
    const T: Float64Array[] = W.map(row => {
      const sum = row.reduce((s, v) => s + v, 0) || 1;
      return new Float64Array(row.map(v => v / sum));
    });

    // Initialize label distributions F
    // Labeled nodes: one-hot; unlabeled: uniform
    let F: Float64Array[] = Array.from({ length: nSamples }, (_, i) => {
      const dist = new Float64Array(nClasses);
      const label = y[i] ?? -1;
      if (label !== -1) {
        const ci = classIndex.get(label) ?? 0;
        dist[ci] = 1;
      } else {
        dist.fill(1 / nClasses);
      }
      return dist;
    });

    // Y0: clamped label matrix
    const Y0: Float64Array[] = F.map(f => new Float64Array(f));

    // Label spreading: F(t+1) = alpha * T * F(t) + (1 - alpha) * Y0
    for (let iter = 0; iter < this.maxIter; iter++) {
      const FNew: Float64Array[] = Array.from({ length: nSamples }, () => new Float64Array(nClasses));

      // T * F
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nSamples; j++) {
          for (let c = 0; c < nClasses; c++) {
            FNew[i]![c] = (FNew[i]![c] ?? 0) + (T[i]?.[j] ?? 0) * (F[j]?.[c] ?? 0);
          }
        }
      }

      // alpha * T * F + (1-alpha) * Y0
      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < nClasses; c++) {
          FNew[i]![c] = this.alpha * (FNew[i]![c] ?? 0) + (1 - this.alpha) * (Y0[i]?.[c] ?? 0);
        }
        // Normalize rows
        const rowSum = FNew[i]!.reduce((s, v) => s + v, 0) || 1;
        for (let c = 0; c < nClasses; c++) FNew[i]![c] = (FNew[i]![c] ?? 0) / rowSum;
      }

      // Check convergence
      let maxChange = 0;
      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < nClasses; c++) {
          maxChange = Math.max(maxChange, Math.abs((FNew[i]?.[c] ?? 0) - (F[i]?.[c] ?? 0)));
        }
      }
      F = FNew;
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.labelDistributions_ = F;
    this.transductionLabels_ = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestClass = 0;
      let bestProb = F[i]?.[0] ?? 0;
      for (let c = 1; c < nClasses; c++) {
        if ((F[i]?.[c] ?? 0) > bestProb) { bestProb = F[i]![c]!; bestClass = c; }
      }
      this.transductionLabels_[i] = this.classes_[bestClass] ?? 0;
    }

    return this;
  }

  private _buildAffinityMatrix(X: Float64Array[], nFeatures: number): Float64Array[] {
    const n = X.length;
    const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));

    if (this.kernel === "rbf") {
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          let dist = 0;
          for (let f = 0; f < nFeatures; f++) dist += ((X[i]?.[f] ?? 0) - (X[j]?.[f] ?? 0)) ** 2;
          const w = Math.exp(-this.gamma * dist);
          W[i]![j] = w;
          W[j]![i] = w;
        }
      }
    } else {
      // kNN kernel
      const k = Math.min(this.nNeighbors, n - 1);
      for (let i = 0; i < n; i++) {
        const dists = X.map((xj, j) => {
          if (i === j) return Infinity;
          let d = 0;
          for (let f = 0; f < nFeatures; f++) d += ((X[i]?.[f] ?? 0) - (xj[f] ?? 0)) ** 2;
          return d;
        });
        const sorted = dists.map((d, j) => ({ d, j })).sort((a, b) => a.d - b.d).slice(0, k);
        for (const { j } of sorted) { W[i]![j] = 1; W[j]![i] = 1; }
      }
    }
    return W;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.X_ || !this.labelDistributions_ || !this.classes_) throw new NotFittedError("LabelSpreadingFull");
    const nClasses = this.classes_.length;
    const nTrain = this.X_.length;
    const nFeatures = this.X_[0]?.length ?? 0;

    return new Int32Array(X.map(xi => {
      // Find nearest training sample and return its transduced label
      let minDist = Infinity;
      let bestIdx = 0;
      for (let j = 0; j < nTrain; j++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) d += ((xi[f] ?? 0) - (this.X_![j]?.[f] ?? 0)) ** 2;
        if (d < minDist) { minDist = d; bestIdx = j; }
      }

      const probs = this.labelDistributions_![bestIdx]!;
      let bestClass = 0;
      let bestProb = probs[0] ?? 0;
      for (let c = 1; c < nClasses; c++) {
        if ((probs[c] ?? 0) > bestProb) { bestProb = probs[c]!; bestClass = c; }
      }
      return this.classes_![bestClass] ?? 0;
    }));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.X_ || !this.labelDistributions_) throw new NotFittedError("LabelSpreadingFull");
    const nTrain = this.X_.length;
    const nFeatures = this.X_[0]?.length ?? 0;

    return X.map(xi => {
      let minDist = Infinity;
      let bestIdx = 0;
      for (let j = 0; j < nTrain; j++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) d += ((xi[f] ?? 0) - (this.X_![j]?.[f] ?? 0)) ** 2;
        if (d < minDist) { minDist = d; bestIdx = j; }
      }
      return new Float64Array(this.labelDistributions_![bestIdx]!);
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }
}
