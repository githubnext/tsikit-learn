/**
 * Additional semi-supervised learning: LabelPropagation extensions.
 * Mirrors sklearn.semi_supervised extras.
 */

import { NotFittedError } from "../exceptions.js";

export class SemiSupervisedPropagation {
  kernel: "rbf" | "knn";
  gamma: number;
  nNeighbors: number;
  maxIter: number;
  tol: number;
  alpha: number;

  labelDistributions_: Float64Array[] | null = null;
  classes_: Int32Array | null = null;
  transductionLabels_: Int32Array | null = null;

  constructor(
    options: {
      kernel?: "rbf" | "knn";
      gamma?: number;
      nNeighbors?: number;
      maxIter?: number;
      tol?: number;
      alpha?: number;
    } = {},
  ) {
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 20;
    this.nNeighbors = options.nNeighbors ?? 7;
    this.maxIter = options.maxIter ?? 30;
    this.tol = options.tol ?? 1e-3;
    this.alpha = options.alpha ?? 0.2;
  }

  private _rbfKernel(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.exp(-this.gamma * s);
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const labeled = Array.from({ length: n }, (_, i) => (y[i] ?? -1) !== -1);
    const classes = Array.from(new Set(Array.from(y).filter((v) => v !== -1))).sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;

    // Build affinity matrix
    const W: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) W[i]![j] = this._rbfKernel(X[i] ?? new Float64Array(0), X[j] ?? new Float64Array(0));
      }
    }

    // Row normalize
    const T: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      const rowSum = (W[i] ?? []).reduce((a, b) => a + b, 0);
      for (let j = 0; j < n; j++) T[i]![j] = rowSum > 0 ? (W[i]?.[j] ?? 0) / rowSum : 0;
    }

    // Initialize label distributions
    const F: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
    const Y: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));

    for (let i = 0; i < n; i++) {
      if (labeled[i]) {
        const classIdx = classes.indexOf(y[i] ?? 0);
        if (classIdx >= 0) {
          F[i]![classIdx] = 1;
          Y[i]![classIdx] = 1;
        }
      }
    }

    // Label propagation
    for (let iter = 0; iter < this.maxIter; iter++) {
      const FNew: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const tij = T[i]?.[j] ?? 0;
          if (tij === 0) continue;
          for (let c = 0; c < k; c++) {
            FNew[i]![c] = (FNew[i]?.[c] ?? 0) + tij * (F[j]?.[c] ?? 0);
          }
        }
        // Mix with initial labels
        for (let c = 0; c < k; c++) {
          FNew[i]![c] = (1 - this.alpha) * (FNew[i]?.[c] ?? 0) + this.alpha * (Y[i]?.[c] ?? 0);
        }
        // Normalize row
        const rowSum = Array.from(FNew[i] ?? []).reduce((a, b) => a + b, 0);
        if (rowSum > 0) {
          for (let c = 0; c < k; c++) FNew[i]![c] = (FNew[i]?.[c] ?? 0) / rowSum;
        }
      }

      // Check convergence
      let maxChange = 0;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          maxChange = Math.max(maxChange, Math.abs((FNew[i]?.[c] ?? 0) - (F[i]?.[c] ?? 0)));
        }
      }

      for (let i = 0; i < n; i++) F[i] = FNew[i]!;
      if (maxChange < this.tol) break;
    }

    this.labelDistributions_ = F;

    // Assign labels
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestVal = -1;
      for (let c = 0; c < k; c++) {
        if ((F[i]?.[c] ?? 0) > bestVal) {
          bestVal = F[i]?.[c] ?? 0;
          best = classes[c] ?? c;
        }
      }
      labels[i] = labeled[i] ? (y[i] ?? 0) : best;
    }
    this.transductionLabels_ = labels;

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.transductionLabels_) throw new NotFittedError("SemiSupervisedPropagation is not fitted");
    return this.transductionLabels_.slice(0, X.length) as Int32Array;
  }
}
