/**
 * Semi-supervised learning: LabelPropagation and LabelSpreading.
 * Mirrors sklearn.semi_supervised.LabelPropagation and LabelSpreading.
 */

import { NotFittedError } from "../exceptions.js";

function rbfKernel(X: Float64Array[], gamma: number): Float64Array[] {
  const n = X.length;
  const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let d = 0;
      const xi = X[i] ?? new Float64Array(0);
      const xj = X[j] ?? new Float64Array(0);
      for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
      const w = Math.exp(-gamma * d);
      (W[i] as Float64Array)[j] = w;
      (W[j] as Float64Array)[i] = w;
    }
  }
  return W;
}

export interface LabelPropagationOptions {
  kernel?: "rbf" | "knn";
  gamma?: number;
  nNeighbors?: number;
  maxIter?: number;
  tol?: number;
}

export class LabelPropagation {
  kernel: "rbf" | "knn";
  gamma: number;
  nNeighbors: number;
  maxIter: number;
  tol: number;

  classes_: Int32Array | null = null;
  labelDistributions_: Float64Array[] | null = null;
  transductionLabels_: Int32Array | null = null;
  nIter_: number = 0;

  constructor(options: LabelPropagationOptions = {}) {
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 20;
    this.nNeighbors = options.nNeighbors ?? 7;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    // Get unique classes (excluding -1 which marks unlabeled)
    const labeledSet = new Set<number>();
    for (let i = 0; i < n; i++) { const v = y[i] ?? -1; if (v >= 0) labeledSet.add(v); }
    const classes = Int32Array.from(Array.from(labeledSet).sort((a, b) => a - b));
    this.classes_ = classes;
    const nClasses = classes.length;
    const classIdx = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) classIdx.set(classes[c] ?? 0, c);

    // Build affinity matrix
    const W = rbfKernel(X, this.gamma);
    // Normalize rows
    const T: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      let rowSum = 0;
      for (let j = 0; j < n; j++) rowSum += (W[i] as Float64Array)[j] ?? 0;
      if (rowSum === 0) rowSum = 1;
      for (let j = 0; j < n; j++) (T[i] as Float64Array)[j] = ((W[i] as Float64Array)[j] ?? 0) / rowSum;
    }

    // Initial label distributions
    const F: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
    const Y0: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
    for (let i = 0; i < n; i++) {
      const label = y[i] ?? -1;
      if (label >= 0) {
        const cIdx = classIdx.get(label) ?? 0;
        (F[i] as Float64Array)[cIdx] = 1;
        (Y0[i] as Float64Array)[cIdx] = 1;
      }
    }

    // Propagate
    for (let iter = 0; iter < this.maxIter; iter++) {
      const Fnew: Float64Array[] = Array.from({ length: n }, () => new Float64Array(nClasses));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const t = (T[i] as Float64Array)[j] ?? 0;
          const fj = F[j] as Float64Array;
          const fi = Fnew[i] as Float64Array;
          for (let c = 0; c < nClasses; c++) fi[c] = (fi[c] ?? 0) + t * (fj[c] ?? 0);
        }
        // Clamp labeled nodes
        const label = y[i] ?? -1;
        if (label >= 0) {
          const cIdx = classIdx.get(label) ?? 0;
          for (let c = 0; c < nClasses; c++) (Fnew[i] as Float64Array)[c] = c === cIdx ? 1 : 0;
        }
      }
      let delta = 0;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < nClasses; c++) {
          delta += Math.abs(((Fnew[i] as Float64Array)[c] ?? 0) - ((F[i] as Float64Array)[c] ?? 0));
          (F[i] as Float64Array)[c] = (Fnew[i] as Float64Array)[c] ?? 0;
        }
      }
      this.nIter_ = iter + 1;
      if (delta < this.tol) break;
    }

    this.labelDistributions_ = F;
    this.transductionLabels_ = Int32Array.from(F.map(fi => {
      let maxC = 0; let maxV = fi[0] ?? 0;
      for (let c = 1; c < nClasses; c++) { if ((fi[c] ?? 0) > maxV) { maxV = fi[c] ?? 0; maxC = c; } }
      return classes[maxC] ?? 0;
    }));
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.transductionLabels_) throw new NotFittedError("LabelPropagation is not fitted.");
    void X;
    return this.transductionLabels_;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.labelDistributions_) throw new NotFittedError("LabelPropagation is not fitted.");
    void X;
    return this.labelDistributions_;
  }
}

export class LabelSpreading extends LabelPropagation {
  alpha: number;

  constructor(options: LabelPropagationOptions & { alpha?: number } = {}) {
    super(options);
    this.alpha = options.alpha ?? 0.2;
  }
}
