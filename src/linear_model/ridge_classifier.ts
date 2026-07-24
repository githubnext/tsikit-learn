/**
 * RidgeClassifier and RidgeClassifierCV.
 * Mirrors sklearn.linear_model.RidgeClassifier and RidgeClassifierCV.
 */

import { NotFittedError } from "../exceptions.js";
import { addDiagonal, gramMatrix, safeDot, xtDotY } from "../utils/extmath.js";
import { checkArray, checkXy } from "../utils/validation.js";

function choleskyLinsolve(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  // Cholesky decomposition in-place copy
  const L: Float64Array[] = Array.from(
    { length: n },
    (_, i) => new Float64Array(A[i]!),
  );
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < j; k++)
      L[j]![j]! -= (L[j]![k]! ?? 0) * (L[j]![k]! ?? 0);
    L[j]![j]! = Math.sqrt(Math.max(L[j]![j]! ?? 0, 1e-14));
    const diag = L[j]![j]! ?? 1;
    for (let i = j + 1; i < n; i++) {
      for (let k = 0; k < j; k++)
        L[i]![j]! -= (L[i]![k]! ?? 0) * (L[j]![k]! ?? 0);
      L[i]![j]! /= diag;
    }
  }
  // Forward substitution Ly = b
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i] ?? 0;
    for (let k = 0; k < i; k++) s -= (L[i]![k]! ?? 0) * (y[k] ?? 0);
    y[i]! = s / (L[i]![i]! ?? 1);
  }
  // Backward substitution L^T x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i] ?? 0;
    for (let k = i + 1; k < n; k++) s -= (L[k]![i]! ?? 0) * (x[k] ?? 0);
    x[i]! = s / (L[i]![i]! ?? 1);
  }
  return x;
}

export interface RidgeClassifierOptions {
  alpha?: number;
  fit_intercept?: boolean;
  copy_X?: boolean;
  max_iter?: number;
  tol?: number;
  class_weight?: Record<number, number> | "balanced" | null;
  solver?: "auto" | "cholesky";
}

/**
 * Classifier using Ridge regression.
 * Converts multiclass to multi-output regression via 1-of-K encoding, then
 * applies ridge regression and argmax for final class assignment.
 */
export class RidgeClassifier {
  alpha: number;
  fit_intercept: boolean;
  copy_X: boolean;
  max_iter: number;
  tol: number;
  class_weight: Record<number, number> | "balanced" | null;
  solver: "auto" | "cholesky";

  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  classes_: Int32Array | null = null;

  constructor(options: RidgeClassifierOptions = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.fit_intercept = options.fit_intercept ?? true;
    this.copy_X = options.copy_X ?? true;
    this.max_iter = options.max_iter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.class_weight = options.class_weight ?? null;
    this.solver = options.solver ?? "auto";
  }

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    checkXy(X, y);
    const n = X.length;
    const p = X[0]!.length;

    // Discover classes
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(y[i] ?? 0);
    const classes = Int32Array.from([...classSet].sort((a, b) => a - b));
    this.classes_ = classes;
    const k = classes.length;

    // Sample weights (class_weight handling)
    const sampleWeights = new Float64Array(n).fill(1);
    if (this.class_weight === "balanced") {
      const counts = new Map<number, number>();
      for (let i = 0; i < n; i++)
        counts.set(y[i] ?? 0, (counts.get(y[i] ?? 0) ?? 0) + 1);
      for (let i = 0; i < n; i++) {
        const c = y[i] ?? 0;
        sampleWeights[i]! = n / (k * (counts.get(c) ?? 1));
      }
    } else if (this.class_weight !== null) {
      for (let i = 0; i < n; i++)
        sampleWeights[i]! = this.class_weight[y[i] ?? 0] ?? 1;
    }

    // Build indicator matrix Y [n x k] (−1 / +1 encoding)
    const Y: Float64Array[] = Array.from({ length: n }, () =>
      new Float64Array(k).fill(-1),
    );
    for (let i = 0; i < n; i++) {
      const ci = classes.indexOf(y[i] ?? 0);
      if (ci >= 0) Y[i]![ci]! = sampleWeights[i]! * 2 - 1;
    }

    // Weighted X
    const Xw = X.map((row, i) => {
      const w = Math.sqrt(sampleWeights[i]!);
      return Float64Array.from(row, (v) => v * w);
    });

    // Center X if fit_intercept
    const xMean = new Float64Array(p);
    if (this.fit_intercept) {
      for (let i = 0; i < n; i++)
        for (let j = 0; j < p; j++) xMean[j]! += Xw[i]![j]! ?? 0;
      for (let j = 0; j < p; j++) xMean[j]! /= n;
      for (let i = 0; i < n; i++)
        for (let j = 0; j < p; j++) Xw[i]![j]! -= xMean[j]!;
    }

    // Gram matrix + ridge
    const G = gramMatrix(Xw);
    addDiagonal(G, this.alpha);

    // Solve for each output
    this.coef_ = Array.from({ length: k }, (_, ci) => {
      const rhs = xtDotY(
        Xw,
        Float64Array.from({ length: n }, (_, i) => Y[i]![ci]! ?? 0),
      );
      return choleskyLinsolve(G, rhs);
    });

    if (this.fit_intercept) {
      this.intercept_ = new Float64Array(k);
      for (let ci = 0; ci < k; ci++) {
        let yMean = 0;
        for (let i = 0; i < n; i++) yMean += Y[i]![ci]! ?? 0;
        yMean /= n;
        let dot = 0;
        for (let j = 0; j < p; j++)
          dot += (this.coef_[ci]![j]! ?? 0) * (xMean[j]! ?? 0);
        this.intercept_[ci]! = yMean - dot;
      }
    } else {
      this.intercept_ = new Float64Array(k);
    }

    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array[] {
    if (!this.coef_ || !this.intercept_ || !this.classes_)
      throw new NotFittedError("RidgeClassifier is not fitted");
    checkArray(X);
    return X.map((row) => {
      const scores = new Float64Array(this.classes_!.length);
      for (let ci = 0; ci < this.classes_!.length; ci++) {
        let s = this.intercept_![ci]! ?? 0;
        for (let j = 0; j < row.length; j++)
          s += (this.coef_![ci]![j]! ?? 0) * (row[j]! ?? 0);
        scores[ci]! = s;
      }
      return scores;
    });
  }

  predict(X: Float64Array[]): Int32Array {
    const decisions = this.decisionFunction(X);
    const classes = this.classes_!;
    return Int32Array.from(decisions, (scores) => {
      let best = 0;
      for (let ci = 1; ci < scores.length; ci++)
        if (
          (scores[ci]! ?? Number.NEGATIVE_INFINITY) >
          (scores[best]! ?? Number.NEGATIVE_INFINITY)
        )
          best = ci;
      return classes[best]! ?? 0;
    });
  }

  score(X: Float64Array[], y: Int32Array | Float64Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

export interface RidgeClassifierCVOptions {
  alphas?: number[];
  fit_intercept?: boolean;
  class_weight?: Record<number, number> | "balanced" | null;
  scoring?: null;
  store_cv_values?: boolean;
  cv?: number;
}

/**
 * Ridge classifier with built-in leave-one-out or k-fold CV for alpha selection.
 */
export class RidgeClassifierCV {
  alphas: number[];
  fit_intercept: boolean;
  class_weight: Record<number, number> | "balanced" | null;
  cv: number;

  alpha_: number | null = null;
  best_score_: number | null = null;
  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  classes_: Int32Array | null = null;
  cv_values_: Float64Array[] | null = null;

  constructor(options: RidgeClassifierCVOptions = {}) {
    this.alphas = options.alphas ?? [0.1, 1.0, 10.0];
    this.fit_intercept = options.fit_intercept ?? true;
    this.class_weight = options.class_weight ?? null;
    this.cv = options.cv ?? 5;
  }

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    checkXy(X, y);
    const n = X.length;
    const foldSize = Math.floor(n / this.cv);

    let bestAlpha = this.alphas[0]!;
    let bestScore = Number.NEGATIVE_INFINITY;
    const scores = new Float64Array(this.alphas.length);

    for (let ai = 0; ai < this.alphas.length; ai++) {
      const alpha = this.alphas[ai]!;
      let totalScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const start = fold * foldSize;
        const end = fold === this.cv - 1 ? n : start + foldSize;
        const valIdx = Array.from({ length: end - start }, (_, i) => start + i);
        const trainIdx: number[] = [];
        for (let i = 0; i < n; i++) if (i < start || i >= end) trainIdx.push(i);
        const Xtrain = trainIdx.map((i) => X[i]!);
        const ytrain = Int32Array.from(trainIdx, (i) => y[i] ?? 0);
        const Xval = valIdx.map((i) => X[i]!);
        const yval = Int32Array.from(valIdx, (i) => y[i] ?? 0);
        const clf = new RidgeClassifier({
          alpha,
          fit_intercept: this.fit_intercept,
          class_weight: this.class_weight,
        });
        clf.fit(Xtrain, ytrain);
        totalScore += clf.score(Xval, yval);
      }
      scores[ai]! = totalScore / this.cv;
      if (scores[ai]! > bestScore) {
        bestScore = scores[ai]!;
        bestAlpha = alpha;
      }
    }

    this.alpha_ = bestAlpha;
    this.best_score_ = bestScore;
    this.cv_values_ = [scores];

    // Refit on all data with best alpha
    const best = new RidgeClassifier({
      alpha: bestAlpha,
      fit_intercept: this.fit_intercept,
      class_weight: this.class_weight,
    });
    best.fit(X, y);
    this.coef_ = best.coef_;
    this.intercept_ = best.intercept_;
    this.classes_ = best.classes_;

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.coef_ || !this.classes_)
      throw new NotFittedError("RidgeClassifierCV is not fitted");
    const clf = new RidgeClassifier({ alpha: this.alpha_! });
    clf.coef_ = this.coef_;
    clf.intercept_ = this.intercept_;
    clf.classes_ = this.classes_;
    return clf.predict(X);
  }

  score(X: Float64Array[], y: Int32Array | Float64Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}
