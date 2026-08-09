/**
 * Mutual information feature selection.
 * Mirrors sklearn.feature_selection.mutual_info_classif and mutual_info_regression.
 */

import { NotFittedError } from "../exceptions.js";

/** Estimate mutual information between X column and y using k-NN estimator (simplified). */
function mutualInfoContinuous(
  x: Float64Array,
  y: Float64Array,
  nNeighbors: number = 3,
): number {
  const n = x.length;
  // Sort by x
  const idx = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (x[a] ?? 0) - (x[b] ?? 0),
  );

  // Estimate mutual info via normalized histogram approach (simplified)
  // Using entropy difference: MI(X;Y) ~ H(X) + H(Y) - H(X,Y)
  const bins = Math.max(2, Math.floor(Math.sqrt(n)));

  function entropy1D(vals: Float64Array): number {
    const mn = Math.min(...Array.from(vals));
    const mx = Math.max(...Array.from(vals));
    const range = mx - mn;
    if (range < 1e-14) return 0;
    const counts = new Float64Array(bins);
    for (const v of vals) {
      const bi = Math.min(bins - 1, Math.floor(((v - mn) / range) * bins));
      counts[bi] = (counts[bi] ?? 0) + 1;
    }
    let h = 0;
    for (const c of counts)
      if (c > 0) {
        const p = c / n;
        h -= p * Math.log(p);
      }
    return h;
  }

  const hx = entropy1D(x);
  const hy = entropy1D(y);

  // Joint entropy (2D histogram)
  const mnX = Math.min(...Array.from(x));
  const mxX = Math.max(...Array.from(x));
  const mnY = Math.min(...Array.from(y));
  const mxY = Math.max(...Array.from(y));
  const rangeX = mxX - mnX + 1e-14;
  const rangeY = mxY - mnY + 1e-14;
  const jointCounts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const bx = Math.min(
      bins - 1,
      Math.floor((((x[i] ?? 0) - mnX) / rangeX) * bins),
    );
    const by = Math.min(
      bins - 1,
      Math.floor((((y[i] ?? 0) - mnY) / rangeY) * bins),
    );
    const key = bx * bins + by;
    jointCounts.set(key, (jointCounts.get(key) ?? 0) + 1);
  }
  let hjoint = 0;
  for (const c of jointCounts.values()) {
    const p = c / n;
    hjoint -= p * Math.log(p);
  }

  return Math.max(0, hx + hy - hjoint);
}

function mutualInfoDiscrete(x: Float64Array, labels: Int32Array): number {
  const n = x.length;
  const bins = Math.max(2, Math.floor(Math.sqrt(n)));
  const mn = Math.min(...Array.from(x));
  const mx = Math.max(...Array.from(x));
  const range = mx - mn + 1e-14;

  const classSet = new Set(Array.from(labels));
  const classes = Array.from(classSet).sort((a, b) => a - b);
  const nClasses = classes.length;
  const classToIdx = new Map(classes.map((c, i) => [c, i]));

  // Compute P(X=bin), P(Y=class), P(X=bin, Y=class)
  const pX = new Float64Array(bins);
  const pY = new Float64Array(nClasses);
  const pXY: Float64Array[] = Array.from(
    { length: bins },
    () => new Float64Array(nClasses),
  );

  for (let i = 0; i < n; i++) {
    const bx = Math.min(
      bins - 1,
      Math.floor((((x[i] ?? 0) - mn) / range) * bins),
    );
    const yi = classToIdx.get(labels[i] ?? 0) ?? 0;
    pX[bx] = (pX[bx] ?? 0) + 1;
    pY[yi] = (pY[yi] ?? 0) + 1;
    pXY[bx]![yi] = (pXY[bx]![yi] ?? 0) + 1;
  }

  let mi = 0;
  for (let bx = 0; bx < bins; bx++) {
    for (let yi = 0; yi < nClasses; yi++) {
      const joint = (pXY[bx]![yi] ?? 0) / n;
      const px = (pX[bx] ?? 0) / n;
      const py = (pY[yi] ?? 0) / n;
      if (joint > 0 && px > 0 && py > 0)
        mi += joint * Math.log(joint / (px * py));
    }
  }
  return Math.max(0, mi);
}

/**
 * Estimate mutual information between each feature and the classification target.
 * Mirrors sklearn.feature_selection.mutual_info_classif.
 */
export function mutualInfoClassif(
  X: Float64Array[],
  y: Int32Array,
  options: { nNeighbors?: number } = {},
): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const mi = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const xj = new Float64Array(X.map((xi) => xi[j] ?? 0));
    mi[j] = mutualInfoDiscrete(xj, y);
  }
  return mi;
}

/**
 * Estimate mutual information between each feature and the continuous target.
 * Mirrors sklearn.feature_selection.mutual_info_regression.
 */
export function mutualInfoRegression(
  X: Float64Array[],
  y: Float64Array,
  options: { nNeighbors?: number } = {},
): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const nNeighbors = options.nNeighbors ?? 3;
  const mi = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const xj = new Float64Array(X.map((xi) => xi[j] ?? 0));
    mi[j] = mutualInfoContinuous(xj, y, nNeighbors);
  }
  return mi;
}

export interface GenericUnivariateSelectOptions {
  scoreFunc?: (
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ) => [Float64Array, Float64Array] | Float64Array;
  mode?: "percentile" | "k_best" | "fpr" | "fdr" | "fwe";
  param?: number;
}

/**
 * Univariate feature selector with configurable strategy.
 * Mirrors sklearn.feature_selection.GenericUnivariateSelect.
 */
export class GenericUnivariateSelect {
  scoreFunc: (
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ) => [Float64Array, Float64Array] | Float64Array;
  mode: "percentile" | "k_best" | "fpr" | "fdr" | "fwe";
  param: number;

  scores_: Float64Array | null = null;
  pvalues_: Float64Array | null = null;
  selectedMask_: boolean[] | null = null;

  constructor(options: GenericUnivariateSelectOptions = {}) {
    // Default: chi2-like fallback using variance
    this.scoreFunc =
      options.scoreFunc ??
      ((X) => {
        const p = (X[0] ?? new Float64Array(0)).length;
        const scores = new Float64Array(p);
        const pvals = new Float64Array(p);
        for (let j = 0; j < p; j++) {
          let s = 0;
          let s2 = 0;
          for (const xi of X) {
            s += xi[j] ?? 0;
            s2 += (xi[j] ?? 0) ** 2;
          }
          const n = X.length;
          scores[j] = s2 / n - (s / n) ** 2;
          pvals[j] = 0.5;
        }
        return [scores, pvals] as [Float64Array, Float64Array];
      });
    this.mode = options.mode ?? "percentile";
    this.param = options.param ?? 10;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const result = this.scoreFunc(X, y);
    if (Array.isArray(result) && result.length === 2) {
      this.scores_ = result[0] as Float64Array;
      this.pvalues_ = result[1] as Float64Array;
    } else {
      this.scores_ = result as Float64Array;
      this.pvalues_ = new Float64Array((result as Float64Array).length).fill(
        0.5,
      );
    }

    const p = this.scores_.length;
    const scores = this.scores_;

    if (this.mode === "k_best") {
      const k = Math.min(Math.floor(this.param), p);
      const sortedIdx = Array.from({ length: p }, (_, i) => i).sort(
        (a, b) => (scores[b] ?? 0) - (scores[a] ?? 0),
      );
      const topK = new Set(sortedIdx.slice(0, k));
      this.selectedMask_ = Array.from({ length: p }, (_, i) => topK.has(i));
    } else {
      // percentile
      const pct = Math.min(100, Math.max(0, this.param));
      const sortedScores = Array.from(scores).sort((a, b) => b - a);
      const threshold =
        sortedScores[Math.floor((1 - pct / 100) * sortedScores.length)] ?? 0;
      this.selectedMask_ = Array.from(
        { length: p },
        (_, i) => (scores[i] ?? 0) >= threshold,
      );
    }

    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedMask_)
      throw new NotFittedError("GenericUnivariateSelect is not fitted yet.");
    const selIdx = this.selectedMask_
      .map((v, i) => (v ? i : -1))
      .filter((i) => i !== -1);
    return X.map((xi) => new Float64Array(selIdx.map((j) => xi[j] ?? 0)));
  }

  fitTransform(
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  getSupport(): boolean[] {
    if (!this.selectedMask_)
      throw new NotFittedError("GenericUnivariateSelect is not fitted yet.");
    return [...this.selectedMask_];
  }
}
