/**
 * Classification metrics.
 * Mirrors sklearn.metrics (classification subset).
 */

import { ValueError } from "../exceptions.js";

/** Accuracy score. */
export function accuracy_score(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  normalize = true,
): number {
  if (yTrue.length !== yPred.length) {
    throw new ValueError("yTrue and yPred must have the same length");
  }
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) === (yPred[i] ?? 0)) correct++;
  }
  return normalize ? (yTrue.length > 0 ? correct / yTrue.length : 0) : correct;
}

/** Confusion matrix. Returns a 2D array [actual][predicted]. */
export function confusion_matrix(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  labels?: Int32Array,
): number[][] {
  const labelSet =
    labels ??
    (() => {
      const s = new Set<number>();
      for (const v of yTrue) s.add(v);
      for (const v of yPred) s.add(v);
      return new Int32Array([...s].sort((a, b) => a - b));
    })();

  const n = labelSet.length;
  const labelIdx = new Map<number, number>();
  for (let i = 0; i < n; i++) labelIdx.set(labelSet[i] ?? 0, i);

  const matrix: number[][] = Array.from({ length: n }, () =>
    new Array<number>(n).fill(0),
  );
  for (let i = 0; i < yTrue.length; i++) {
    const ti = labelIdx.get(yTrue[i] ?? 0);
    const pi = labelIdx.get(yPred[i] ?? 0);
    if (ti !== undefined && pi !== undefined) {
      (matrix[ti] as number[])[pi] = ((matrix[ti] as number[])[pi] ?? 0) + 1;
    }
  }
  return matrix;
}

/** Precision score for binary or multiclass (macro average). */
export function precision_score(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  options: { average?: "binary" | "macro" | "micro"; posLabel?: number } = {},
): number {
  const { average = "binary", posLabel = 1 } = options;
  const classes = (() => {
    const s = new Set<number>();
    for (const v of yTrue) s.add(v);
    return new Int32Array([...s].sort((a, b) => a - b));
  })();

  if (average === "binary") {
    let tp = 0;
    let fp = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yPred[i] ?? 0) === posLabel) {
        if ((yTrue[i] ?? 0) === posLabel) tp++;
        else fp++;
      }
    }
    return tp + fp === 0 ? 0 : tp / (tp + fp);
  }

  if (average === "macro") {
    let total = 0;
    for (const c of classes) {
      let tp = 0;
      let fp = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if ((yPred[i] ?? 0) === c) {
          if ((yTrue[i] ?? 0) === c) tp++;
          else fp++;
        }
      }
      total += tp + fp === 0 ? 0 : tp / (tp + fp);
    }
    return classes.length > 0 ? total / classes.length : 0;
  }

  // micro
  let tp = 0;
  let fp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yPred[i] ?? 0) === (yTrue[i] ?? 0)) tp++;
    else fp++;
  }
  return tp + fp === 0 ? 0 : tp / (tp + fp);
}

/** Recall score. */
export function recall_score(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  options: { average?: "binary" | "macro" | "micro"; posLabel?: number } = {},
): number {
  const { average = "binary", posLabel = 1 } = options;
  const classes = (() => {
    const s = new Set<number>();
    for (const v of yTrue) s.add(v);
    return new Int32Array([...s].sort((a, b) => a - b));
  })();

  if (average === "binary") {
    let tp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === posLabel) {
        if ((yPred[i] ?? 0) === posLabel) tp++;
        else fn++;
      }
    }
    return tp + fn === 0 ? 0 : tp / (tp + fn);
  }

  if (average === "macro") {
    let total = 0;
    for (const c of classes) {
      let tp = 0;
      let fn = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if ((yTrue[i] ?? 0) === c) {
          if ((yPred[i] ?? 0) === c) tp++;
          else fn++;
        }
      }
      total += tp + fn === 0 ? 0 : tp / (tp + fn);
    }
    return classes.length > 0 ? total / classes.length : 0;
  }

  return accuracy_score(yTrue, yPred);
}

/** F1 score. */
export function f1_score(
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  options: { average?: "binary" | "macro" | "micro"; posLabel?: number } = {},
): number {
  const p = precision_score(yTrue, yPred, options);
  const r = recall_score(yTrue, yPred, options);
  return p + r === 0 ? 0 : (2 * p * r) / (p + r);
}

/** Log loss (cross-entropy). */
export function log_loss(
  yTrue: Float64Array | Int32Array,
  yProba: Float64Array[],
  eps = 1e-15,
): number {
  let total = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const row = yProba[i] ?? new Float64Array(0);
    const label = yTrue[i] ?? 0;
    // For binary: row[1] is P(class=1)
    const p = Math.min(1 - eps, Math.max(eps, row[label] ?? eps));
    total += -Math.log(p);
  }
  return yTrue.length > 0 ? total / yTrue.length : 0;
}
