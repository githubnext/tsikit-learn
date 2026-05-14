/**
 * Inspection utilities: permutation_importance and partial_dependence.
 * Mirrors sklearn.inspection.
 */

import { NotFittedError } from "../exceptions.js";

// ─── PermutationImportance ─────────────────────────────────────────────────────

export interface PredictorWithScore {
  predict(X: Float64Array[]): Int32Array | Float64Array;
  score?(X: Float64Array[], y: Int32Array | Float64Array): number;
}

export interface PermutationImportanceResult {
  importances: Float64Array[];
  importancesMean: Float64Array;
  importancesStd: Float64Array;
}

function accuracyScore(preds: Int32Array | Float64Array, y: Int32Array | Float64Array): number {
  let correct = 0;
  for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
  return correct / y.length;
}

function r2Score(preds: Float64Array, y: Float64Array): number {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    ssRes += ((y[i] ?? 0) - (preds[i] ?? 0)) ** 2;
    ssTot += ((y[i] ?? 0) - mean) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

export function permutationImportance(
  estimator: PredictorWithScore,
  X: Float64Array[],
  y: Int32Array | Float64Array,
  opts: {
    nRepeats?: number;
    randomState?: number;
    scoring?: "accuracy" | "r2";
  } = {},
): PermutationImportanceResult {
  const nRepeats = opts.nRepeats ?? 5;
  const seedInit = opts.randomState ?? 42;
  const n = X.length;
  const d = X[0]?.length ?? 0;

  const basePreds = estimator.predict(X);
  const isClassification = basePreds instanceof Int32Array;
  const baseScore = isClassification
    ? accuracyScore(basePreds, y)
    : r2Score(basePreds as Float64Array, y as Float64Array);

  const importances: Float64Array[] = Array.from({ length: d }, () => new Float64Array(nRepeats));

  let rngSeed = seedInit;
  const rand = () => {
    rngSeed = (rngSeed * 1664525 + 1013904223) & 0xffffffff;
    return (rngSeed >>> 0) / 0xffffffff;
  };

  for (let f = 0; f < d; f++) {
    for (let r = 0; r < nRepeats; r++) {
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = indices[i]!;
        indices[i]! = indices[j]!;
        indices[j]! = tmp;
      }

      const Xperm: Float64Array[] = X.map((xi, i) => {
        const row = Float64Array.from(xi);
        row[f]! = (X[indices[i] ?? 0] as Float64Array)[f] ?? 0;
        return row;
      });

      const permPreds = estimator.predict(Xperm);
      const permScore = isClassification
        ? accuracyScore(permPreds, y)
        : r2Score(permPreds as Float64Array, y as Float64Array);

      (importances[f] as Float64Array)[r]! = baseScore - permScore;
    }
  }

  const importancesMean = Float64Array.from(importances, (imp) => {
    const arr = imp as Float64Array;
    return arr.reduce((s, v) => s + v, 0) / nRepeats;
  });

  const importancesStd = Float64Array.from(importances, (imp, f) => {
    const arr = imp as Float64Array;
    const mean = importancesMean[f] ?? 0;
    return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / nRepeats);
  });

  return { importances, importancesMean, importancesStd };
}

// ─── PartialDependence ────────────────────────────────────────────────────────

export interface PartialDependenceResult {
  average: Float64Array[];
  gridValues: Float64Array[];
}

export function partialDependence(
  estimator: { predict(X: Float64Array[]): Int32Array | Float64Array },
  X: Float64Array[],
  features: number[],
  opts: {
    gridResolution?: number;
  } = {},
): PartialDependenceResult {
  const gridResolution = opts.gridResolution ?? 100;
  const n = X.length;

  const gridValues: Float64Array[] = features.map((f) => {
    const vals = X.map((xi) => xi[f] ?? 0).sort((a, b) => a - b);
    const unique = [...new Set(vals)];
    if (unique.length <= gridResolution) return Float64Array.from(unique);
    const step = (unique.length - 1) / (gridResolution - 1);
    return Float64Array.from({ length: gridResolution }, (_, i) => unique[Math.round(i * step)] ?? 0);
  });

  const average: Float64Array[] = features.map((f, fi) => {
    const grid = gridValues[fi] as Float64Array;
    return Float64Array.from(grid, (gridVal) => {
      const Xmod: Float64Array[] = X.map((xi) => {
        const row = Float64Array.from(xi);
        row[f]! = gridVal;
        return row;
      });
      const preds = estimator.predict(Xmod);
      let sum = 0;
      for (let i = 0; i < n; i++) sum += preds[i] ?? 0;
      return sum / n;
    });
  });

  return { average, gridValues };
}
