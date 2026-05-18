/**
 * DecisionBoundaryDisplay: visualization of classifier decision boundaries.
 * Mirrors sklearn.inspection.DecisionBoundaryDisplay.
 */

import { NotFittedError } from "../exceptions.js";

export interface DecisionBoundaryDisplayOptions {
  /** Number of grid points along each axis. */
  nGridPoints?: number;
  /** Padding fraction to extend grid beyond data range. */
  eps?: number;
  /** Response method: 'predict', 'predict_proba', or 'decision_function'. */
  responseMethod?: "predict" | "predict_proba" | "decision_function";
  /** Feature indices to plot (default: [0, 1]). */
  featureIdx?: [number, number];
}

export type GridClassifier = {
  predict(X: Float64Array[]): Int32Array;
  predictProba?: (X: Float64Array[]) => Float64Array[];
  decisionFunction?: (X: Float64Array[]) => Float64Array;
};

export interface DecisionBoundaryResult {
  /** Grid x-axis values (xx0). */
  xx0: Float64Array;
  /** Grid y-axis values (xx1). */
  xx1: Float64Array;
  /** Response values on grid (nGridPoints x nGridPoints). */
  response: Float64Array[];
  /** Number of grid points per axis. */
  nGridPoints: number;
}

/**
 * Visualize the decision boundary of a classifier over a 2D feature grid.
 * Mirrors sklearn.inspection.DecisionBoundaryDisplay.
 */
export class DecisionBoundaryDisplay {
  xx0: Float64Array;
  xx1: Float64Array;
  response: Float64Array[];
  nGridPoints: number;

  constructor(result: DecisionBoundaryResult) {
    this.xx0 = result.xx0;
    this.xx1 = result.xx1;
    this.response = result.response;
    this.nGridPoints = result.nGridPoints;
  }

  /**
   * Create a DecisionBoundaryDisplay from an estimator and training data.
   */
  static fromEstimator(
    estimator: GridClassifier,
    X: Float64Array[],
    options: DecisionBoundaryDisplayOptions = {},
  ): DecisionBoundaryDisplay {
    const n = options.nGridPoints ?? 50;
    const eps = options.eps ?? 0.05;
    const featureIdx = options.featureIdx ?? [0, 1];
    const responseMethod = options.responseMethod ?? "predict";

    if (X.length === 0) throw new Error("X must not be empty");

    const f0 = featureIdx[0]!;
    const f1 = featureIdx[1]!;

    let x0Min = Number.POSITIVE_INFINITY;
    let x0Max = Number.NEGATIVE_INFINITY;
    let x1Min = Number.POSITIVE_INFINITY;
    let x1Max = Number.NEGATIVE_INFINITY;

    for (const xi of X) {
      const v0 = xi[f0] ?? 0;
      const v1 = xi[f1] ?? 0;
      if (v0 < x0Min) x0Min = v0;
      if (v0 > x0Max) x0Max = v0;
      if (v1 < x1Min) x1Min = v1;
      if (v1 > x1Max) x1Max = v1;
    }

    const r0 = x0Max - x0Min;
    const r1 = x1Max - x1Min;
    x0Min -= eps * r0;
    x0Max += eps * r0;
    x1Min -= eps * r1;
    x1Max += eps * r1;

    const xx0 = new Float64Array(n).map(
      (_, i) => x0Min + (i / (n - 1)) * (x0Max - x0Min),
    );
    const xx1 = new Float64Array(n).map(
      (_, i) => x1Min + (i / (n - 1)) * (x1Max - x1Min),
    );

    // Build grid
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    const gridPoints: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const pt = new Float64Array(nFeatures);
        pt[f0] = xx0[j] ?? 0;
        pt[f1] = xx1[i] ?? 0;
        gridPoints.push(pt);
      }
    }

    let flatResponse: number[];
    if (responseMethod === "predict_proba" && estimator.predictProba) {
      const proba = estimator.predictProba(gridPoints);
      flatResponse = proba.map(p => p[1] ?? 0);
    } else if (responseMethod === "decision_function" && estimator.decisionFunction) {
      const df = estimator.decisionFunction(gridPoints);
      flatResponse = Array.from(df);
    } else {
      const pred = estimator.predict(gridPoints);
      flatResponse = Array.from(pred);
    }

    // Reshape to n x n
    const response: Float64Array[] = Array.from({ length: n }, (_, i) =>
      new Float64Array(flatResponse.slice(i * n, (i + 1) * n)),
    );

    return new DecisionBoundaryDisplay({ xx0, xx1, response, nGridPoints: n });
  }
}
