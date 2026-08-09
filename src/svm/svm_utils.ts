/**
 * SVM utility functions and kernel computations.
 * Mirrors sklearn.svm.base and related utilities.
 */

export type KernelType = "linear" | "poly" | "rbf" | "sigmoid" | "precomputed";

export interface KernelParams {
  gamma?: number | "scale" | "auto";
  coef0?: number;
  degree?: number;
}

/**
 * Compute the kernel matrix between two sets of vectors.
 */
export function computeKernel(
  X: Float64Array[],
  Y: Float64Array[],
  kernel: KernelType,
  params: KernelParams = {},
): Float64Array[] {
  const nX = X.length;
  const nY = Y.length;
  const nFeatures = X[0]?.length ?? 0;

  const gamma =
    params.gamma === "scale" ||
    params.gamma === "auto" ||
    params.gamma === undefined
      ? 1 / nFeatures
      : params.gamma;
  const coef0 = params.coef0 ?? 0;
  const degree = params.degree ?? 3;

  const K: Float64Array[] = [];

  for (let i = 0; i < nX; i++) {
    const row = new Float64Array(nY);
    for (let j = 0; j < nY; j++) {
      let val = 0;
      const xi = X[i]!;
      const yj = Y[j]!;

      switch (kernel) {
        case "linear": {
          for (let f = 0; f < nFeatures; f++)
            val += (xi[f] ?? 0) * (yj[f] ?? 0);
          break;
        }
        case "poly": {
          for (let f = 0; f < nFeatures; f++)
            val += (xi[f] ?? 0) * (yj[f] ?? 0);
          val = (gamma * val + coef0) ** degree;
          break;
        }
        case "rbf": {
          let dist = 0;
          for (let f = 0; f < nFeatures; f++)
            dist += ((xi[f] ?? 0) - (yj[f] ?? 0)) ** 2;
          val = Math.exp(-gamma * dist);
          break;
        }
        case "sigmoid": {
          for (let f = 0; f < nFeatures; f++)
            val += (xi[f] ?? 0) * (yj[f] ?? 0);
          val = Math.tanh(gamma * val + coef0);
          break;
        }
        case "precomputed": {
          // X already is the kernel matrix
          val = xi[j] ?? 0;
          break;
        }
      }
      row[j] = val;
    }
    K.push(row);
  }
  return K;
}

/**
 * Compute the gram matrix K(X, X) with the given kernel.
 */
export function gramMatrix(
  X: Float64Array[],
  kernel: KernelType,
  params: KernelParams = {},
): Float64Array[] {
  return computeKernel(X, X, kernel, params);
}

/**
 * Compute dual coefficients for a simple SVR.
 * Returns the support vectors, dual coefs, bias.
 */
export interface SVMModel {
  supportVectors: Float64Array[];
  dualCoef: Float64Array;
  intercept: number;
  kernel: KernelType;
  params: KernelParams;
}

/** Compute decision function values for a set of samples. */
export function svmDecisionFunction(
  X: Float64Array[],
  model: SVMModel,
): Float64Array {
  const K = computeKernel(X, model.supportVectors, model.kernel, model.params);
  return new Float64Array(
    K.map((row) => {
      let score = model.intercept;
      for (let j = 0; j < row.length; j++) {
        score += (row[j] ?? 0) * (model.dualCoef[j] ?? 0);
      }
      return score;
    }),
  );
}

/** Platt scaling: convert SVM scores to probabilities. */
export function plattScaling(
  scores: Float64Array,
  A: number,
  B: number,
): Float64Array {
  return new Float64Array(scores.map((s) => 1 / (1 + Math.exp(A * s + B))));
}

/** Compute Platt calibration parameters from scores and labels. */
export function fitPlattScaling(
  scores: Float64Array,
  y: Int32Array,
): { A: number; B: number } {
  const n = scores.length;
  const nPos = Array.from(y).filter((v) => v > 0).length;
  const nNeg = n - nPos;
  const tPos = (nPos + 1) / (nPos + 2);
  const tNeg = 1 / (nNeg + 2);

  let A = 0;
  let B = Math.log((nNeg + 1) / (nPos + 1));

  for (let iter = 0; iter < 100; iter++) {
    let dA = 0;
    let dB = 0;
    for (let i = 0; i < n; i++) {
      const t = y[i]! > 0 ? tPos : tNeg;
      const p = 1 / (1 + Math.exp(A * (scores[i] ?? 0) + B));
      dA += (p - t) * (scores[i] ?? 0);
      dB += p - t;
    }
    A -= (0.01 * dA) / n;
    B -= (0.01 * dB) / n;
  }
  return { A, B };
}

/** Hinge loss for SVM. */
export function hingeLoss(
  yTrue: Int32Array,
  decisionValues: Float64Array,
): number {
  let loss = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const margin = (yTrue[i] ?? 0) * (decisionValues[i] ?? 0);
    loss += Math.max(0, 1 - margin);
  }
  return loss / yTrue.length;
}

/** @deprecated Use svmDecisionFunction, plattScaling, fitPlattScaling, hingeLoss instead. */
export const SVMUtils = {
  decisionFunction: svmDecisionFunction,
  plattScaling,
  fitPlattScaling,
  hingeLoss,
};
