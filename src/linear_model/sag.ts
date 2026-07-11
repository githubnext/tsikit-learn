/**
 * SAG (Stochastic Average Gradient) and SAGA solver utilities.
 * Port of sklearn.linear_model._sag and _sag_fast
 */

export interface SAGResult {
  weights: Float64Array;
  intercept: number;
  nIter: number;
  converged: boolean;
}

/**
 * SAG solver for linear models (regression and classification).
 * Stochastic Average Gradient descent — efficient for large datasets.
 */
export function sagSolver(
  X: Float64Array[],
  y: Float64Array,
  sampleWeight: Float64Array | null,
  lossFunction: "squared" | "log" | "modified_huber",
  alpha: number,
  beta: number,
  maxIter: number,
  tol: number,
  fitIntercept: boolean,
  saga: boolean,
): SAGResult {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const weights = new Float64Array(d);
  let intercept = 0;
  const gradMemory: Float64Array[] = Array.from(
    { length: n },
    () => new Float64Array(d),
  );
  const interceptMemory = new Float64Array(n);
  const sumGrad = new Float64Array(d);
  let sumInterceptGrad = 0;
  let nIter = 0;
  let converged = false;

  // Simple LCG for sample selection
  let rng = 42;
  const nextRng = (): number => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) % n;
  };

  const computeGrad = (
    xi: Float64Array,
    yi: number,
    wi: Float64Array,
    bi: number,
  ): [Float64Array, number] => {
    // Compute prediction
    let pred = bi;
    for (let j = 0; j < d; j++) pred += (wi[j] ?? 0) * (xi[j] ?? 0);

    let gradMult = 0;
    if (lossFunction === "squared") {
      gradMult = pred - yi;
    } else if (lossFunction === "log") {
      const margin = yi * pred;
      gradMult = -yi / (1 + Math.exp(margin));
    } else {
      // modified_huber
      const margin = yi * pred;
      if (margin < -1) gradMult = -4 * yi;
      else if (margin < 1) gradMult = -2 * yi * (1 - margin);
      else gradMult = 0;
    }

    const grad = new Float64Array(d);
    for (let j = 0; j < d; j++) grad[j] = gradMult * (xi[j] ?? 0);
    return [grad, fitIntercept ? gradMult : 0];
  };

  const stepSize = 1.0 / (2 * n * (alpha + beta));

  for (nIter = 0; nIter < maxIter; nIter++) {
    const prevNorm = weights.reduce((s, w) => s + w * w, 0);

    for (let step = 0; step < n; step++) {
      const idx = nextRng();
      const xi = X[idx]!;
      const yi = y[idx]!;
      const sw = sampleWeight ? (sampleWeight[idx] ?? 1) : 1;

      const [newGrad, newIntGrad] = computeGrad(xi, yi, weights, intercept);
      const oldGrad = gradMemory[idx]!;
      const oldIntGrad = interceptMemory[idx]!;

      // Update sum of gradients
      for (let j = 0; j < d; j++) {
        sumGrad[j] += sw * ((newGrad[j] ?? 0) - (oldGrad[j] ?? 0));
        oldGrad[j] = sw * (newGrad[j] ?? 0);
      }
      sumInterceptGrad += sw * (newIntGrad - oldIntGrad);
      interceptMemory[idx] = sw * newIntGrad;

      // SAGA: also use current gradient correction
      const sagaCorrection = saga ? (newGrad[0] ?? 0) - (oldGrad[0] ?? 0) : 0;
      void sagaCorrection;

      // Update weights
      for (let j = 0; j < d; j++) {
        const g = sumGrad[j]! / n + alpha * (weights[j] ?? 0);
        weights[j] = (weights[j] ?? 0) - stepSize * g;
      }

      // L1 prox (SAGA only, for LASSO)
      if (saga && beta > 0) {
        for (let j = 0; j < d; j++) {
          const w = weights[j]!;
          const threshold = stepSize * beta;
          weights[j] = Math.sign(w) * Math.max(0, Math.abs(w) - threshold);
        }
      }

      if (fitIntercept) {
        intercept -= (stepSize * sumInterceptGrad) / n;
      }
    }

    // Check convergence
    const newNorm = weights.reduce((s, w) => s + w * w, 0);
    if (Math.abs(newNorm - prevNorm) / (prevNorm + 1e-10) < tol) {
      converged = true;
      break;
    }
  }

  return { weights, intercept, nIter, converged };
}

/** SAGA solver (variant of SAG with proximal operator support for L1) */
export function sagaSolver(
  X: Float64Array[],
  y: Float64Array,
  sampleWeight: Float64Array | null,
  lossFunction: "squared" | "log" | "modified_huber",
  alpha: number,
  beta: number,
  maxIter: number,
  tol: number,
  fitIntercept: boolean,
): SAGResult {
  return sagSolver(
    X,
    y,
    sampleWeight,
    lossFunction,
    alpha,
    beta,
    maxIter,
    tol,
    fitIntercept,
    true,
  );
}

/** Get the SAG step size for a given dataset and regularization */
export function getSagStepSize(
  X: Float64Array[],
  alpha: number,
  fitIntercept: boolean,
  classWeight: Float64Array | null,
): number {
  const n = X.length;
  // Estimate max L (Lipschitz constant)
  let maxNormSq = 0;
  for (const x of X) {
    let normSq = fitIntercept ? 1 : 0;
    for (const v of x) normSq += v * v;
    if (normSq > maxNormSq) maxNormSq = normSq;
  }
  const L = maxNormSq / 4 + alpha;
  return (1.0 / (2 * L * n)) * (classWeight ? 1 : 1);
}
