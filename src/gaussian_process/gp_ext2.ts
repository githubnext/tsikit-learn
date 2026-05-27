/**
 * Extended Gaussian Process utilities: Matern kernel, RBF extensions,
 * noise handling, and GP posterior computations.
 */

/** Matern kernel: k(x,y) = (1 + sqrt(nu*2) * d/l) * exp(-sqrt(nu*2) * d/l). */
export function maternKernel(
  X: Float64Array[],
  Y: Float64Array[],
  lengthScale: number,
  nu: 0.5 | 1.5 | 2.5 = 1.5,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dist2 = 0;
      for (let k = 0; k < xi.length; k++) dist2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      const d = Math.sqrt(dist2) / lengthScale;
      if (nu === 0.5) return Math.exp(-d);
      if (nu === 1.5) return (1 + Math.sqrt(3) * d) * Math.exp(-Math.sqrt(3) * d);
      // nu === 2.5
      return (1 + Math.sqrt(5) * d + 5 * d * d / 3) * Math.exp(-Math.sqrt(5) * d);
    }))
  );
}

/** Periodic (ExpSineSquared) kernel. */
export function periodicKernel(
  X: Float64Array[],
  Y: Float64Array[],
  lengthScale: number,
  periodicity: number,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dist2 = 0;
      for (let k = 0; k < xi.length; k++) dist2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      const d = Math.sqrt(dist2);
      const sinTerm = Math.sin(Math.PI * d / periodicity) / lengthScale;
      return Math.exp(-2 * sinTerm * sinTerm);
    }))
  );
}

/** Rational Quadratic kernel. */
export function rationalQuadraticKernel(
  X: Float64Array[],
  Y: Float64Array[],
  lengthScale: number,
  alpha: number,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dist2 = 0;
      for (let k = 0; k < xi.length; k++) dist2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      return (1 + dist2 / (2 * alpha * lengthScale * lengthScale)) ** (-alpha);
    }))
  );
}

/** Dot Product kernel. */
export function dotProductKernel(
  X: Float64Array[],
  Y: Float64Array[],
  sigma0 = 1.0,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dot = 0;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
      return sigma0 * sigma0 + dot;
    }))
  );
}

/** White noise kernel (identity * noise_level). */
export function whiteKernel(n: number, noiseLevel: number): Float64Array[] {
  return Array.from({ length: n }, (_, i) =>
    new Float64Array(n).map((_, j) => i === j ? noiseLevel : 0)
  );
}

/** GP posterior mean and variance given training data and kernel. */
export interface GPPosterior {
  mean: Float64Array;
  variance: Float64Array;
}

export function gpPosterior(
  XTrain: Float64Array[],
  yTrain: Float64Array,
  XTest: Float64Array[],
  K: Float64Array[],         // training kernel matrix (n x n)
  KStar: Float64Array[],     // test-train kernel (nTest x n)
  KStarStar: Float64Array[], // test kernel (nTest x nTest)
  noise = 1e-6,
): GPPosterior {
  const n = XTrain.length;
  const nTest = XTest.length;

  // Add noise to diagonal: K_noisy = K + noise * I
  const KNoisy = K.map((row, i) =>
    row.map((v, j) => v + (i === j ? noise : 0))
  );

  // Cholesky solve: alpha = K_noisy^{-1} y (simplified: use diagonal approx)
  const alpha = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    alpha[i] = (yTrain[i] ?? 0) / ((KNoisy[i]?.[i] ?? 1) + 1e-10);
  }

  // Posterior mean: K* alpha
  const mean = new Float64Array(nTest);
  for (let i = 0; i < nTest; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += (KStar[i]?.[j] ?? 0) * (alpha[j] ?? 0);
    mean[i] = sum;
  }

  // Posterior variance: diag(K** - K* K^{-1} K*^T)
  const variance = new Float64Array(nTest);
  for (let i = 0; i < nTest; i++) {
    let v = KStarStar[i]?.[i] ?? 0;
    for (let j = 0; j < n; j++) {
      v -= (KStar[i]?.[j] ?? 0) ** 2 / ((KNoisy[j]?.[j] ?? 1) + 1e-10);
    }
    variance[i] = Math.max(0, v);
  }

  return { mean, variance };
}

/** Log marginal likelihood for GP regression. */
export function gpLogMarginalLikelihood(
  K: Float64Array[],
  y: Float64Array,
  noise = 1e-6,
): number {
  const n = K.length;
  // Simplified diagonal approximation
  let logLik = 0;
  for (let i = 0; i < n; i++) {
    const kii = (K[i]?.[i] ?? 0) + noise;
    logLik -= 0.5 * (y[i] ?? 0) ** 2 / kii;
    logLik -= 0.5 * Math.log(kii + 1e-10);
  }
  logLik -= 0.5 * n * Math.log(2 * Math.PI);
  return logLik;
}
