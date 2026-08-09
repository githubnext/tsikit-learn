/**
 * Gaussian linear model utilities and Bayesian posterior helpers.
 * Mirrors scikit-learn's linear_model._bayes internals.
 */

/**
 * Compute the posterior mean and covariance for a Bayesian linear regression.
 * Prior: w ~ N(0, 1/alpha * I), Likelihood: y ~ N(Xw, 1/lambda * I)
 */
export function bayesianPosterior(
  X: Float64Array[],
  y: Float64Array,
  alpha: number,
  lambda: number,
): { mean: Float64Array; covariance: Float64Array[] } {
  const n = X.length;
  const p = X[0]?.length ?? 0;

  // S_N^{-1} = alpha * I + lambda * X^T X
  const sinvDiag = new Float64Array(p).fill(alpha);
  const Sigma: Float64Array[] = Array.from(
    { length: p },
    () => new Float64Array(p),
  );

  // X^T X
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += (X[k]?.[i] ?? 0) * (X[k]?.[j] ?? 0);
      Sigma[i]![j] = (i === j ? alpha : 0) + lambda * s;
    }
  }

  // Invert using Gauss-Jordan (small p assumed)
  const inv = invertMatrix(Sigma, p);

  // m_N = lambda * S_N * X^T y
  const Xty = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += (X[k]?.[j] ?? 0) * (y[k] ?? 0);
    Xty[j] = lambda * s;
  }

  const mean = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = 0;
    for (let j = 0; j < p; j++) s += (inv[i]?.[j] ?? 0) * (Xty[j] ?? 0);
    mean[i] = s;
  }

  void sinvDiag;
  return { mean, covariance: inv };
}

function invertMatrix(M: Float64Array[], n: number): Float64Array[] {
  const aug: Float64Array[] = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(2 * n);
    for (let j = 0; j < n; j++) row[j] = M[i]?.[j] ?? 0;
    row[n + i] = 1;
    return row;
  });

  for (let col = 0; col < n; col++) {
    let pivot = col;
    let maxV = Math.abs(aug[col]?.[col] ?? 0);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row]?.[col] ?? 0);
      if (v > maxV) {
        maxV = v;
        pivot = row;
      }
    }
    const tmp = aug[col]!;
    aug[col] = aug[pivot]!;
    aug[pivot] = tmp;

    const pivotVal = aug[col]![col] ?? 1;
    for (let j = 0; j < 2 * n; j++) {
      aug[col]![j] = (aug[col]![j] ?? 0) / pivotVal;
    }
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row]![col] ?? 0;
      for (let j = 0; j < 2 * n; j++) {
        aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
      }
    }
  }

  return aug.map((row) => row.slice(n) as Float64Array);
}

/**
 * Compute log marginal likelihood for Bayesian linear regression.
 */
export function logMarginalLikelihood(
  X: Float64Array[],
  y: Float64Array,
  alpha: number,
  lambda: number,
): number {
  const { mean: mN, covariance: SN } = bayesianPosterior(X, y, alpha, lambda);
  const n = X.length;
  const p = mN.length;

  // -0.5 * (lambda * ||y - X*mN||^2 + alpha * ||mN||^2) + 0.5 * log|lambda*SN|
  let residNorm = 0;
  for (let i = 0; i < n; i++) {
    let pred = 0;
    for (let j = 0; j < p; j++) pred += (X[i]?.[j] ?? 0) * (mN[j] ?? 0);
    residNorm += ((y[i] ?? 0) - pred) ** 2;
  }
  let priorNorm = 0;
  for (let j = 0; j < p; j++) priorNorm += (mN[j] ?? 0) ** 2;

  let logDet = 0;
  for (let i = 0; i < p; i++)
    logDet += Math.log(Math.abs(SN[i]?.[i] ?? 1) + 1e-10);

  return (
    -0.5 * lambda * residNorm -
    0.5 * alpha * priorNorm +
    0.5 * logDet +
    0.5 * p * Math.log(lambda) +
    0.5 * n * Math.log(alpha / (2 * Math.PI))
  );
}
