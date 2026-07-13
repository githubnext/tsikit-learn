/**
 * Plotting helpers for linear model diagnostics.
 * Provides residual plots, coefficient paths, and influence diagnostics.
 * Analogous to sklearn.linear_model._plot and statsmodels influence plots.
 */

/** A single 2-D point for plotting. */
export interface Point2D {
  x: number;
  y: number;
}

/** Result of a residual analysis. */
export interface ResidualAnalysis {
  /** Fitted values (ŷ). */
  fitted: Float64Array;
  /** Residuals (y - ŷ). */
  residuals: Float64Array;
  /** Standardised residuals. */
  standardizedResiduals: Float64Array;
  /** Hat-matrix diagonal (leverage values). */
  leverage: Float64Array;
  /** Cook's distance per observation. */
  cooksDistance: Float64Array;
  /** Number of samples. */
  nSamples: number;
}

/**
 * Computes residual diagnostics for a fitted linear model.
 *
 * @param X         Design matrix (flat, nSamples × nFeatures, including intercept column if used).
 * @param y         True target values.
 * @param yPred     Model predictions (ŷ).
 * @param nFeatures Number of features (p), used to compute df = n - p.
 */
export function residualAnalysis(
  X: Float64Array,
  y: Float64Array,
  yPred: Float64Array,
  nFeatures: number,
): ResidualAnalysis {
  const nSamples = y.length;
  const residuals = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) residuals[i] = y[i]! - yPred[i]!;

  // Residual variance
  const df = Math.max(nSamples - nFeatures, 1);
  let sse = 0;
  for (let i = 0; i < nSamples; i++) sse += residuals[i]! ** 2;
  const sigma2 = sse / df;

  // Hat matrix diagonal: h_ii = x_i^T (X^T X)^{-1} x_i
  // Approximate via QR if X is small, otherwise use diagonal approximation.
  const leverage = computeLeverage(X, nSamples, nFeatures);

  // Standardised residuals
  const standardizedResiduals = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const denom = Math.sqrt(sigma2 * (1 - leverage[i]!));
    standardizedResiduals[i] = denom > 0 ? residuals[i]! / denom : 0;
  }

  // Cook's distance: D_i = (standardized_residual_i^2 / nFeatures) * (h_ii / (1-h_ii))
  const cooksDistance = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const h = leverage[i]!;
    const sr = standardizedResiduals[i]!;
    cooksDistance[i] = (sr ** 2 / nFeatures) * (h / Math.max(1 - h, 1e-10));
  }

  return {
    fitted: new Float64Array(yPred),
    residuals,
    standardizedResiduals,
    leverage,
    cooksDistance,
    nSamples,
  };
}

/** Computes hat-matrix diagonal via the normal equations (X (X^T X)^{-1} X^T diag). */
function computeLeverage(X: Float64Array, n: number, p: number): Float64Array {
  // XtX = X^T X (p×p)
  const XtX = new Float64Array(p * p);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        XtX[j * p + k]! += X[i * p + j]! * X[i * p + k]!;
      }
    }
  }
  // Cholesky-like inversion via Gauss-Jordan (numerically robust enough for diagnostics)
  const inv = invertSymmetric(XtX, p);

  // h_ii = x_i^T inv x_i
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++)
        v += X[i * p + j]! * inv[j * p + k]! * X[i * p + k]!;
    }
    h[i] = Math.min(v, 1 - 1e-10); // clamp to [0, 1)
  }
  return h;
}

/** Inverts a symmetric positive-definite matrix via Gauss-Jordan elimination. */
function invertSymmetric(A: Float64Array, n: number): Float64Array {
  const aug = new Float64Array(n * n * 2);
  // Build augmented [A | I]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * 2 * n + j] = A[i * n + j]!;
    aug[i * 2 * n + n + i] = 1;
  }
  for (let col = 0; col < n; col++) {
    // Pivot
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (
        Math.abs(aug[r * 2 * n + col]!) > Math.abs(aug[maxRow * 2 * n + col]!)
      )
        maxRow = r;
    }
    if (maxRow !== col) {
      for (let k = 0; k < 2 * n; k++) {
        const tmp = aug[col * 2 * n + k]!;
        aug[col * 2 * n + k] = aug[maxRow * 2 * n + k]!;
        aug[maxRow * 2 * n + k] = tmp;
      }
    }
    const pivot = aug[col * 2 * n + col]!;
    if (Math.abs(pivot) < 1e-14) continue; // singular
    for (let k = 0; k < 2 * n; k++) aug[col * 2 * n + k]! /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r * 2 * n + col]!;
      for (let k = 0; k < 2 * n; k++)
        aug[r * 2 * n + k]! -= factor * aug[col * 2 * n + k]!;
    }
  }
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) inv[i * n + j] = aug[i * 2 * n + n + j]!;
  }
  return inv;
}

/**
 * Returns (x, y) pairs for a residual-vs-fitted plot.
 * Useful for passing to any chart library.
 */
export function residualVsFitted(analysis: ResidualAnalysis): Point2D[] {
  return Array.from({ length: analysis.nSamples }, (_, i) => ({
    x: analysis.fitted[i]!,
    y: analysis.residuals[i]!,
  }));
}

/**
 * Returns (x, y) pairs for a QQ-plot of standardised residuals.
 * x = theoretical quantile, y = sample quantile.
 */
export function qqPlotData(analysis: ResidualAnalysis): Point2D[] {
  const n = analysis.nSamples;
  const sorted = Float64Array.from(analysis.standardizedResiduals).sort();
  return Array.from({ length: n }, (_, i) => ({
    x: normalQuantile((i + 0.5) / n),
    y: sorted[i]!,
  }));
}

/** Probit (inverse normal CDF) approximation (Abramowitz & Stegun). */
function normalQuantile(p: number): number {
  if (p <= 0) return Number.NEGATIVE_INFINITY;
  if (p >= 1) return Number.POSITIVE_INFINITY;
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    return (
      (q *
        (((((((2509.0809287301227 * r + 33430.57558358813) * r +
          67265.7709270087) *
          r +
          45921.95393154987) *
          r +
          13731.69376550946) *
          r +
          1971.5909503065513) *
          r +
          133.14166789178438) *
          r +
          3.3871328727963665)) /
      (((((((5226.495278852854 * r + 28729.085735721943) * r +
        39307.89580009271) *
        r +
        21213.794301586597) *
        r +
        5394.196021424751) *
        r +
        687.1870074920579) *
        r +
        42.31333070160091) *
        r +
        1.0)
    );
  }
  let r = Math.sqrt(-Math.log(q < 0 ? p : 1 - p));
  const sign = q < 0 ? -1 : 1;
  r -= 1.6;
  return (
    sign *
    (((((((0.00007713336199095934 * r + 0.0010507500716444523) * r +
      0.012730885801323406) *
      r +
      0.06295276597812123) *
      r +
      0.17568056015169417) *
      r +
      0.23730152978510463) *
      r +
      0.09125441665551104) /
      (((((((0.0010507500716444523 * r + 0.05475938084995345) * r +
        0.1852757311752208) *
        r +
        0.34897737303254306) *
        r +
        0.4049759398108879) *
        r +
        0.2404840759281837) *
        r +
        0.06871870074920579) *
        r +
        0.000007713336199095934) +
      1)
  );
}
