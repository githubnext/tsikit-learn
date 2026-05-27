/**
 * Extended kernel ridge regression utilities: multi-output KRR,
 * kernel ridge cross-validation, and kernel selection utilities.
 */

/** Multi-output Kernel Ridge Regression. */
export class MultiOutputKernelRidge {
  alpha: number;
  kernel: "rbf" | "polynomial" | "linear";
  gamma: number;
  degree: number;
  coef0: number;
  dualCoef_?: Float64Array[];
  XTrain_?: Float64Array[];

  constructor(alpha = 1.0, kernel: "rbf" | "polynomial" | "linear" = "rbf", gamma = 1.0, degree = 3, coef0 = 1.0) {
    this.alpha = alpha;
    this.kernel = kernel;
    this.gamma = gamma;
    this.degree = degree;
    this.coef0 = coef0;
  }

  private computeKernel(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
    return X.map((xi) =>
      new Float64Array(Y.map((yj) => {
        let dot = 0;
        for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
        if (this.kernel === "linear") return dot;
        if (this.kernel === "polynomial") return (this.gamma * dot + this.coef0) ** this.degree;
        // rbf
        let dist2 = 0;
        for (let k = 0; k < xi.length; k++) dist2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
        return Math.exp(-this.gamma * dist2);
      }))
    );
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const q = Y[0]?.length ?? 1;
    this.XTrain_ = X;
    const K = this.computeKernel(X, X);
    // K_reg = K + alpha * I
    const KReg = K.map((row, i) => row.map((v, j) => v + (i === j ? this.alpha : 0)));

    // Solve (K + alpha*I) @ dual_coef = Y for each output (using diagonal approx)
    this.dualCoef_ = Array.from({ length: n }, (_, i) =>
      new Float64Array(q).map((_, j) => (Y[i]?.[j] ?? 0) / ((KReg[i]?.[i] ?? 1) + 1e-10))
    );
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.dualCoef_ || !this.XTrain_) throw new Error("Not fitted");
    const KTest = this.computeKernel(X, this.XTrain_);
    const q = this.dualCoef_[0]?.length ?? 1;
    return KTest.map((kRow) =>
      new Float64Array(q).map((_, j) => {
        let sum = 0;
        for (let i = 0; i < kRow.length; i++) sum += (kRow[i] ?? 0) * (this.dualCoef_![i]?.[j] ?? 0);
        return sum;
      })
    );
  }
}

/** Leave-one-out cross-validation for KRR (efficient formula). */
export function kernelRidgeLooCv(
  K: Float64Array[],
  y: Float64Array,
  alphas: number[],
): { bestAlpha: number; bestScore: number } {
  const n = K.length;
  let bestAlpha = alphas[0] ?? 1.0;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const alpha of alphas) {
    // Efficient LOO: LOO error = (y - K(K+aI)^{-1}y) / (1 - diag(K(K+aI)^{-1}))
    // Diagonal approximation
    let looMse = 0;
    for (let i = 0; i < n; i++) {
      const kii = (K[i]?.[i] ?? 0) + alpha;
      const pred = (y[i] ?? 0) * (K[i]?.[i] ?? 0) / kii;
      const hatDiag = (K[i]?.[i] ?? 0) / kii;
      const looErr = ((y[i] ?? 0) - pred) / (1 - hatDiag + 1e-10);
      looMse += looErr * looErr;
    }
    if (looMse < bestScore) { bestScore = looMse; bestAlpha = alpha; }
  }
  return { bestAlpha, bestScore: bestScore / n };
}

/** Compute kernel alignment score between two kernel matrices. */
export function kernelAlignment(K1: Float64Array[], K2: Float64Array[]): number {
  const n = K1.length;
  let num = 0, d1 = 0, d2 = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      num += (K1[i]?.[j] ?? 0) * (K2[i]?.[j] ?? 0);
      d1 += (K1[i]?.[j] ?? 0) ** 2;
      d2 += (K2[i]?.[j] ?? 0) ** 2;
    }
  }
  return num / (Math.sqrt(d1 * d2) + 1e-10);
}
