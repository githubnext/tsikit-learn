/**
 * Extended SVM: SVRExt (epsilon-SVR), LinearSVRExt (dual coordinate ascent)
 */

function rbfKernel(x1: Float64Array, x2: Float64Array, gamma: number): number {
  let dist2 = 0;
  for (let i = 0; i < x1.length; i++) dist2 += ((x1[i] ?? 0) - (x2[i] ?? 0)) ** 2;
  return Math.exp(-gamma * dist2);
}

function dotProduct(x1: Float64Array, x2: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x1.length; i++) s += (x1[i] ?? 0) * (x2[i] ?? 0);
  return s;
}

export class SVRExt {
  private C: number;
  private epsilon: number;
  private kernel: "rbf" | "linear" | "poly";
  private gamma: number | "scale" | "auto";
  private degree: number;
  private maxIter: number;
  private tol: number;
  supportVectors_: Float64Array[] | null = null;
  dualCoef_: Float64Array | null = null;
  intercept_: number = 0;
  private nSupport_: number = 0;

  constructor(
    C = 1.0,
    epsilon = 0.1,
    kernel: "rbf" | "linear" | "poly" = "rbf",
    gamma: number | "scale" | "auto" = "scale",
    degree = 3,
    maxIter = 1000,
    tol = 1e-3
  ) {
    this.C = C;
    this.epsilon = epsilon;
    this.kernel = kernel;
    this.gamma = gamma;
    this.degree = degree;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  private computeKernel(x1: Float64Array, x2: Float64Array, g: number): number {
    if (this.kernel === "linear") return dotProduct(x1, x2);
    if (this.kernel === "poly") return (dotProduct(x1, x2) + 1) ** this.degree;
    return rbfKernel(x1, x2, g);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const gammaVal = typeof this.gamma === "number"
      ? this.gamma
      : this.gamma === "scale" ? 1 / (d * (X.reduce((acc, row) => {
          const v = row.reduce((a, b) => a + b * b, 0) / d;
          return acc + v;
        }, 0) / n || 1)) : 1 / d;

    // SMO-like algorithm for SVR
    const alpha = new Float64Array(n);  // alpha_i^+
    const alphaS = new Float64Array(n); // alpha_i^-
    let b = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxViolation = 0;
      for (let i = 0; i < n; i++) {
        let f = b;
        for (let j = 0; j < n; j++) {
          f += ((alpha[j] ?? 0) - (alphaS[j] ?? 0)) * this.computeKernel(X[j]!, X[i]!, gammaVal);
        }
        const yi = y[i] ?? 0;
        const residual = f - yi;
        const violation = Math.max(
          Math.abs(residual + this.epsilon - (alpha[i] ?? 0) / this.C),
          Math.abs(residual - this.epsilon + (alphaS[i] ?? 0) / this.C)
        );
        maxViolation = Math.max(maxViolation, violation);

        // Update alpha[i] and alphaS[i]
        const oldAlpha = alpha[i] ?? 0;
        const oldAlphaS = alphaS[i] ?? 0;
        const Kii = this.computeKernel(X[i]!, X[i]!, gammaVal) + 1e-6;

        const newAlpha = Math.max(0, Math.min(this.C, (alpha[i] ?? 0) - (residual + this.epsilon) / Kii));
        const newAlphaS = Math.max(0, Math.min(this.C, (alphaS[i] ?? 0) + (residual - this.epsilon) / Kii));
        alpha[i] = newAlpha;
        alphaS[i] = newAlphaS;
        b -= (newAlpha - oldAlpha - newAlphaS + oldAlphaS) * Kii * 0.5;
      }
      if (maxViolation < this.tol) break;
    }

    // Extract support vectors
    const svIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (Math.abs((alpha[i] ?? 0) - (alphaS[i] ?? 0)) > 1e-5) svIdx.push(i);
    }
    this.supportVectors_ = svIdx.map((i) => X[i]!);
    this.dualCoef_ = new Float64Array(svIdx.map((i) => (alpha[i] ?? 0) - (alphaS[i] ?? 0)));
    this.intercept_ = b;
    this.nSupport_ = svIdx.length;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.supportVectors_ || !this.dualCoef_) throw new Error("Not fitted");
    const d = this.supportVectors_[0]?.length ?? 0;
    const gammaVal = typeof this.gamma === "number" ? this.gamma : 1 / d;
    return new Float64Array(X.map((xi) => {
      let pred = this.intercept_;
      for (let j = 0; j < this.nSupport_; j++) {
        pred += (this.dualCoef_![j] ?? 0) * this.computeKernel(this.supportVectors_![j]!, xi, gammaVal);
      }
      return pred;
    }));
  }
}

export class LinearSVRExt {
  private C: number;
  private epsilon: number;
  private maxIter: number;
  private tol: number;
  private fitIntercept: boolean;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  nIter_: number = 0;

  constructor(C = 1.0, epsilon = 0.0, maxIter = 1000, tol = 1e-4, fitIntercept = true) {
    this.C = C;
    this.epsilon = epsilon;
    this.maxIter = maxIter;
    this.tol = tol;
    this.fitIntercept = fitIntercept;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const coef = new Float64Array(d);
    let intercept = 0;
    const alpha = new Float64Array(n);
    const alphaS = new Float64Array(n);

    // Dual coordinate ascent
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let i = 0; i < n; i++) {
        // Compute prediction
        let pred = intercept;
        for (let j = 0; j < d; j++) pred += (coef[j] ?? 0) * (X[i]![j] ?? 0);
        const yi = y[i] ?? 0;
        const Kii = X[i]!.reduce((acc, v) => acc + v * v, 0) + 1;

        // Update alpha+
        const dAlpha = Math.max(-alpha[i]!, Math.min(
          this.C - (alpha[i] ?? 0),
          (yi - pred - this.epsilon) / Kii
        ));
        alpha[i] = (alpha[i] ?? 0) + dAlpha;
        maxChange = Math.max(maxChange, Math.abs(dAlpha));
        for (let j = 0; j < d; j++) coef[j] = (coef[j] ?? 0) + dAlpha * (X[i]![j] ?? 0);
        if (this.fitIntercept) intercept += dAlpha;

        // Update alpha-
        const dAlphaS = Math.max(-alphaS[i]!, Math.min(
          this.C - (alphaS[i] ?? 0),
          (pred - yi - this.epsilon) / Kii
        ));
        alphaS[i] = (alphaS[i] ?? 0) + dAlphaS;
        maxChange = Math.max(maxChange, Math.abs(dAlphaS));
        for (let j = 0; j < d; j++) coef[j] = (coef[j] ?? 0) - dAlphaS * (X[i]![j] ?? 0);
        if (this.fitIntercept) intercept -= dAlphaS;
      }
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const coef = this.coef_;
    const intercept = this.intercept_;
    return new Float64Array(X.map((row) => {
      let s = intercept;
      for (let j = 0; j < coef.length; j++) s += (row[j] ?? 0) * (coef[j] ?? 0);
      return s;
    }));
  }
}
