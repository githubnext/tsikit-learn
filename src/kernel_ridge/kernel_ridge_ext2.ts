/**
 * Extended kernel ridge regression utilities.
 * Port of sklearn.kernel_ridge extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Multi-output kernel ridge regression.
 */
export class MultiOutputKernelRidge {
  private alpha: number;
  private kernel: "rbf" | "linear" | "poly";
  private gamma: number;
  private degree: number;
  private dualCoef_: Float64Array[] = [];
  private XTrain_: Float64Array[] = [];
  private fitted = false;

  constructor(options: {
    alpha?: number;
    kernel?: "rbf" | "linear" | "poly";
    gamma?: number;
    degree?: number;
  } = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 1.0;
    this.degree = options.degree ?? 3;
  }

  private kernelFunc(x1: Float64Array, x2: Float64Array): number {
    switch (this.kernel) {
      case "linear": {
        let s = 0;
        for (let i = 0; i < x1.length; i++) s += (x1[i] ?? 0) * (x2[i] ?? 0);
        return s;
      }
      case "poly": {
        let s = 0;
        for (let i = 0; i < x1.length; i++) s += (x1[i] ?? 0) * (x2[i] ?? 0);
        return Math.pow(s + 1, this.degree);
      }
      default: { // rbf
        let d = 0;
        for (let i = 0; i < x1.length; i++) { const diff = (x1[i] ?? 0) - (x2[i] ?? 0); d += diff * diff; }
        return Math.exp(-this.gamma * d);
      }
    }
  }

  private buildKernelMatrix(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const m = X1.length; const n = X2.length;
    return Array.from({ length: m }, (_, i) =>
      Float64Array.from({ length: n }, (__, j) => this.kernelFunc(X1[i]!, X2[j]!))
    );
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const K = this.buildKernelMatrix(X, X);

    // Add alpha to diagonal (K + alpha*I)
    for (let i = 0; i < n; i++) K[i]![i] = (K[i]?.[i] ?? 0) + this.alpha;

    // Solve via Cholesky or simple Gaussian elimination
    const nOut = Y[0]?.length ?? 1;
    this.dualCoef_ = Array.from({ length: nOut }, (_, o) => {
      const rhs = Float64Array.from({ length: n }, (__, i) => Y[i]?.[o] ?? 0);
      return this.solveLinear(K, rhs);
    });

    this.XTrain_ = X;
    this.fitted = true;
    return this;
  }

  private solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
    const n = A.length;
    // Gaussian elimination
    const M: Float64Array[] = A.map(row => Float64Array.from(row));
    const rhs = Float64Array.from(b);

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(M[row]?.[col] ?? 0) > Math.abs(M[maxRow]?.[col] ?? 0)) maxRow = row;
      }
      [M[col], M[maxRow]] = [M[maxRow]!, M[col]!];
      const tmp = rhs[col] ?? 0; rhs[col] = rhs[maxRow] ?? 0; rhs[maxRow] = tmp;

      const pivot = M[col]?.[col] ?? 1;
      for (let row = col + 1; row < n; row++) {
        const factor = (M[row]?.[col] ?? 0) / pivot;
        for (let k = col; k < n; k++) {
          M[row]![k] = (M[row]?.[k] ?? 0) - factor * (M[col]?.[k] ?? 0);
        }
        rhs[row] = (rhs[row] ?? 0) - factor * (rhs[col] ?? 0);
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = rhs[i] ?? 0;
      for (let j = i + 1; j < n; j++) sum -= (M[i]?.[j] ?? 0) * (x[j] ?? 0);
      x[i] = sum / (M[i]?.[i] ?? 1);
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("MultiOutputKernelRidge not fitted");
    const K = this.buildKernelMatrix(X, this.XTrain_);
    const nOut = this.dualCoef_.length;
    return X.map((_, i) => {
      const out = new Float64Array(nOut);
      for (let o = 0; o < nOut; o++) {
        let pred = 0;
        for (let j = 0; j < this.XTrain_.length; j++) {
          pred += (K[i]?.[j] ?? 0) * (this.dualCoef_[o]?.[j] ?? 0);
        }
        out[o] = pred;
      }
      return out;
    });
  }
}

/**
 * Kernel ridge regression with Nyström approximation for scalability.
 */
export class NystromKernelRidge {
  private alpha: number;
  private gamma: number;
  private nComponents: number;
  private landmarks_: Float64Array[] = [];
  private weights_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { alpha?: number; gamma?: number; nComponents?: number } = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.gamma = options.gamma ?? 1.0;
    this.nComponents = options.nComponents ?? 100;
  }

  private rbf(x1: Float64Array, x2: Float64Array): number {
    let d = 0;
    for (let i = 0; i < x1.length; i++) { const diff = (x1[i] ?? 0) - (x2[i] ?? 0); d += diff * diff; }
    return Math.exp(-this.gamma * d);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const m = Math.min(this.nComponents, n);

    // Select landmarks randomly
    const idx = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, m);
    this.landmarks_ = idx.map(i => X[i]!);

    // Build feature matrix Phi (n x m)
    const Phi: Float64Array[] = Array.from({ length: n }, (_, i) =>
      Float64Array.from({ length: m }, (__, j) => this.rbf(X[i]!, this.landmarks_[j]!))
    );

    // Solve (Phi^T Phi + alpha*I) w = Phi^T y
    const PtP = Array.from({ length: m }, (_, i) => {
      const row = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += (Phi[k]?.[i] ?? 0) * (Phi[k]?.[j] ?? 0);
        row[j] = s + (i === j ? this.alpha : 0);
      }
      return row;
    });

    const Pty = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (Phi[i]?.[j] ?? 0) * (y[i] ?? 0);
      Pty[j] = s;
    }

    // Simple solve
    for (let col = 0; col < m; col++) {
      const pivot = PtP[col]?.[col] ?? 1;
      for (let row = col + 1; row < m; row++) {
        const factor = (PtP[row]?.[col] ?? 0) / pivot;
        for (let k = col; k < m; k++) {
          PtP[row]![k] = (PtP[row]?.[k] ?? 0) - factor * (PtP[col]?.[k] ?? 0);
        }
        Pty[row] = (Pty[row] ?? 0) - factor * (Pty[col] ?? 0);
      }
    }

    this.weights_ = new Float64Array(m);
    for (let i = m - 1; i >= 0; i--) {
      let sum = Pty[i] ?? 0;
      for (let j = i + 1; j < m; j++) sum -= (PtP[i]?.[j] ?? 0) * (this.weights_[j] ?? 0);
      this.weights_[i] = sum / (PtP[i]?.[i] ?? 1);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted) throw new NotFittedError("NystromKernelRidge not fitted");
    const m = this.landmarks_.length;
    return Float64Array.from(X, x => {
      let pred = 0;
      for (let j = 0; j < m; j++) {
        pred += this.rbf(x, this.landmarks_[j]!) * (this.weights_[j] ?? 0);
      }
      return pred;
    });
  }
}
