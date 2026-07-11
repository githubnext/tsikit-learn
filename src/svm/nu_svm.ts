/**
 * NuSVC and NuSVR: nu-parameterized support vector machines.
 * Mirrors sklearn.svm.NuSVC and sklearn.svm.NuSVR.
 */

import { NotFittedError } from "../exceptions.js";

function rbfKernel(a: Float64Array, b: Float64Array, gamma: number): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.exp(-gamma * d);
}

function linearKernel(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

function applyKernel(
  a: Float64Array,
  b: Float64Array,
  kernel: "rbf" | "linear" | "poly",
  gamma: number,
  degree: number,
  coef0: number,
): number {
  if (kernel === "rbf") return rbfKernel(a, b, gamma);
  if (kernel === "poly") return (gamma * linearKernel(a, b) + coef0) ** degree;
  return linearKernel(a, b);
}

export interface NuSVCOptions {
  nu?: number;
  kernel?: "rbf" | "linear" | "poly";
  degree?: number;
  gamma?: number | "scale" | "auto";
  coef0?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * Nu-Support Vector Classification.
 * Mirrors sklearn.svm.NuSVC.
 * nu controls an upper bound on the fraction of training errors and a lower
 * bound on the fraction of support vectors.
 */
export class NuSVC {
  nu: number;
  kernel: "rbf" | "linear" | "poly";
  degree: number;
  gamma: number | "scale" | "auto";
  coef0: number;
  maxIter: number;
  tol: number;

  supportVectors_: Float64Array[] | null = null;
  dualCoef_: Float64Array | null = null;
  intercept_: number = 0;
  classes_: Int32Array | null = null;
  gamma_: number = 1;

  constructor(options: NuSVCOptions = {}) {
    this.nu = options.nu ?? 0.5;
    this.kernel = options.kernel ?? "rbf";
    this.degree = options.degree ?? 3;
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    this.classes_ = Int32Array.from(new Set(Array.from(y))).sort();
    if (this.gamma === "scale") {
      let varSum = 0;
      for (const xi of X)
        for (let j = 0; j < p; j++) varSum += (xi[j] ?? 0) ** 2;
      this.gamma_ = p > 0 ? 1 / (p * (varSum / (n * p) || 1)) : 1;
    } else if (this.gamma === "auto") {
      this.gamma_ = p > 0 ? 1 / p : 1;
    } else {
      this.gamma_ = this.gamma;
    }

    // Simplified SMO for binary nu-SVC (nu controls C upper bound)
    const C = 1 / (this.nu * n);
    const alpha = new Float64Array(n);
    const label = new Float64Array(n).map((_, i) =>
      (y[i] ?? 0) === (this.classes_![0] ?? 0) ? -1 : 1,
    );

    // Initialize alphas uniformly so sum(alpha*y)=0
    for (let i = 0; i < n; i++) alpha[i] = C / 2;

    // SMO iterations
    for (let iter = 0; iter < this.maxIter; iter++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        let ei = -label[i]!;
        for (let k = 0; k < n; k++) {
          ei +=
            (alpha[k] ?? 0) *
            (label[k] ?? 0) *
            applyKernel(
              X[i]!,
              X[k]!,
              this.kernel,
              this.gamma_,
              this.degree,
              this.coef0,
            );
        }
        if (
          (label[i]! * ei < -this.tol && alpha[i]! < C) ||
          (label[i]! * ei > this.tol && alpha[i]! > 0)
        ) {
          const j = (i + 1) % n;
          const ej = -label[j]!;
          const kii = applyKernel(
            X[i]!,
            X[i]!,
            this.kernel,
            this.gamma_,
            this.degree,
            this.coef0,
          );
          const kjj = applyKernel(
            X[j]!,
            X[j]!,
            this.kernel,
            this.gamma_,
            this.degree,
            this.coef0,
          );
          const kij = applyKernel(
            X[i]!,
            X[j]!,
            this.kernel,
            this.gamma_,
            this.degree,
            this.coef0,
          );
          const eta = kii + kjj - 2 * kij;
          if (eta <= 0) continue;
          const alphaJOld = alpha[j]!;
          const alphaIOld = alpha[i]!;
          alpha[j] = alphaJOld + (label[j]! * (ei - ej)) / eta;
          alpha[j] = Math.max(0, Math.min(C, alpha[j]!));
          alpha[i] =
            alphaIOld + label[i]! * label[j]! * (alphaJOld - alpha[j]!);
          alpha[i] = Math.max(0, Math.min(C, alpha[i]!));
          if (Math.abs((alpha[j] ?? 0) - alphaJOld) > 1e-5) changed++;
        }
      }
      if (changed === 0) break;
    }

    // Collect support vectors
    const svIdx: number[] = [];
    for (let i = 0; i < n; i++) if ((alpha[i] ?? 0) > 1e-5) svIdx.push(i);
    this.supportVectors_ = svIdx.map((i) => X[i]!);
    this.dualCoef_ = new Float64Array(
      svIdx.map((i) => (alpha[i] ?? 0) * (label[i] ?? 0)),
    );

    // Compute intercept from margin support vectors
    let b = 0;
    let cnt = 0;
    for (const i of svIdx) {
      let s = 0;
      for (let k = 0; k < svIdx.length; k++) {
        s +=
          (this.dualCoef_[k] ?? 0) *
          applyKernel(
            this.supportVectors_[k]!,
            X[i]!,
            this.kernel,
            this.gamma_,
            this.degree,
            this.coef0,
          );
      }
      b += label[i]! - s;
      cnt++;
    }
    this.intercept_ = cnt > 0 ? b / cnt : 0;
    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    if (!this.supportVectors_ || !this.dualCoef_)
      throw new NotFittedError("NuSVC");
    return new Float64Array(
      X.map((xi) => {
        let s = this.intercept_;
        for (let k = 0; k < this.supportVectors_!.length; k++) {
          s +=
            (this.dualCoef_![k] ?? 0) *
            applyKernel(
              this.supportVectors_![k]!,
              xi,
              this.kernel,
              this.gamma_,
              this.degree,
              this.coef0,
            );
        }
        return s;
      }),
    );
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.classes_) throw new NotFittedError("NuSVC");
    const d = this.decisionFunction(X);
    return new Int32Array(
      d.map((v) =>
        v >= 0 ? (this.classes_![1] ?? 1) : (this.classes_![0] ?? 0),
      ),
    );
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++)
      if ((pred[i] ?? 0) === (y[i] ?? 0)) correct++;
    return correct / y.length;
  }
}

export interface NuSVROptions {
  nu?: number;
  C?: number;
  kernel?: "rbf" | "linear" | "poly";
  degree?: number;
  gamma?: number | "scale" | "auto";
  coef0?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * Nu-Support Vector Regression.
 * Mirrors sklearn.svm.NuSVR.
 */
export class NuSVR {
  nu: number;
  C: number;
  kernel: "rbf" | "linear" | "poly";
  degree: number;
  gamma: number | "scale" | "auto";
  coef0: number;
  maxIter: number;
  tol: number;

  supportVectors_: Float64Array[] | null = null;
  dualCoef_: Float64Array | null = null;
  intercept_: number = 0;
  gamma_: number = 1;

  constructor(options: NuSVROptions = {}) {
    this.nu = options.nu ?? 0.5;
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.degree = options.degree ?? 3;
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.maxIter = options.maxIter ?? 500;
    this.tol = options.tol ?? 1e-3;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    if (this.gamma === "scale") {
      let varSum = 0;
      for (const xi of X)
        for (let j = 0; j < p; j++) varSum += (xi[j] ?? 0) ** 2;
      this.gamma_ = p > 0 ? 1 / (p * (varSum / (n * p) || 1)) : 1;
    } else if (this.gamma === "auto") {
      this.gamma_ = p > 0 ? 1 / p : 1;
    } else {
      this.gamma_ = this.gamma;
    }

    // epsilon-insensitive regression using nu: epsilon = nu * sigma(y)
    const yMean = y.reduce((s, v) => s + v, 0) / n;
    let yVar = 0;
    for (const v of y) yVar += (v - yMean) ** 2;
    const epsilon = this.nu * Math.sqrt(yVar / n);

    const alpha = new Float64Array(n); // dual coefficients
    const C = this.C;
    const eps = epsilon;

    // Simplified gradient descent for SVR dual
    for (let iter = 0; iter < this.maxIter; iter++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        let fi = 0;
        for (let k = 0; k < n; k++) {
          fi +=
            (alpha[k] ?? 0) *
            applyKernel(
              X[i]!,
              X[k]!,
              this.kernel,
              this.gamma_,
              this.degree,
              this.coef0,
            );
        }
        const ri = fi - (y[i] ?? 0);
        const grad = ri > eps ? ri - eps : ri < -eps ? ri + eps : 0;
        const step = Math.min(
          Math.abs(grad) * 0.01,
          C - Math.abs(alpha[i] ?? 0),
        );
        const update = grad > 0 ? -step : step;
        if (Math.abs(update) > 1e-6) {
          alpha[i]! += update;
          alpha[i] = Math.max(-C, Math.min(C, alpha[i]!));
          changed++;
        }
      }
      if (changed === 0) break;
    }

    const svIdx: number[] = [];
    for (let i = 0; i < n; i++)
      if (Math.abs(alpha[i] ?? 0) > 1e-5) svIdx.push(i);
    this.supportVectors_ = svIdx.map((i) => X[i]!);
    this.dualCoef_ = new Float64Array(svIdx.map((i) => alpha[i] ?? 0));

    // Compute intercept
    let b = 0;
    let cnt = 0;
    for (const i of svIdx) {
      let fi = 0;
      for (let k = 0; k < svIdx.length; k++) {
        fi +=
          (this.dualCoef_[k] ?? 0) *
          applyKernel(
            this.supportVectors_![k]!,
            X[i]!,
            this.kernel,
            this.gamma_,
            this.degree,
            this.coef0,
          );
      }
      b += (y[i] ?? 0) - fi;
      cnt++;
    }
    this.intercept_ = cnt > 0 ? b / cnt : 0;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.supportVectors_ || !this.dualCoef_)
      throw new NotFittedError("NuSVR");
    return new Float64Array(
      X.map((xi) => {
        let s = this.intercept_;
        for (let k = 0; k < this.supportVectors_!.length; k++) {
          s +=
            (this.dualCoef_![k] ?? 0) *
            applyKernel(
              this.supportVectors_![k]!,
              xi,
              this.kernel,
              this.gamma_,
              this.degree,
              this.coef0,
            );
        }
        return s;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}
