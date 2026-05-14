/**
 * Support Vector Classifier and Regressor.
 * Mirrors sklearn.svm.SVC and SVR.
 * Uses a simplified SMO (Sequential Minimal Optimization) for binary SVC.
 */

import { NotFittedError } from "../exceptions.js";

function rbfKernel(
  a: Float64Array,
  b: Float64Array,
  gamma: number,
): number {
  let dist2 = 0;
  for (let i = 0; i < a.length; i++) {
    dist2 += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  }
  return Math.exp(-gamma * dist2);
}

function linearKernel(a: Float64Array, b: Float64Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return dot;
}

function polyKernel(
  a: Float64Array,
  b: Float64Array,
  degree: number,
  coef0: number,
): number {
  let dot = coef0;
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return dot ** degree;
}

export class SVC {
  C: number;
  kernel: string;
  degree: number;
  gamma: number | "scale" | "auto";
  coef0: number;
  tol: number;
  maxIter: number;

  alpha_: Float64Array | null = null;
  b_: number = 0;
  supportVectors_: Float64Array[] | null = null;
  supportLabels_: Float64Array | null = null;
  classes_: Float64Array | null = null;

  private _gamma: number = 1;

  constructor(
    options: {
      C?: number;
      kernel?: string;
      degree?: number;
      gamma?: number | "scale" | "auto";
      coef0?: number;
      tol?: number;
      maxIter?: number;
    } = {},
  ) {
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.degree = options.degree ?? 3;
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0.0;
    this.tol = options.tol ?? 1e-3;
    this.maxIter = options.maxIter ?? 1000;
  }

  private _kernelFn(a: Float64Array, b: Float64Array): number {
    if (this.kernel === "linear") return linearKernel(a, b);
    if (this.kernel === "poly") return polyKernel(a, b, this.degree, this.coef0);
    return rbfKernel(a, b, this._gamma);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);

    // Compute gamma
    if (this.gamma === "scale") {
      let varSum = 0;
      for (let j = 0; j < p; j++) {
        let mean = 0;
        for (let i = 0; i < n; i++) mean += (X[i] ?? new Float64Array(p))[j] ?? 0;
        mean /= n;
        for (let i = 0; i < n; i++) varSum += ((X[i] ?? new Float64Array(p))[j] ?? 0 - mean) ** 2;
      }
      this._gamma = p > 0 && varSum > 0 ? 1 / (p * varSum / (n * p)) : 1;
    } else if (this.gamma === "auto") {
      this._gamma = p > 0 ? 1 / p : 1;
    } else {
      this._gamma = this.gamma;
    }

    // Map to ±1
    const posClass = uniqueClasses[uniqueClasses.length - 1] ?? 1;
    const yLabels = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      yLabels[i] = (y[i] ?? 0) === posClass ? 1 : -1;
    }

    // SMO-lite
    const alpha = new Float64Array(n);
    const b = 0;

    // Compute kernel matrix
    const K: number[][] = [];
    for (let i = 0; i < n; i++) {
      K[i] = [];
      for (let j = 0; j < n; j++) {
        (K[i] as number[])[j] = this._kernelFn(
          X[i] ?? new Float64Array(p),
          X[j] ?? new Float64Array(p),
        );
      }
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      const numChanged = 0;

      for (let i = 0; i < n; i++) {
        // Compute decision value
        let fi = -b;
        for (let k = 0; k < n; k++) {
          fi += (alpha[k] ?? 0) * (yLabels[k] ?? 0) * ((K[i] as number[])[k] ?? 0);
        }
        const Ei = fi - (yLabels[i] ?? 0);

        if (
          ((yLabels[i] ?? 0) * Ei < -this.tol && (alpha[i] ?? 0) < this.C) ||
          ((yLabels[i] ?? 0) * Ei > this.tol && (alpha[i] ?? 0) > 0)
        ) {
          // Pick j randomly
          let j = Math.floor(Math.random() * n);
          if (j === i) j = (j + 1) % n;

          let fj = -b;
          for (let k = 0; k < n; k++) {
            fj += (alpha[k] ?? 0) * (yLabels[k] ?? 0) * ((K[j] as number[])[k] ?? 0);
          }
          const Ej = fj - (yLabels[j] ?? 0);

          const alphaIOld = alpha[i] ?? 0;
          const alphaJOld = alpha[j] ?? 0;

          // Compute bounds
          let L: number;
          let H: number;
          if ((yLabels[i] ?? 0) !== (yLabels[j] ?? 0)) {
            L = Math.max(0, alphaJOld - alphaIOld);
            H = Math.min(this.C, this.C + alphaJOld - alphaIOld);
          } else {
            L = Math.max(0, alphaIOld + alphaJOld - this.C);
            H = Math.min(this.C, alphaIOld + alphaJOld);
          }
          if (L >= H) continue;

          const eta =
            2 * ((K[i] as number[])[j] ?? 0) -
            ((K[i] as number[])[i] ?? 0) -
            ((K[j] as number[])[j] ?? 0);
          if (eta >= 0) continue;

          let alphaJNew = alphaJOld - (yLabels[j] ?? 0) * (Ei - Ej) / eta;
          alphaJNew = Math.min(H, Math.max(L, alphaJNew));
          if (Math.abs(alphaJNew - alphaJOld) < 1e-5) continue;

          alpha[j] = alphaJNew;
          alpha[i] =
            alphaIOld +
            (yLabels[i] ?? 0) * (yLabels[j] ?? 0) * (alphaJOld - alphaJNew);

          // Update b
          const b1 =
            b +
            Ei +
            (yLabels[i] ?? 0) * ((alpha[i] ?? 0) - alphaIOld) * ((K[i] as number[])[i] ?? 0) +
            (yLabels[j] ?? 0) * ((alpha[j] ?? 0) - alphaJOld) * ((K[i] as number[])[j] ?? 0);
          const b2 =
            b +
            Ej +
            (yLabels[i] ?? 0) * ((alpha[i] ?? 0) - alphaIOld) * ((K[i] as number[])[j] ?? 0) +
            (yLabels[j] ?? 0) * ((alpha[j] ?? 0) - alphaJOld) * ((K[j] as number[])[j] ?? 0);

          if ((alpha[i] ?? 0) > 0 && (alpha[i] ?? 0) < this.C) b = b1;
          else if ((alpha[j] ?? 0) > 0 && (alpha[j] ?? 0) < this.C) b = b2;
          else b = (b1 + b2) / 2;

          numChanged++;
        }
      }

      if (numChanged === 0) break;
    }

    // Store support vectors
    const svIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((alpha[i] ?? 0) > 1e-5) svIdx.push(i);
    }

    this.alpha_ = new Float64Array(svIdx.map((i) => alpha[i] ?? 0));
    this.supportVectors_ = svIdx.map((i) => X[i] ?? new Float64Array(p));
    this.supportLabels_ = new Float64Array(svIdx.map((i) => yLabels[i] ?? 0));
    this.b_ = b;

    return this;
  }

  decision_function(X: Float64Array[]): Float64Array {
    if (this.alpha_ === null) throw new NotFittedError("SVC");
    const sv = this.supportVectors_ as Float64Array[];
    const svLabels = this.supportLabels_ as Float64Array;
    return new Float64Array(
      X.map((xi) => {
        let val = -this.b_;
        for (let k = 0; k < sv.length; k++) {
          val +=
            (this.alpha_![k] ?? 0) *
            (svLabels[k] ?? 0) *
            this._kernelFn(xi, sv[k] ?? new Float64Array(0));
        }
        return val;
      }),
    );
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("SVC");
    const classes = this.classes_;
    const dv = this.decision_function(X);
    const posClass = classes[classes.length - 1] ?? 1;
    const negClass = classes[0] ?? 0;
    return new Float64Array(dv.map((v) => (v >= 0 ? posClass : negClass)));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class SVR {
  C: number;
  kernel: string;
  degree: number;
  gamma: number | "scale" | "auto";
  coef0: number;
  epsilon: number;
  tol: number;
  maxIter: number;

  alpha_: Float64Array | null = null;
  b_: number = 0;
  supportVectors_: Float64Array[] | null = null;
  dualCoef_: Float64Array | null = null;

  private _gamma: number = 1;

  constructor(
    options: {
      C?: number;
      kernel?: string;
      degree?: number;
      gamma?: number | "scale" | "auto";
      coef0?: number;
      epsilon?: number;
      tol?: number;
      maxIter?: number;
    } = {},
  ) {
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.degree = options.degree ?? 3;
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0.0;
    this.epsilon = options.epsilon ?? 0.1;
    this.tol = options.tol ?? 1e-3;
    this.maxIter = options.maxIter ?? 1000;
  }

  private _kernelFn(a: Float64Array, b: Float64Array): number {
    if (this.kernel === "linear") return linearKernel(a, b);
    if (this.kernel === "poly") return polyKernel(a, b, this.degree, this.coef0);
    return rbfKernel(a, b, this._gamma);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    if (this.gamma === "scale") {
      let varSum = 0;
      for (let j = 0; j < p; j++) {
        let mean = 0;
        for (let i = 0; i < n; i++) mean += (X[i] ?? new Float64Array(p))[j] ?? 0;
        mean /= n;
        for (let i = 0; i < n; i++) varSum += (((X[i] ?? new Float64Array(p))[j] ?? 0) - mean) ** 2;
      }
      this._gamma = p > 0 && varSum > 0 ? n / varSum : 1;
    } else if (this.gamma === "auto") {
      this._gamma = p > 0 ? 1 / p : 1;
    } else {
      this._gamma = this.gamma;
    }

    // Dual form: alpha - alpha* (simplified gradient descent)
    const dualCoef = new Float64Array(n); // alpha_i - alpha_i*
    let b = 0;

    const K: number[][] = [];
    for (let i = 0; i < n; i++) {
      K[i] = [];
      for (let j = 0; j < n; j++) {
        (K[i] as number[])[j] = this._kernelFn(
          X[i] ?? new Float64Array(p),
          X[j] ?? new Float64Array(p),
        );
      }
    }

    const lr = 0.01;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let i = 0; i < n; i++) {
        let pred = b;
        for (let k = 0; k < n; k++) {
          pred += (dualCoef[k] ?? 0) * ((K[i] as number[])[k] ?? 0);
        }
        const err = pred - (y[i] ?? 0);
        let grad = 0;
        if (err > this.epsilon) grad = 1;
        else if (err < -this.epsilon) grad = -1;

        const newCoef = Math.min(
          this.C,
          Math.max(-this.C, (dualCoef[i] ?? 0) - lr * grad),
        );
        const delta = Math.abs(newCoef - (dualCoef[i] ?? 0));
        if (delta > maxDelta) maxDelta = delta;
        dualCoef[i] = newCoef;
      }

      let predSum = 0;
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let k = 0; k < n; k++) {
          pred += (dualCoef[k] ?? 0) * ((K[i] as number[])[k] ?? 0);
        }
        predSum += (y[i] ?? 0) - pred;
      }
      b = predSum / n;

      if (maxDelta < this.tol) break;
    }

    const svIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (Math.abs(dualCoef[i] ?? 0) > 1e-5) svIdx.push(i);
    }

    this.dualCoef_ = new Float64Array(svIdx.map((i) => dualCoef[i] ?? 0));
    this.supportVectors_ = svIdx.map((i) => X[i] ?? new Float64Array(p));
    this.b_ = b;

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.dualCoef_ === null) throw new NotFittedError("SVR");
    const sv = this.supportVectors_ as Float64Array[];
    return new Float64Array(
      X.map((xi) => {
        let val = this.b_;
        for (let k = 0; k < sv.length; k++) {
          val +=
            (this.dualCoef_![k] ?? 0) *
            this._kernelFn(xi, sv[k] ?? new Float64Array(0));
        }
        return val;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}
