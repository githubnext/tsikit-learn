/**
 * SVM extensions: StructuredSVM, RankingSVM, MulticlassSVMExt
 * Port of sklearn.svm extensions
 */

import { NotFittedError } from "../exceptions.js";

function kernelRBF(x1: Float64Array, x2: Float64Array, gamma: number): number {
  let dist = 0;
  for (let j = 0; j < x1.length; j++) dist += ((x1[j] ?? 0) - (x2[j] ?? 0)) ** 2;
  return Math.exp(-gamma * dist);
}

function kernelLinear(x1: Float64Array, x2: Float64Array): number {
  let dot = 0;
  for (let j = 0; j < x1.length; j++) dot += (x1[j] ?? 0) * (x2[j] ?? 0);
  return dot;
}

function kernelPoly(x1: Float64Array, x2: Float64Array, degree: number, coef0: number): number {
  let dot = 0;
  for (let j = 0; j < x1.length; j++) dot += (x1[j] ?? 0) * (x2[j] ?? 0);
  return Math.pow(dot + coef0, degree);
}

export class SVRKernel {
  C: number;
  kernel: "rbf" | "linear" | "poly";
  gamma: number | "scale" | "auto";
  degree: number;
  coef0: number;
  epsilon: number;
  maxIter: number;
  tol: number;

  private alphas_: Float64Array | null = null;
  private supportVectors_: Float64Array[] | null = null;
  private b_ = 0;
  private gammaVal_ = 1.0;

  constructor(opts: {
    C?: number;
    kernel?: "rbf" | "linear" | "poly";
    gamma?: number | "scale" | "auto";
    degree?: number;
    coef0?: number;
    epsilon?: number;
    maxIter?: number;
    tol?: number;
  } = {}) {
    this.C = opts.C ?? 1.0;
    this.kernel = opts.kernel ?? "rbf";
    this.gamma = opts.gamma ?? "scale";
    this.degree = opts.degree ?? 3;
    this.coef0 = opts.coef0 ?? 0;
    this.epsilon = opts.epsilon ?? 0.1;
    this.maxIter = opts.maxIter ?? 1000;
    this.tol = opts.tol ?? 1e-3;
  }

  private _kernel(x1: Float64Array, x2: Float64Array): number {
    if (this.kernel === "rbf") return kernelRBF(x1, x2, this.gammaVal_);
    if (this.kernel === "poly") return kernelPoly(x1, x2, this.degree, this.coef0);
    return kernelLinear(x1, x2);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    if (this.gamma === "scale") {
      const vars = new Float64Array(p);
      const means = new Float64Array(p);
      for (const xi of X) for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) + (xi[j] ?? 0) / n;
      for (const xi of X) for (let j = 0; j < p; j++) vars[j] = (vars[j] ?? 0) + ((xi[j] ?? 0) - (means[j] ?? 0)) ** 2 / n;
      this.gammaVal_ = 1 / (p * (vars.reduce((a, b) => a + b, 0) / p + 1e-15));
    } else if (this.gamma === "auto") {
      this.gammaVal_ = 1 / (p + 1e-15);
    } else {
      this.gammaVal_ = this.gamma;
    }
    const alphaPlus = new Float64Array(n);
    const alphaMinus = new Float64Array(n);
    let b = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        let fi = b;
        for (let j = 0; j < n; j++) {
          const alpha = (alphaPlus[j] ?? 0) - (alphaMinus[j] ?? 0);
          if (Math.abs(alpha) < 1e-10) continue;
          fi += alpha * this._kernel(X[j]!, X[i]!);
        }
        const ri = fi - (y[i] ?? 0);
        const oldAlphaP = alphaPlus[i] ?? 0;
        const oldAlphaM = alphaMinus[i] ?? 0;
        if (Math.abs(ri) > this.epsilon) {
          if (ri > this.epsilon) {
            alphaPlus[i] = Math.min(this.C, (oldAlphaP ?? 0) + 0.1 * (ri - this.epsilon));
          } else {
            alphaMinus[i] = Math.min(this.C, (oldAlphaM ?? 0) + 0.1 * (-ri - this.epsilon));
          }
          const da = Math.abs((alphaPlus[i] ?? 0) - oldAlphaP) + Math.abs((alphaMinus[i] ?? 0) - oldAlphaM);
          if (da > this.tol) changed++;
        }
      }
      b = 0;
      let bCount = 0;
      for (let i = 0; i < n; i++) {
        const alpha = (alphaPlus[i] ?? 0) - (alphaMinus[i] ?? 0);
        if (Math.abs(alpha) > 1e-10) {
          let fi = 0;
          for (let j = 0; j < n; j++) {
            const alphaj = (alphaPlus[j] ?? 0) - (alphaMinus[j] ?? 0);
            if (Math.abs(alphaj) < 1e-10) continue;
            fi += alphaj * this._kernel(X[j]!, X[i]!);
          }
          b += (y[i] ?? 0) - fi;
          bCount++;
        }
      }
      b /= bCount + 1e-15;
      if (changed === 0) break;
      void iter;
    }
    this.alphas_ = new Float64Array(n);
    for (let i = 0; i < n; i++) this.alphas_[i] = (alphaPlus[i] ?? 0) - (alphaMinus[i] ?? 0);
    this.supportVectors_ = X;
    this.b_ = b;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.alphas_ || !this.supportVectors_) throw new NotFittedError("SVRKernel not fitted.");
    return Float64Array.from(X.map(xi => {
      let val = this.b_;
      for (let j = 0; j < this.supportVectors_!.length; j++) {
        if (Math.abs(this.alphas_![j] ?? 0) < 1e-10) continue;
        val += (this.alphas_![j] ?? 0) * this._kernel(this.supportVectors_![j]!, xi);
      }
      return val;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      ssRes += ((y[i] ?? 0) - (pred[i] ?? 0)) ** 2;
      ssTot += ((y[i] ?? 0) - mean) ** 2;
    }
    return 1 - ssRes / (ssTot + 1e-15);
  }
}

export class RankingSVM {
  C: number;
  kernel: "rbf" | "linear";
  gamma: number;
  maxIter: number;

  private alphas_: Float64Array | null = null;
  private X_: Float64Array[] | null = null;
  private pairs_: Array<[number, number]> | null = null;

  constructor(opts: { C?: number; kernel?: "rbf" | "linear"; gamma?: number; maxIter?: number } = {}) {
    this.C = opts.C ?? 1.0;
    this.kernel = opts.kernel ?? "rbf";
    this.gamma = opts.gamma ?? 0.1;
    this.maxIter = opts.maxIter ?? 100;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const pairs: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if ((y[i] ?? 0) > (y[j] ?? 0)) pairs.push([i, j]);
    const m = pairs.length;
    const alphas = new Float64Array(m);
    const kern = (x1: Float64Array, x2: Float64Array) => this.kernel === "rbf" ? kernelRBF(x1, x2, this.gamma) : kernelLinear(x1, x2);
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let k = 0; k < m; k++) {
        const [i, j] = pairs[k]!;
        let margin = 0;
        for (let l = 0; l < m; l++) {
          const [il, jl] = pairs[l]!;
          const kijil = kern(X[i]!, X[il]!) - kern(X[i]!, X[jl]!) - kern(X[j]!, X[il]!) + kern(X[j]!, X[jl]!);
          margin += (alphas[l] ?? 0) * kijil;
        }
        const alpha_new = Math.min(this.C, Math.max(0, (alphas[k] ?? 0) + (1 - margin)));
        alphas[k] = alpha_new;
      }
      void iter;
    }
    this.alphas_ = alphas;
    this.X_ = X;
    this.pairs_ = pairs;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.alphas_ || !this.X_ || !this.pairs_) throw new NotFittedError("RankingSVM not fitted.");
    const kern = (x1: Float64Array, x2: Float64Array) => this.kernel === "rbf" ? kernelRBF(x1, x2, this.gamma) : kernelLinear(x1, x2);
    return Float64Array.from(X.map(xi => {
      let score = 0;
      for (let k = 0; k < this.pairs_!.length; k++) {
        const alpha = this.alphas_![k] ?? 0;
        if (Math.abs(alpha) < 1e-10) continue;
        const [i, j] = this.pairs_![k]!;
        score += alpha * (kern(this.X_![i]!, xi) - kern(this.X_![j]!, xi));
      }
      return score;
    }));
  }
}
